"""Deep-model scaffolds for the AAD pipeline, behind RUN_DEEP flag.

Implemented (code complete; not trained under this CPU iteration):
    1. StimulusReconstructionCNN  — regress 28-band gammatone envelope
       (attended vs unattended) from 32-ch × 2-s EEG windows.
    2. FoundationEEGProbe         — freeze a LaBraM-style encoder and
       linear-probe the attended-speaker label on its embeddings.
    3. CrossModalAttention        — EEG tokens × gaze tokens dual-stream
       transformer for 4-way AAD.

Author note (see CLAUDE memory `feedback_analysis.md`): implemented for
the paper scaffold, not executed here. Run with GPU on PAS2301 once
available.
"""
from __future__ import annotations

import os

import numpy as np

RUN_DEEP = bool(int(os.environ.get("RUN_DEEP", "0")))


# ------------------------------------------------------------------
# 1. Stimulus reconstruction CNN -----------------------------------
# ------------------------------------------------------------------
if RUN_DEEP:
    import torch
    from torch import nn

    class StimulusReconstructionCNN(nn.Module):
        """1-D temporal CNN that reconstructs a 28-band gammatone
        envelope from 32-channel EEG at 64 Hz. The expected input shape
        is (B, 32, T=128) for 2-s windows; output shape (B, 28, T).
        """

        def __init__(self, n_chans: int = 32, n_bands: int = 28):
            super().__init__()
            self.spatial = nn.Conv1d(n_chans, 64, kernel_size=1)
            self.temporal1 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.temporal2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
            self.head = nn.Conv1d(128, n_bands, kernel_size=1)
            self.act = nn.GELU()
            self.norm = nn.LayerNorm(128)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.act(self.spatial(x))
            h = self.act(self.temporal1(h))
            h = self.act(self.temporal2(h))
            h = self.head(h)
            return h

    def train_stim_recon(
        eeg_trials: list[np.ndarray],
        env_trials: list[np.ndarray],
        *, epochs: int = 50, lr: float = 1e-3,
    ) -> "StimulusReconstructionCNN":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StimulusReconstructionCNN().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        def _batch(eegs, envs, win=128):
            for e, v in zip(eegs, envs):
                T = min(len(e), len(v))
                for s in range(0, T - win + 1, win):
                    yield (
                        torch.tensor(e[s:s+win].T, dtype=torch.float32)[None],
                        torch.tensor(v[s:s+win].T, dtype=torch.float32)[None],
                    )

        for ep in range(epochs):
            losses = []
            model.train()
            for x, y in _batch(eeg_trials, env_trials):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                losses.append(float(loss))
            print(f"epoch {ep:3d}  loss={np.mean(losses):.4f}")
        return model

# ------------------------------------------------------------------
# 2. Foundation-model probe ----------------------------------------
# ------------------------------------------------------------------
if RUN_DEEP:
    try:
        from transformers import AutoModel
        _FOUNDATION_BACKBONES = {
            "labram-base": "BAAI/LaBraM-Base",
        }

        class FoundationEEGProbe(nn.Module):
            def __init__(self, backbone: str = "labram-base",
                         n_classes: int = 4):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(
                    _FOUNDATION_BACKBONES[backbone]
                )
                for p in self.encoder.parameters():
                    p.requires_grad = False
                hidden = self.encoder.config.hidden_size
                self.cls = nn.Linear(hidden, n_classes)

            def forward(self, x):
                with torch.no_grad():
                    emb = self.encoder(x).last_hidden_state.mean(dim=1)
                return self.cls(emb)
    except ImportError:
        pass  # transformers not installed on CPU-only nodes

# ------------------------------------------------------------------
# 3. Cross-modal attention (EEG ↔ gaze dual stream) ----------------
# ------------------------------------------------------------------
if RUN_DEEP:
    class CrossModalAttention(nn.Module):
        """Two token streams: (B, N_e, D) EEG tokens, (B, N_g, D) gaze
        tokens. A joint transformer with cross-attention fuses them
        before a 4-class head.
        """

        def __init__(
            self,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            n_classes: int = 4,
            eeg_feat_dim: int = 32,
            gaze_feat_dim: int = 23,
        ):
            super().__init__()
            self.eeg_embed = nn.Linear(eeg_feat_dim, d_model)
            self.gaze_embed = nn.Linear(gaze_feat_dim, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, batch_first=True,
            )
            self.joint = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.cls = nn.Linear(d_model, n_classes)

        def forward(self, eeg_tokens, gaze_tokens):
            e = self.eeg_embed(eeg_tokens)
            g = self.gaze_embed(gaze_tokens)
            tokens = torch.cat([e, g], dim=1)
            h = self.joint(tokens)
            pooled = h.mean(dim=1)
            return self.cls(pooled)


# ------------------------------------------------------------------
# CLI stub ---------------------------------------------------------
# ------------------------------------------------------------------
def main():
    if not RUN_DEEP:
        print("RUN_DEEP=0 — skipping training. Set RUN_DEEP=1 and run on "
              "GPU to execute.")
        return
    print("RUN_DEEP=1 entry point. Wire a loader + training loop here.")


if __name__ == "__main__":
    main()
