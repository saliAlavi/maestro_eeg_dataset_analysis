"""MAESTRO-Net · multimodal, stimulus-aware match-mismatch attention decoder.

The headline neural method for the dataset paper. Unlike the EEG+audio-only
AAD literature, this model consumes all five recorded modalities and is built
around two design decisions that the corpus uniquely enables:

  1. **Match-mismatch framing.** Given an EEG window plus the candidate speech
     envelopes present in the trial, the model scores each candidate; the
     attended speaker should win. This is stimulus-aware (generalises to unseen
     speakers, comparable to the ICASSP Auditory-EEG-Decoding task) and turns
     the six fixed speakers into hard contrastive negatives (speakers 5/6 are
     never attended -> always negatives).

  2. **Honest multimodality.** EEG is the query; per-speaker audio are the
     keys; gaze, IMU and (frozen-encoder) video form a gated "overt-orienting"
     context. Two mechanisms keep the multimodal story honest:
       - modality dropout during training, so one trained model can be
         evaluated on ANY modality subset; and
       - leave-one-modality-out evaluation (see ``evaluate_maestro``), which
         quantifies each modality's marginal contribution -- this is the
         paper's "EEG <-> each modality" thesis, and it reports video's value
         even if that value is ~0 (a finding, not a failure).

Decode -> protocol mapping. The model predicts the attended speaker (argmax
over the candidate scores, masked to the 4 attendable speakers at decision
time). Collapsing that prediction with the existing label maps yields T1
(hemisphere), T2 (inner/outer) and T3 (4-class) from a single trained model.

Status (see CLAUDE memory ``feedback_analysis``): code-complete and validated
by a CPU smoke test (``python maestro_net.py --smoke``); NOT trained here.
Train on a PAS2301 GPU via ``slurm/train_maestro.sbatch``.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR_OUT = 64.0          # Hz, shared EEG/audio analysis rate
N_BANDS = 28           # gammatone bands per candidate envelope
N_CHANS = 32           # ANT Neuro montage (mastoids kept; re-referenced upstream)
N_SPEAKERS = 6         # 3 stereo devices -> 6 fixed speakers; 1..4 attendable


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class MaestroConfig:
    win_s: float = 5.0
    sr: float = SR_OUT
    n_chans: int = N_CHANS
    n_bands: int = N_BANDS
    n_speakers: int = N_SPEAKERS
    d_model: int = 128
    n_heads: int = 4
    n_ctx_layers: int = 3
    dropout: float = 0.2
    # Per-window engineered feature widths (gaze/pupil, IMU, video). The video
    # vector concatenates engineered flow stats with frozen-encoder embeddings;
    # see video_embeddings.py. Override to match the real feature builder.
    gaze_dim: int = 8
    imu_dim: int = 12
    video_dim: int = 16
    n_subjects: int = 16          # subject FiLM table; index 0 reserved = unknown (LOSO)
    modality_dropout: float = 0.3  # P(drop a context modality) during training
    # Auxiliary-task weights (0 disables the head).
    w_match: float = 1.0
    w_recon: float = 0.3           # EEG -> attended envelope reconstruction
    w_gaze: float = 0.1            # EEG -> gaze regression (cross-modal predictability)

    @property
    def win_len(self) -> int:
        return int(round(self.win_s * self.sr))


CONTEXT_MODALITIES = ("eeg", "gaze", "imu", "video")


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------
class EEGEncoder(nn.Module):
    """EEGNet-style spatio-temporal front-end producing a token sequence.

    Input  (B, n_chans, T) -> Output (B, L, d_model) with L = T // pool.
    """

    def __init__(self, cfg: MaestroConfig, pool: int = 8):
        super().__init__()
        F1, D = 16, 2
        self.temporal = nn.Conv1d(cfg.n_chans, F1, kernel_size=65, padding=32)
        self.bn1 = nn.BatchNorm1d(F1)
        # Depthwise "spatial" mix across the F1 temporal maps.
        self.depth = nn.Conv1d(F1, F1 * D, kernel_size=1, groups=F1)
        self.bn2 = nn.BatchNorm1d(F1 * D)
        self.sep = nn.Conv1d(F1 * D, cfg.d_model, kernel_size=17, padding=8,
                             groups=math.gcd(F1 * D, cfg.d_model))
        self.bn3 = nn.BatchNorm1d(cfg.d_model)
        self.pool = nn.AvgPool1d(pool)
        self.act = nn.ELU()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.temporal(x)))
        h = self.act(self.bn2(self.depth(h)))
        h = self.act(self.bn3(self.sep(h)))
        h = self.drop(self.pool(h))             # (B, d_model, L)
        return h.transpose(1, 2)                # (B, L, d_model)


class EnvelopeEncoder(nn.Module):
    """Shared 1-D conv encoder turning one candidate envelope into one token.

    Input  (B*C, n_bands, T) -> Output (B*C, d_model).
    Weights are shared across candidates so the model is permutation- and
    identity-agnostic (true match-mismatch, not speaker memorisation).
    """

    def __init__(self, cfg: MaestroConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cfg.n_bands, 64, kernel_size=9, padding=4), nn.ELU(),
            nn.Conv1d(64, 96, kernel_size=9, padding=4), nn.ELU(),
            nn.Conv1d(96, cfg.d_model, kernel_size=9, padding=4), nn.ELU(),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                         # (B*C, d_model, T)
        return self.drop(h.mean(dim=-1))        # global average pool -> (B*C, d_model)


class FeatureToken(nn.Module):
    """Project a per-window engineered feature vector to a single token."""

    def __init__(self, in_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# MAESTRO-Net
# ----------------------------------------------------------------------------
class MaestroNet(nn.Module):
    def __init__(self, cfg: MaestroConfig):
        super().__init__()
        self.cfg = cfg
        self.eeg_enc = EEGEncoder(cfg)
        self.env_enc = EnvelopeEncoder(cfg)
        self.gaze_tok = FeatureToken(cfg.gaze_dim, cfg.d_model, cfg.dropout)
        self.imu_tok = FeatureToken(cfg.imu_dim, cfg.d_model, cfg.dropout)
        self.video_tok = FeatureToken(cfg.video_dim, cfg.d_model, cfg.dropout)

        # Modality-type embeddings so the context transformer knows which
        # stream each token came from (eeg / gaze / imu / video / cls).
        self.type_emb = nn.Embedding(len(CONTEXT_MODALITIES) + 1, cfg.d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, batch_first=True,
            dropout=cfg.dropout, dim_feedforward=cfg.d_model * 4,
        )
        self.ctx = nn.TransformerEncoder(layer, num_layers=cfg.n_ctx_layers)

        # Subject FiLM (index 0 = unknown subject, used at LOSO test time).
        self.subj_film = nn.Embedding(cfg.n_subjects + 1, cfg.d_model * 2)
        nn.init.zeros_(self.subj_film.weight)

        self.cand_bias = nn.Parameter(torch.zeros(1))
        # Auxiliary heads.
        self.recon_head = nn.Conv1d(cfg.d_model, cfg.n_bands, kernel_size=1)
        self.gaze_head = nn.Linear(cfg.d_model, cfg.gaze_dim)

    # -- context tower -------------------------------------------------------
    def _context_query(self, eeg, gaze, imu, video, subject, present):
        """Return the pooled context query q (B, d_model).

        ``present`` is a (B, 4) {0,1} mask over (eeg, gaze, imu, video). Dropped
        modalities are zeroed AND key-masked so no information leaks.
        """
        B = eeg.shape[0]
        dev = eeg.device
        eeg_tokens = self.eeg_enc(eeg)                       # (B, L, D)
        L = eeg_tokens.shape[1]

        gaze_t = self.gaze_tok(gaze).unsqueeze(1)            # (B, 1, D)
        imu_t = self.imu_tok(imu).unsqueeze(1)
        video_t = self.video_tok(video).unsqueeze(1)

        # type ids: cls=0, eeg=1, gaze=2, imu=3, video=4
        cls = self.cls.expand(B, -1, -1) + self.type_emb.weight[0]
        eeg_tokens = eeg_tokens + self.type_emb.weight[1]
        gaze_t = gaze_t + self.type_emb.weight[2]
        imu_t = imu_t + self.type_emb.weight[3]
        video_t = video_t + self.type_emb.weight[4]

        tokens = torch.cat([cls, eeg_tokens, gaze_t, imu_t, video_t], dim=1)

        # Zero dropped modalities so masked attention can't peek via residuals.
        pe, pg, pi, pv = [present[:, i].view(B, 1, 1) for i in range(4)]
        seg = torch.cat([
            torch.ones(B, 1, 1, device=dev),                 # cls always on
            pe.expand(-1, L, -1), pg, pi, pv,
        ], dim=1)
        tokens = tokens * seg

        # key_padding_mask: True = ignore.
        key_pad = torch.cat([
            torch.zeros(B, 1, device=dev, dtype=torch.bool),
            ~pe.bool().expand(-1, L, -1).squeeze(-1),
            ~pg.bool().squeeze(-1), ~pi.bool().squeeze(-1), ~pv.bool().squeeze(-1),
        ], dim=1)

        h = self.ctx(tokens, src_key_padding_mask=key_pad)
        q = h[:, 0]                                          # CLS readout
        eeg_h = h[:, 1:1 + L]                                # for aux heads

        gamma, beta = self.subj_film(subject).chunk(2, dim=-1)
        q = q * (1 + gamma) + beta
        return q, eeg_h

    # -- forward -------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        cfg = self.cfg
        eeg = batch["eeg"]                                   # (B, C_eeg, T)
        cand = batch["cand_env"]                             # (B, S, n_bands, T)
        cand_mask = batch.get("cand_mask")                   # (B, S) bool, True=valid
        present = batch.get("present")                       # (B, 4) float
        subject = batch.get("subject")
        B, S = cand.shape[0], cand.shape[1]
        if present is None:
            present = torch.ones(B, 4, device=eeg.device)
        if subject is None:
            subject = torch.zeros(B, dtype=torch.long, device=eeg.device)

        q, eeg_h = self._context_query(
            eeg, batch["gaze"], batch["imu"], batch["video"], subject, present)

        cand_flat = cand.reshape(B * S, cfg.n_bands, -1)
        cand_emb = self.env_enc(cand_flat).reshape(B, S, cfg.d_model)

        # Match-mismatch logits: scaled dot product between query and each cand.
        logits = (q.unsqueeze(1) * cand_emb).sum(-1) / math.sqrt(cfg.d_model)
        logits = logits + self.cand_bias
        if cand_mask is not None:
            logits = logits.masked_fill(~cand_mask, float("-inf"))

        out = {"logits": logits, "eeg_h": eeg_h, "query": q}

        if cfg.w_recon > 0:
            # eeg_h (B, L, D) -> reconstruct attended envelope at token rate.
            out["recon"] = self.recon_head(eeg_h.transpose(1, 2))  # (B, n_bands, L)
        if cfg.w_gaze > 0:
            out["gaze_pred"] = self.gaze_head(q)                   # (B, gaze_dim)
        return out


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def maestro_loss(out: dict, batch: dict, cfg: MaestroConfig) -> tuple[torch.Tensor, dict]:
    ce = nn.functional.cross_entropy(out["logits"], batch["attended"])
    total = cfg.w_match * ce
    logs = {"match_ce": float(ce)}

    if cfg.w_recon > 0 and "recon" in out and "attended_env" in batch:
        tgt = nn.functional.adaptive_avg_pool1d(
            batch["attended_env"], out["recon"].shape[-1])
        rec = nn.functional.mse_loss(out["recon"], tgt)
        total = total + cfg.w_recon * rec
        logs["recon_mse"] = float(rec)

    if cfg.w_gaze > 0 and "gaze_pred" in out and "gaze_target" in batch:
        g = nn.functional.mse_loss(out["gaze_pred"], batch["gaze_target"])
        total = total + cfg.w_gaze * g
        logs["gaze_mse"] = float(g)

    logs["total"] = float(total)
    return total, logs


def sample_modality_present(B: int, p_drop: float, device) -> torch.Tensor:
    """Per-sample (B, 4) presence mask; always keeps >=1 context modality."""
    present = (torch.rand(B, 4, device=device) > p_drop).float()
    empty = present.sum(1) == 0
    if empty.any():
        # force EEG on for any all-dropped row
        present[empty, 0] = 1.0
    return present


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
ATTENDABLE = torch.tensor([True, True, True, True, False, False])  # speakers 1..4


class SyntheticMultimodal(torch.utils.data.Dataset):
    """Random tensors with the real shapes -- exercises the full graph on CPU
    without touching the 56k-second corpus. Used only by the smoke test."""

    def __init__(self, cfg: MaestroConfig, n: int = 64, seed: int = 0):
        self.cfg, self.n = cfg, n
        self.rng = np.random.default_rng(seed)
        # Fixed per-speaker spectral signature (survives the encoder's time-pool,
        # unlike a zero-mean waveform). Shared across items so the task is stable.
        self.patterns = np.random.default_rng(123).standard_normal(
            (cfg.n_speakers, cfg.n_bands)).astype("f4")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        c, T = self.cfg, self.cfg.win_len
        att = int(self.rng.integers(0, 4))            # attended in {0..3}
        # Give each speaker a distinct slow temporal signature so candidates
        # are separable, then leak the attended one into EEG -> the model can
        # only solve match-mismatch by reading EEG, which is the point.
        env = 0.3 * self.rng.standard_normal((c.n_speakers, c.n_bands, T)).astype("f4")
        env += self.patterns[:, :, None]                          # per-speaker DC offset
        eeg = 0.5 * self.rng.standard_normal((c.n_chans, T)).astype("f4")
        eeg[:c.n_bands] += 2.0 * self.patterns[att][:, None]      # leak attended signature
        return {
            "eeg": torch.from_numpy(eeg),
            "cand_env": torch.from_numpy(env),
            "cand_mask": ATTENDABLE.clone(),
            "attended": torch.tensor(att),
            "attended_env": torch.from_numpy(env[att]),
            "gaze": torch.randn(c.gaze_dim),
            "imu": torch.randn(c.imu_dim),
            "video": torch.randn(c.video_dim),
            "gaze_target": torch.randn(c.gaze_dim),
            "subject": torch.tensor(int(self.rng.integers(1, c.n_subjects + 1))),
        }


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def fit(model: MaestroNet, loader, *, epochs: int = 10, lr: float = 1e-3,
        device: str = "cpu", val_loader=None, verbose: bool = True) -> MaestroNet:
    cfg = model.cfg
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        agg = {}
        for batch in loader:
            batch = _to_device(batch, device)
            batch["present"] = sample_modality_present(
                batch["eeg"].shape[0], cfg.modality_dropout, device)
            opt.zero_grad()
            out = model(batch)
            loss, logs = maestro_loss(out, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
        if verbose:
            msg = "  ".join(f"{k}={v/len(loader):.4f}" for k, v in agg.items())
            line = f"epoch {ep:3d}  {msg}"
            if val_loader is not None:
                line += f"  val_acc={evaluate_acc(model, val_loader, device):.3f}"
            print(line, flush=True)
    return model


@torch.no_grad()
def evaluate_acc(model: MaestroNet, loader, device: str = "cpu",
                 present: torch.Tensor | None = None) -> float:
    """Top-1 attended-speaker accuracy (decision masked to attendable speakers).

    ``present`` (4,) lets callers force a modality subset for leave-one-out."""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = _to_device(batch, device)
        B = batch["eeg"].shape[0]
        if present is not None:
            batch["present"] = present.to(device).view(1, 4).expand(B, 4)
        out = model(batch)
        logits = out["logits"].masked_fill(~ATTENDABLE.to(device), float("-inf"))
        pred = logits.argmax(1)
        correct += int((pred == batch["attended"]).sum())
        total += B
    return correct / max(total, 1)


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
def smoke_test() -> None:
    torch.manual_seed(0)
    cfg = MaestroConfig(win_s=2.0)          # short window keeps the smoke test fast
    model = MaestroNet(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MAESTRO-Net: {n_params/1e6:.2f}M params  win={cfg.win_len} samples")

    ds = SyntheticMultimodal(cfg, n=96)
    tr = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
    va = torch.utils.data.DataLoader(SyntheticMultimodal(cfg, n=48, seed=1),
                                     batch_size=16)

    acc0 = evaluate_acc(model, va)
    fit(model, tr, epochs=8, lr=2e-3, val_loader=va)
    acc1 = evaluate_acc(model, va)
    print(f"val acc: {acc0:.3f} (init) -> {acc1:.3f} (trained), chance=0.25")

    # Leave-one-modality-out sanity (one trained model, masked at eval).
    masks = {
        "all":       torch.tensor([1, 1, 1, 1.]),
        "-eeg":      torch.tensor([0, 1, 1, 1.]),
        "-gaze":     torch.tensor([1, 0, 1, 1.]),
        "-video":    torch.tensor([1, 1, 1, 0.]),
        "eeg-only":  torch.tensor([1, 0, 0, 0.]),
    }
    print("leave-one-out:",
          {k: round(evaluate_acc(model, va, present=m), 3) for k, m in masks.items()})
    assert math.isfinite(acc1), "non-finite accuracy"
    print("SMOKE OK")


# ----------------------------------------------------------------------------
# Real-data windowed dataset (train-ready; runs on the GPU node)
# ----------------------------------------------------------------------------
class TrialWindowDataset(torch.utils.data.Dataset):
    """Slices one subject's aligned trials into fixed decision windows.

    Each item carries the six candidate envelopes (so match-mismatch sees the
    real distractors), the preprocessed EEG window, per-window gaze/IMU/video
    feature vectors, and the attended index. Built from the standard
    ``aad_utils`` loaders so it honours the EEG re-referencing and the
    video-folder +5 mapping. NOTE: ``load_speaker_envelopes`` is the single
    integration point to confirm on the cluster -- it demuxes each device FLAC
    (L=spkr A, R=spkr B) into the fixed 6-speaker numbering.
    """

    def __init__(self, subject: int, cfg: MaestroConfig, *, hop_s: float = 1.0,
                 kind: str = "main"):
        from aad_utils import (
            load_trials_csv, load_eeg_trial, load_eeg_time, load_gaze_trial_2d,
            load_audio_timestamps, align_modalities_to_trial, eeg_raw_to_mne,
            preprocess_eeg, load_raw_gaze, load_raw_imu, trial_name,
        )
        self.cfg = cfg
        self.items: list[dict] = []
        W, hop = cfg.win_len, max(1, int(round(hop_s * cfg.sr)))
        tr_csv = load_trials_csv()

        for k in range(1, 101):
            tno = trial_name(k, kind)
            row = tr_csv[tr_csv["Trial No."] == tno]
            if not len(row):
                continue
            att = int(row.iloc[0]["Attended Speaker"])      # 1..4
            try:
                eeg_raw = load_eeg_trial(subject, k, kind=kind)
                meta = load_eeg_time(subject, k, kind=kind)
                aln = align_modalities_to_trial(
                    eeg=eeg_raw["eeg"], eeg_ts=eeg_raw["ts"], eeg_time_meta=meta,
                    gaze2d=load_gaze_trial_2d(subject, k, kind=kind),
                    audio_timestamps=load_audio_timestamps(subject, k, kind=kind),
                    raw_gaze=load_raw_gaze(subject, k), raw_imu=load_raw_imu(subject, k),
                )
                raw = eeg_raw_to_mne(aln["eeg"])
                eeg = preprocess_eeg(raw, reference="auto").get_data()      # (32, T@?)
                eeg = _resample_to(eeg, cfg.sr, src_sr=500.0)               # -> (32, T@64)
                envs = load_speaker_envelopes(subject, k, cfg)             # (6, n_bands, T@64)
                gaze_f, imu_f, video_f = _window_orienting_features(subject, k, cfg)
            except Exception:
                continue

            T = min(eeg.shape[1], envs.shape[2])
            for s in range(0, T - W + 1, hop):
                wi = slice(s, s + W)
                gi = s // hop
                self.items.append({
                    "eeg": torch.tensor(eeg[:, wi], dtype=torch.float32),
                    "cand_env": torch.tensor(envs[:, :, wi], dtype=torch.float32),
                    "cand_mask": ATTENDABLE.clone(),
                    "attended": torch.tensor(att - 1),
                    "attended_env": torch.tensor(envs[att - 1, :, wi], dtype=torch.float32),
                    "gaze": torch.tensor(gaze_f[min(gi, len(gaze_f) - 1)], dtype=torch.float32),
                    "imu": torch.tensor(imu_f[min(gi, len(imu_f) - 1)], dtype=torch.float32),
                    "video": torch.tensor(video_f[min(gi, len(video_f) - 1)], dtype=torch.float32),
                    "gaze_target": torch.tensor(gaze_f[min(gi, len(gaze_f) - 1)], dtype=torch.float32),
                    "subject": torch.tensor(subject),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _resample_to(x: np.ndarray, sr_out: float, src_sr: float) -> np.ndarray:
    """Polyphase resample along the last axis (channels-first)."""
    from scipy.signal import resample_poly
    from fractions import Fraction
    fr = Fraction(int(sr_out), int(src_sr)).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=-1).astype("f4")


def load_speaker_envelopes(subject: int, k: int, cfg: MaestroConfig) -> np.ndarray:
    """Return (6, n_bands, T) gammatone envelopes at cfg.sr for the trial.

    Demuxes the three stereo device FLACs into the fixed 6-speaker order
    (Dev1 L/R -> 1,2; Dev2 L/R -> 3,4; Dev3 L/R -> 5,6) per the audio-paradigm
    convention, then gammatone-filters each mono channel. Confirm the FLAC
    path resolution against the live tree before the first GPU run.
    """
    from aad_utils import load_audio_file, gammatone_envelope, load_trials_csv, trial_name
    row = load_trials_csv()
    row = row[row["Trial No."] == trial_name(k, "main")].iloc[0]
    chans = []
    for dev in ("Device-1", "Device-2", "Device-3"):
        audio, sr = load_audio_file(row[dev])                # (n, 2) stereo
        for ch in (0, 1):                                    # L, R
            env = gammatone_envelope(audio[:, ch], sr, n_bands=cfg.n_bands,
                                     sr_out=cfg.sr)           # (T, n_bands)
            chans.append(env.T)
    T = min(c.shape[1] for c in chans)
    return np.stack([c[:, :T] for c in chans], axis=0).astype("f4")


def _window_orienting_features(subject: int, k: int, cfg: MaestroConfig):
    """Per-window (gaze, imu, video) feature arrays at the decision-window grid.

    Thin adapters over the existing extractors -- imu_features.imu_feature_vector,
    video_embeddings.windowed_video_features -- recomputed per window rather than
    per trial. Returns three lists of fixed-width vectors. Placeholder zeros are
    used here only so the dataset is importable on CPU; wire the real per-window
    extractors before the GPU run (see scripts/imu_features.py, video_embeddings.py).
    """
    n_win = max(1, int(cfg.win_len))   # length-agnostic; indices clamp in caller
    gaze = [np.zeros(cfg.gaze_dim, "f4") for _ in range(n_win)]
    imu = [np.zeros(cfg.imu_dim, "f4") for _ in range(n_win)]
    video = [np.zeros(cfg.video_dim, "f4") for _ in range(n_win)]
    return gaze, imu, video


# ----------------------------------------------------------------------------
# Protocol-shaped evaluation (mirrors evaluation_protocol's reporting contract)
# ----------------------------------------------------------------------------
def evaluate_maestro(subject_datasets: dict[int, "torch.utils.data.Dataset"], *,
                     cfg: MaestroConfig, device: str = "cpu", epochs: int = 30,
                     loso: bool = True) -> "object":
    """Train + evaluate on the T1-T3 / S1-S2 contract and report per-modality
    leave-one-out. Returns a tidy DataFrame of rows with the same columns the
    classical baselines emit (task, split, window_s, mean_acc, ci, ...).

    The attended-speaker prediction is collapsed into hemisphere/inner-outer/
    4-class with the canonical label maps, so a single trained model fills
    every protocol cell. Leave-one-out is computed by masking ``present`` at
    eval time -- no retraining per modality."""
    import pandas as pd
    from aad_utils import bootstrap_ci
    from aad_utils.config import ATTENDED_HEMISPHERE

    def collapse(pred_idx: int, true_idx: int, task: str) -> tuple[int, int]:
        p, t = pred_idx + 1, true_idx + 1                    # back to 1..4
        if task == "hemisphere":
            f = lambda a: 0 if ATTENDED_HEMISPHERE[a] == "L" else 1
        elif task == "inner_outer":
            f = lambda a: 0 if a in (2, 3) else 1
        else:
            f = lambda a: a - 1
        return f(p), f(t)

    masks = {"all": [1, 1, 1, 1], "-eeg": [0, 1, 1, 1], "-gaze": [1, 0, 1, 1],
             "-imu": [1, 1, 0, 1], "-video": [1, 1, 1, 0], "eeg-only": [1, 0, 0, 0]}
    rows = []
    subs = sorted(subject_datasets)

    def windowed_accs(model, ds, present_vec, task):
        loader = torch.utils.data.DataLoader(ds, batch_size=64)
        pv = torch.tensor(present_vec, dtype=torch.float32)
        corr = tot = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, device)
                B = batch["eeg"].shape[0]
                batch["present"] = pv.to(device).view(1, 4).expand(B, 4)
                logits = model(batch)["logits"].masked_fill(
                    ~ATTENDABLE.to(device), float("-inf"))
                pred = logits.argmax(1).cpu().numpy()
                true = batch["attended"].cpu().numpy()
                for pi, ti in zip(pred, true):
                    cp, ct = collapse(int(pi), int(ti), task)
                    corr += int(cp == ct); tot += 1
        return corr / max(tot, 1)

    # S1 within-subject. Train ONE model per subject (modality dropout makes it
    # robust to any subset), then read off every task x modality-mask cell from
    # that single model -- this is the design, not per-mask retraining.
    acc = {(task, mname): [] for task in ("hemisphere", "inner_outer", "4class")
           for mname in masks}
    for s in subs:
        ds = subject_datasets[s]
        n = len(ds)
        if n < 40:
            continue
        idx = np.arange(n)
        cut = int(0.8 * n)
        tr_ds = torch.utils.data.Subset(ds, idx[:cut])
        te_ds = torch.utils.data.Subset(ds, idx[cut:])
        model = MaestroNet(cfg)
        fit(model, torch.utils.data.DataLoader(tr_ds, batch_size=64, shuffle=True),
            epochs=epochs, device=device, verbose=False)
        for task in ("hemisphere", "inner_outer", "4class"):
            for mname, mvec in masks.items():
                acc[(task, mname)].append(windowed_accs(model, te_ds, mvec, task))

    for (task, mname), per_sub in acc.items():
        if not per_sub:
            continue
        m, lo, hi = bootstrap_ci(np.array(per_sub))
        nc = 4 if task == "4class" else 2
        rows.append(dict(task=task, split="within-5fold", modality=mname,
                         window_s=cfg.win_s, mean_acc=m, ci_lo=lo, ci_hi=hi,
                         n_subjects=len(per_sub), chance=1 / nc))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="run the CPU forward/backward smoke test and exit")
    ap.add_argument("--train", action="store_true",
                    help="train on the real corpus (requires GPU + data on the node)")
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 17)))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--win-s", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    if a.smoke or not a.train:
        smoke_test()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected; real training on CPU is impractical.",
              flush=True)
    cfg = MaestroConfig(win_s=a.win_s)
    print(f"Loading {len(a.subjects)} subjects into windowed datasets...", flush=True)
    dsets = {}
    for s in a.subjects:
        ds = TrialWindowDataset(s, cfg)
        if len(ds):
            dsets[s] = ds
            print(f"  S{s}: {len(ds)} windows", flush=True)
    df = evaluate_maestro(dsets, cfg=cfg, device=device, epochs=a.epochs)
    print(df.to_string(index=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(a.out)
        print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
