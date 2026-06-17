"""MAESTRO-Net -- multimodal, stimulus-aware match-mismatch AAD decoder.

EEG is the query; the six per-speaker envelopes are the keys; gaze / IMU /
video are gated "overt-orienting" context tokens fused by a small transformer.
Four mechanisms make the multimodal story honest and strong:

  * **Match-mismatch** scoring of the real candidate envelopes (speakers 5/6 are
    permanent hard negatives).
  * **Modality dropout** during training, so one model is evaluable on ANY
    modality subset -> leave-one-modality-out is read off a single model.
  * **Subject FiLM** (zero-init, so an unseen LOSO subject defaults to identity).
  * **Adversarial gaze head** (gradient reversal): the EEG query is penalised for
    being gaze-predictable, isolating EEG-beyond-overt-orienting -- the project's
    central scientific question.

Aux losses: InfoNCE EEG<->attended-envelope alignment, and EEG->attended-envelope
reconstruction.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..base import TorchModel
from ..components import EEGEncoder, EnvelopeEncoder, FeatureToken, grad_reverse
from ..factory import MODEL_REGISTRY

CONTEXT = ("eeg", "gaze", "imu", "video")   # present4 order


class _MaestroNet(nn.Module):
    def __init__(self, fd, *, d_model=128, n_heads=4, n_ctx_layers=3, dropout=0.2,
                 n_subjects=16):
        super().__init__()
        self.fd = fd
        self.d_model = d_model
        self.eeg_enc = EEGEncoder(fd["n_chans"], d_model, dropout)
        self.env_enc = EnvelopeEncoder(fd["n_bands"], d_model, dropout)
        self.gaze_tok = FeatureToken(fd["gaze_dim"], d_model, dropout)
        self.imu_tok = FeatureToken(fd["imu_dim"], d_model, dropout)
        self.video_tok = FeatureToken(fd["video_dim"], d_model, dropout)
        self.type_emb = nn.Embedding(len(CONTEXT) + 1, d_model)  # cls + 4
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True,
            dropout=dropout, dim_feedforward=d_model * 4)
        self.ctx = nn.TransformerEncoder(layer, num_layers=n_ctx_layers)
        self.subj_film = nn.Embedding(n_subjects + 1, d_model * 2)
        nn.init.zeros_(self.subj_film.weight)
        self.cand_bias = nn.Parameter(torch.zeros(1))
        self.recon_head = nn.Conv1d(d_model, fd["n_bands"], 1)
        self.gaze_head = nn.Linear(d_model, fd["gaze_dim"])

    def _context_query(self, eeg, gaze, imu, video, subject, present4):
        B = eeg.shape[0]; dev = eeg.device
        eeg_tok = self.eeg_enc(eeg)                       # (B,L,d)
        L = eeg_tok.shape[1]
        gaze_t = self.gaze_tok(gaze).unsqueeze(1)
        imu_t = self.imu_tok(imu).unsqueeze(1)
        video_t = self.video_tok(video).unsqueeze(1)

        cls = self.cls.expand(B, -1, -1) + self.type_emb.weight[0]
        eeg_tok = eeg_tok + self.type_emb.weight[1]
        gaze_t = gaze_t + self.type_emb.weight[2]
        imu_t = imu_t + self.type_emb.weight[3]
        video_t = video_t + self.type_emb.weight[4]
        tokens = torch.cat([cls, eeg_tok, gaze_t, imu_t, video_t], dim=1)

        pe, pg, pi, pv = [present4[:, i].view(B, 1, 1) for i in range(4)]
        seg = torch.cat([torch.ones(B, 1, 1, device=dev),
                         pe.expand(-1, L, -1), pg, pi, pv], dim=1)
        tokens = tokens * seg
        key_pad = torch.cat([
            torch.zeros(B, 1, device=dev, dtype=torch.bool),
            ~pe.bool().expand(-1, L, -1).squeeze(-1),
            ~pg.bool().squeeze(-1), ~pi.bool().squeeze(-1), ~pv.bool().squeeze(-1),
        ], dim=1)
        h = self.ctx(tokens, src_key_padding_mask=key_pad)
        q = h[:, 0]; eeg_h = h[:, 1:1 + L]
        gamma, beta = self.subj_film(subject).chunk(2, dim=-1)
        q = q * (1 + gamma) + beta
        return q, eeg_h

    def forward(self, eeg, cand, gaze, imu, video, subject, present4):
        B, S = cand.shape[0], cand.shape[1]
        q, eeg_h = self._context_query(eeg, gaze, imu, video, subject, present4)
        cand_flat = cand.reshape(B * S, cand.shape[2], cand.shape[3])
        cemb = self.env_enc(cand_flat).reshape(B, S, self.d_model)
        logits = (q.unsqueeze(1) * cemb).sum(-1) / math.sqrt(self.d_model) + self.cand_bias
        return {"logits": logits, "q": q, "eeg_h": eeg_h, "cand_emb": cemb}


@MODEL_REGISTRY.register("maestro")
class MaestroModel(TorchModel):
    name = "maestro"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.d_model = int(cfg.get("d_model", 128))
        self.n_heads = int(cfg.get("n_heads", 4))
        self.n_ctx_layers = int(cfg.get("n_ctx_layers", 3))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.n_subjects = int(cfg.get("n_subjects", 16))
        self.modality_dropout = float(cfg.get("modality_dropout", 0.3))
        self.w_match = float(cfg.get("w_match", 1.0))
        self.w_info = float(cfg.get("w_info", 0.3))
        self.w_recon = float(cfg.get("w_recon", 0.3))
        self.w_adv = float(cfg.get("w_adv_gaze", 0.2))
        self.info_temp = float(cfg.get("info_temp", 0.1))
        self.adv_lambda = float(cfg.get("adv_lambda", 1.0))

    def build_module(self):
        return _MaestroNet(self.fd, d_model=self.d_model, n_heads=self.n_heads,
                           n_ctx_layers=self.n_ctx_layers, dropout=self.dropout,
                           n_subjects=self.n_subjects)

    def _present4(self, batch, override=None, train=False):
        """Build (B,4) presence over [eeg,gaze,imu,video]; EEG always on."""
        B = batch["eeg"].shape[0]; dev = batch["eeg"].device
        base = torch.cat([torch.ones(B, 1, device=dev), batch["present"]], dim=1)
        if override is not None:
            ov = torch.tensor(override, dtype=torch.float32, device=dev).view(1, 4)
            return base * ov
        if train and self.modality_dropout > 0:
            keep = (torch.rand(B, 3, device=dev) > self.modality_dropout).float()
            base[:, 1:] = base[:, 1:] * keep
        return base

    def compute_loss(self, batch):
        present4 = self._present4(batch, train=True)
        out = self.module(batch["eeg"], batch["cand_env"], batch["gaze"],
                          batch["imu"], batch["video"], batch["subject"], present4)
        logits, q, eeg_h, cemb = out["logits"], out["q"], out["eeg_h"], out["cand_emb"]
        att = batch["attended"]; B = att.shape[0]
        ce = nn.functional.cross_entropy(logits, att)
        total = self.w_match * ce
        logs = {"ce": float(ce)}

        # InfoNCE: q vs attended envelope embedding, in-batch negatives.
        att_emb = cemb[torch.arange(B, device=att.device), att]    # (B,d)
        qn = nn.functional.normalize(q, dim=-1)
        an = nn.functional.normalize(att_emb, dim=-1)
        sim = qn @ an.t() / self.info_temp
        info = nn.functional.cross_entropy(sim, torch.arange(B, device=att.device))
        total = total + self.w_info * info; logs["info"] = float(info)

        # Reconstruction of attended envelope from EEG tokens.
        if self.w_recon > 0:
            recon = self.module.recon_head(eeg_h.transpose(1, 2))   # (B,n_bands,L)
            att_env = batch["cand_env"][torch.arange(B, device=att.device), att]
            tgt = nn.functional.adaptive_avg_pool1d(att_env, recon.shape[-1])
            rec = nn.functional.mse_loss(recon, tgt)
            total = total + self.w_recon * rec; logs["recon"] = float(rec)

        # Adversarial gaze: penalise gaze-predictability of the query.
        if self.w_adv > 0:
            gpred = self.module.gaze_head(grad_reverse(q, self.adv_lambda))
            mask = batch["present"][:, 0].view(B, 1)               # gaze present
            if mask.sum() > 0:
                adv = ((gpred - batch["gaze"]) ** 2 * mask).sum() / (mask.sum() * gpred.shape[1])
                total = total + self.w_adv * adv; logs["adv_gaze"] = float(adv)

        with torch.no_grad():
            m = batch["cand_mask"].bool()
            logs["acc"] = float((logits.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean())
        logs["total"] = float(total)
        return total, logs

    def predict_logits(self, batch, present_override=None):
        present4 = self._present4(batch, override=present_override, train=False)
        return self.module(batch["eeg"], batch["cand_env"], batch["gaze"],
                           batch["imu"], batch["video"], batch["subject"],
                           present4)["logits"]

    # Leave-one-modality-out, read off the single trained model.
    LOMO_MASKS = {
        "all": [1, 1, 1, 1], "-gaze": [1, 0, 1, 1], "-imu": [1, 1, 0, 1],
        "-video": [1, 1, 1, 0], "eeg_only": [1, 0, 0, 0], "-eeg": [0, 1, 1, 1],
    }

    def evaluate(self, view, ctx, prefix="test/", present_override=None):
        from ..base import compute_aad_metrics
        fd = getattr(self, "fd", {}) or {}
        task = fd.get("task_type", "speaker"); n_cand = fd.get("n_candidates", 4)
        true = view.as_numpy()["attended"]
        out = {}
        for mname, mvec in self.LOMO_MASKS.items():
            pred = self.predict(view, ctx, present_override=mvec)
            out.update(compute_aad_metrics(pred, true, prefix=f"{prefix}{mname}/",
                                           task_type=task, n_cand=n_cand))
        # Mirror the "all" metrics to the canonical keys the runner aggregates.
        for key in ("acc", "acc_hemisphere", "acc_inner_outer", "chance", "n"):
            v = out.get(f"{prefix}all/{key}")
            if v is not None:
                out[f"{prefix}{key}"] = v
        out.setdefault(f"{prefix}n", len(true))
        return out
