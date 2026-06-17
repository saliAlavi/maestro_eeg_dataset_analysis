"""aad_contrastive -- content-based attended-SOURCE identification via contrastive
EEG<->envelope matching, with auditory biological priors.

Goal (the brief): identify the attended AUDIO SOURCE by learning a shared space where
the EEG sits CLOSE to the attended speaker's envelope and FAR from the unattended
ones. This is *content* matching (neural envelope tracking) -- NOT spatial/direction
decoding. Candidates are permuted and the encoder is fully position-agnostic, so the
decision can only come from EEG<->envelope correspondence.

Biological priors baked into the architecture:
  * Spatial filter (1x1 conv over channels) -> auditory-cortex source components
    (CSP/beamformer-like), on raw re-referenced EEG (BatchNorm whitens; no per-channel
    z-score which would erase relative amplitudes).
  * TRF temporal filter (~0-400 ms kernel) -> the cortical envelope-tracking response
    function, in the delta-theta band (cache is 1-10 Hz).
  * Similarity = lag-tolerant temporal CORRELATION (CCA-style), amplitude-invariant --
    the quantity the brain actually tracks. A small lag search absorbs residual jitter.

Contrastive training (InfoNCE):
  * positive  = the attended speaker's envelope;
  * hard negs = the unattended speakers in the SAME 6-speaker scene (incl. the
    always-distractor speakers 5/6) -- same acoustics, only attention differs;
  * extra negs= attended envelopes of OTHER trials in the batch (in-batch negatives).
  * candidate order permuted per sample -> no positional/direction shortcut.
Prediction = pick the attendable speaker (1..4) whose envelope is most similar.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

from ..base import TorchModel
from ..factory import MODEL_REGISTRY


def _znt(x):                                  # zero-mean, unit-norm over the time axis
    x = x - x.mean(-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


class _SpatialTRF(nn.Module):
    """Raw EEG -> D spatial-temporal components (spatial filter + TRF temporal filter)."""

    def __init__(self, n_chans, K=16, D=32, trf=27, dropout=0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_chans)                 # whiten channels (keep relative amplitude)
        self.spatial = nn.Conv1d(n_chans, K, 1)            # auditory spatial filters
        self.bn1 = nn.BatchNorm1d(K)
        self.trf = nn.Conv1d(K, D, trf, padding=trf // 2, groups=math.gcd(K, D))  # TRF (0-~400 ms)
        self.bn2 = nn.BatchNorm1d(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, eeg):                                # (B,C,W)
        h = F.elu(self.bn1(self.spatial(self.bn0(eeg))))
        return self.drop(self.bn2(self.trf(h)))            # (B,D,W)


class _EnvEnc(nn.Module):
    """Speaker envelope -> D components matched to the EEG components."""

    def __init__(self, n_bands, D=32, ksize=17, dropout=0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_bands)
        self.c1 = nn.Conv1d(n_bands, D, ksize, padding=ksize // 2)
        self.bn1 = nn.BatchNorm1d(D)
        self.c2 = nn.Conv1d(D, D, ksize, padding=ksize // 2)
        self.bn2 = nn.BatchNorm1d(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # (N,bands,W)
        h = F.elu(self.bn1(self.c1(self.bn0(x))))
        return self.drop(self.bn2(self.c2(h)))             # (N,D,W)


class _AADContrastive(nn.Module):
    def __init__(self, fd, K=16, D=32, trf=27, env_k=17, max_lag=24, lag_temp=0.5, dropout=0.2):
        super().__init__()
        self.eeg = _SpatialTRF(fd["n_chans"], K, D, trf, dropout)
        self.env = _EnvEnc(fd["n_bands"], D, env_k, dropout)
        self.D = D
        self.lags = list(range(-max_lag, max_lag + 1))
        self.lag_temp = lag_temp

    def embed_eeg(self, eeg):
        return _znt(self.eeg(eeg))                         # (B,D,W)

    def embed_env(self, env):
        return _znt(self.env(env))                         # (N,D,W)

    def lag_sim(self, E, V):
        """E (B,D,W), V (B,S,D,W) -> (B,S): soft-max-over-lags corr, mean over D."""
        Eu = E.unsqueeze(1); T = V.shape[-1]; per = []
        for L in self.lags:
            if L > 0:   c = (Eu[..., L:] * V[..., :T - L]).sum(-1)
            elif L < 0: c = (Eu[..., :L] * V[..., -L:]).sum(-1)
            else:       c = (Eu * V).sum(-1)
            per.append(c)
        st = torch.stack(per, -1)                          # (B,S,D,nlag)
        return (self.lag_temp * torch.logsumexp(st / self.lag_temp, -1)).mean(-1)  # (B,S)

    def cross_sim(self, E, A):
        """Zero-lag sim(E_i, A_j) for all i,j -> (B,B) in-batch negatives (efficient)."""
        B = E.shape[0]
        return (E.reshape(B, -1) @ A.reshape(B, -1).t()) / self.D


@MODEL_REGISTRY.register("aad_contrastive")
class AADContrastiveModel(TorchModel):
    name = "aad_contrastive"

    def __init__(self, cfg, feature_dims):
        super().__init__(cfg, feature_dims)
        self.K = int(cfg.get("K", 16))
        self.D = int(cfg.get("D", 32))
        self.trf = int(cfg.get("trf", 27))           # ~420 ms @ 64 Hz
        self.env_k = int(cfg.get("env_k", 17))
        self.max_lag = int(cfg.get("max_lag", 24))   # ±375 ms lag search
        self.lag_temp = float(cfg.get("lag_temp", 0.5))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.temp = float(cfg.get("temp", 0.1))      # InfoNCE temperature
        self.in_batch_negs = bool(cfg.get("in_batch_negs", True))
        self.permute = bool(cfg.get("permute_candidates", True))

    def build_module(self):
        return _AADContrastive(self.fd, self.K, self.D, self.trf, self.env_k,
                               self.max_lag, self.lag_temp, self.dropout)

    def _embed_cands(self, cand):
        B, S = cand.shape[0], cand.shape[1]
        V = self.module.embed_env(cand.reshape(B * S, cand.shape[2], cand.shape[3]))
        return V.reshape(B, S, self.D, -1)                 # (B,S,D,W)

    def compute_loss(self, batch):
        eeg, cand, att = batch["eeg"], batch["cand_env"], batch["attended"]
        B, S = cand.shape[0], cand.shape[1]
        dev = eeg.device
        # Permute the 6 candidates per sample so candidate INDEX carries no info.
        if self.permute:
            perm = torch.argsort(torch.rand(B, S, device=dev), dim=1)          # (B,S)
            cand = torch.gather(cand, 1, perm.view(B, S, 1, 1).expand(-1, -1, cand.shape[2], cand.shape[3]))
            att = (perm == att.view(B, 1)).long().argmax(1)                    # attended's new index
        E = self.module.embed_eeg(eeg)                     # (B,D,W)
        V = self._embed_cands(cand)                        # (B,S,D,W)
        s_scene = self.module.lag_sim(E, V)                # (B,S) same-scene similarities
        logits = s_scene
        if self.in_batch_negs and B > 1:
            att_emb = V[torch.arange(B, device=dev), att]  # (B,D,W) attended embeddings
            cross = self.module.cross_sim(E, att_emb)      # (B,B)
            cross = cross.masked_fill(torch.eye(B, device=dev, dtype=torch.bool), float("-inf"))
            logits = torch.cat([s_scene, cross], dim=1)    # (B, S+B); positive still at `att`
        loss = F.cross_entropy(logits / self.temp, att)
        with torch.no_grad():
            m = batch["cand_mask"].bool()
            if self.permute:
                m = torch.gather(m, 1, perm)
            acc = (s_scene.masked_fill(~m, float("-inf")).argmax(1) == att).float().mean()
        return loss, {"infonce": float(loss), "acc": float(acc)}

    def predict_logits(self, batch, present_override=None):
        # No permutation at eval: score the (fixed-order) 6 speakers; base masks 5/6.
        E = self.module.embed_eeg(batch["eeg"])
        V = self._embed_cands(batch["cand_env"])
        return self.module.lag_sim(E, V)                   # (B,6)

    def evaluate(self, view, ctx, prefix="test/", present_override=None):
        """SELF-VALIDATING eval: trial-level accuracy AND the EEG-SHUFFLE control.

        Reports the normal trial-level decision (canonical) plus ``shuf_acc`` -- the
        same decision but with the EEG scrambled across windows (random EEG, envelopes
        intact). If ``shuf_acc`` ~ ``acc`` the model is using the AUDIO, not the EEG
        (a confound). Genuine EEG decoding requires acc >> shuf_acc. Task-type aware
        (works for the shifted match-mismatch task, not just the speaker task).
        """
        import numpy as np
        import torch
        from ..base import compute_aad_metrics

        fd = getattr(self, "fd", {}) or {}
        task = fd.get("task_type", "speaker"); n_cand = int(fd.get("n_candidates", 4))
        dev = ctx.device
        self.module.eval()

        # materialise all test windows once
        N = len(view)
        eeg, cand, mask, true = [], [], [], []
        for i in range(N):
            s = view.materialize(i)
            eeg.append(s["eeg"]); cand.append(s["cand_env"]); mask.append(s["cand_mask"])
            true.append(int(s["attended"]))
        if N == 0:
            return {f"{prefix}n": 0}
        eeg = torch.tensor(np.stack(eeg)); cand = torch.tensor(np.stack(cand))
        mask = np.stack(mask); true = np.array(true)
        rec_ptr = np.array([view.indices[i].rec_ptr for i in range(N)])
        subj = np.array([view.records[view.indices[i].rec_ptr].subject for i in range(N)])

        def posteriors(eeg_t):
            P = []
            with torch.no_grad():
                for j in range(0, N, self.batch_size):
                    E = self.module.embed_eeg(eeg_t[j:j + self.batch_size].to(dev))
                    B = E.shape[0]; S = cand.shape[1]
                    V = self.module.embed_env(
                        cand[j:j + self.batch_size].reshape(B * S, cand.shape[2], cand.shape[3]).to(dev)
                    ).reshape(B, S, self.D, -1)
                    sc = self.module.lag_sim(E, V).cpu().numpy()
                    sc = np.where(mask[j:j + self.batch_size], sc, -np.inf)
                    e = np.exp(sc - np.nanmax(np.where(np.isfinite(sc), sc, -np.inf), 1, keepdims=True))
                    e[~np.isfinite(sc)] = 0.0; P.append(e / e.sum(1, keepdims=True))
            return np.concatenate(P)

        def trial_preds(P):
            tp, tt, ts = [], [], []
            for r in np.unique(rec_ptr):
                idx = np.where(rec_ptr == r)[0]
                tp.append(int(P[idx].sum(0).argmax())); tt.append(int(true[idx[0]])); ts.append(int(subj[idx[0]]))
            return np.array(tp), np.array(tt), np.array(ts)

        P = posteriors(eeg)
        rng = np.random.default_rng(0)
        Pshuf = posteriors(eeg[torch.as_tensor(rng.permutation(N))])    # EEG scrambled

        out = {}
        out.update(compute_aad_metrics(P.argmax(1), true, prefix=f"{prefix}window/",
                                       task_type=task, n_cand=n_cand))
        tp, tt, ts = trial_preds(P)
        out.update(compute_aad_metrics(tp, tt, prefix=prefix, task_type=task, n_cand=n_cand))
        sp, _, _ = trial_preds(Pshuf)
        out[f"{prefix}shuf_acc"] = float((sp == tt).mean())            # EEG-shuffle control
        for s in np.unique(ts):
            m = ts == s
            out[f"{prefix}subj{int(s)}/acc"] = float((tp[m] == tt[m]).mean())
        out[f"{prefix}n"] = int(len(true))
        return out
