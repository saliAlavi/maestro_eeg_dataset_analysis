"""Linear backward (stimulus-reconstruction) AAD decoder -- the classical baseline.

Trains a single linear decoder ``g`` that reconstructs the *attended* broadband
speech envelope from time-lagged EEG. At decision time we reconstruct the
envelope for a test window and correlate it with each candidate speaker's
envelope; the speaker with the highest correlation (restricted to the four
attendable speakers) wins. This is the de-facto reference method in the AAD
literature (O'Sullivan et al., 2015) and the honest floor every neural model
must beat.
"""
from __future__ import annotations

import logging

import numpy as np

from ..base import ClassicalModel
from ..factory import MODEL_REGISTRY

log = logging.getLogger("model.linear_backward")


def _lag_embed(eeg: np.ndarray, n_lags: int) -> np.ndarray:
    """(C, W) -> (W, C*n_lags) using forward lags eeg(t+tau), tau=0..n_lags-1."""
    C, W = eeg.shape
    out = np.zeros((n_lags, C, W), np.float32)
    for tau in range(n_lags):
        out[tau, :, : W - tau] = eeg[:, tau:]
    return out.transpose(2, 1, 0).reshape(W, C * n_lags)


def _broadband(cand_env: np.ndarray) -> np.ndarray:
    """(N,6,B,W) -> (N,6,W) mean over bands, z-scored per window."""
    bb = cand_env.mean(axis=2)
    mu = bb.mean(axis=-1, keepdims=True)
    sd = bb.std(axis=-1, keepdims=True) + 1e-6
    return (bb - mu) / sd


@MODEL_REGISTRY.register("linear_backward")
class LinearBackwardModel(ClassicalModel):
    name = "linear_backward"

    def __init__(self, cfg, feature_dims: dict):
        self.cfg = cfg
        self.fd = feature_dims
        self.n_lags = int(cfg.get("n_lags", 16))
        self.alpha = float(cfg.get("alpha", 1e3))
        self.max_samples = int(cfg.get("max_train_samples", 200_000))
        self.g = None
        self._rng = np.random.default_rng(int(cfg.get("seed", 0)))

    def fit_numpy(self, data: dict) -> None:
        eeg = data["eeg"]                          # (N,C,W)
        bb = _broadband(data["cand_env"])          # (N,6,W)
        att = data["attended"]                     # (N,)
        N = eeg.shape[0]
        Xs, ys = [], []
        for i in range(N):
            X = _lag_embed(eeg[i], self.n_lags)    # (W, C*nlags)
            y = bb[i, att[i]]                      # (W,) attended broadband
            Xs.append(X); ys.append(y)
        X = np.concatenate(Xs, 0)
        y = np.concatenate(ys, 0)
        if len(y) > self.max_samples:
            idx = self._rng.choice(len(y), self.max_samples, replace=False)
            X, y = X[idx], y[idx]
        # Ridge closed form: g = (X'X + aI)^-1 X'y  (z-score X columns first).
        self._mu = X.mean(0); self._sd = X.std(0) + 1e-6
        Xz = (X - self._mu) / self._sd
        A = Xz.T @ Xz + self.alpha * np.eye(Xz.shape[1], dtype=np.float64)
        b = Xz.T @ y
        self.g = np.linalg.solve(A, b).astype(np.float32)
        log.info("[linear_backward] fit g: %d feats, %d samples, alpha=%g",
                 self.g.shape[0], len(y), self.alpha)

    def predict_numpy(self, data: dict) -> np.ndarray:
        eeg = data["eeg"]
        bb = _broadband(data["cand_env"])          # (N,6,W)
        N = eeg.shape[0]
        preds = np.zeros(N, int)
        for i in range(N):
            X = _lag_embed(eeg[i], self.n_lags)
            Xz = (X - self._mu) / self._sd
            shat = Xz @ self.g                     # (W,)
            shat = (shat - shat.mean()) / (shat.std() + 1e-6)
            corrs = (bb[i] @ shat) / len(shat)     # (6,) since bb is z-scored
            corrs[4:] = -np.inf                    # speakers 5/6 never attended
            preds[i] = int(np.argmax(corrs))
        return preds
