"""Riemannian tangent-space AAD -- spatial-covariance decoder (classical, strong).

Per window we estimate the EEG spatial covariance, project it to the tangent
space of the SPD manifold, and classify the attended speaker (4-class) with
multinomial logistic regression. Covariance/Riemannian features capture
spatial-attention signatures (e.g. alpha-band lateralisation) and are
consistently among the most robust, calibration-cheap AAD methods at short
decision windows.
"""
from __future__ import annotations

import logging

import numpy as np

from ..base import ClassicalModel
from ..factory import MODEL_REGISTRY

log = logging.getLogger("model.riemann_tangent")


@MODEL_REGISTRY.register("riemann_tangent")
class RiemannTangentModel(ClassicalModel):
    name = "riemann_tangent"

    def __init__(self, cfg, feature_dims: dict):
        self.cfg = cfg
        self.fd = feature_dims
        self.estimator = str(cfg.get("estimator", "oas"))
        self.C = float(cfg.get("C", 1.0))
        self.pipe = None

    def _build(self):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("cov", Covariances(estimator=self.estimator)),
            ("ts", TangentSpace(metric="riemann")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=2000, multi_class="auto")),
        ])

    def fit_numpy(self, data: dict) -> None:
        eeg = np.asarray(data["eeg"], np.float64)     # (N,C,W)
        y = np.asarray(data["attended"], int)
        self.pipe = self._build()
        self.pipe.fit(eeg, y)
        log.info("[riemann_tangent] fit on %d windows, %d classes",
                 len(y), len(np.unique(y)))

    def predict_numpy(self, data: dict) -> np.ndarray:
        eeg = np.asarray(data["eeg"], np.float64)
        pred = self.pipe.predict(eeg).astype(int)
        return np.clip(pred, 0, 3)
