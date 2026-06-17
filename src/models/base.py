"""Model layer: the constant model interface + shared training machinery.

Every model -- classical or neural -- implements ``AbstractModel`` so the runner
treats them identically:

    model.fit(train_view, val_view, ctx)
    metrics = model.evaluate(test_view, ctx)     # dict of AAD accuracies

Two base classes remove boilerplate:
  * ``ClassicalModel`` -- you implement ``fit_numpy`` / ``predict_numpy``.
  * ``TorchModel``     -- you implement ``build_module`` / ``compute_loss`` /
    ``predict_logits``; the shared loop handles device, AdamW, tqdm, wandb,
    checkpointing and best-model selection.

Prediction convention: every model returns an attended-speaker index in 0..3.
Neural models emit 6 candidate logits and the base masks speakers 5/6 (never
attended) before the argmax.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..common.wandb_utils import WandbLogger
from ..data.base import AADView

log = logging.getLogger("model")

# 0-indexed attended-speaker label maps (speaker 1..4 -> index 0..3).
HEMISPHERE = {0: 0, 1: 0, 2: 1, 3: 1}          # 1,2 -> Left ; 3,4 -> Right
INNER_OUTER = {0: 1, 1: 0, 2: 0, 3: 1}         # near {2,3} -> inner(0) ; far {1,4} -> outer(1)


def compute_aad_metrics(pred: np.ndarray, true: np.ndarray, prefix: str = "",
                        task_type: str = "speaker", n_cand: int = 4) -> dict:
    pred = np.asarray(pred).astype(int)
    true = np.asarray(true).astype(int)
    n = len(true)
    if n == 0:
        return {f"{prefix}n": 0}
    acc = float((pred == true).mean())
    out = {f"{prefix}acc": acc, f"{prefix}n": int(n), f"{prefix}chance": 1.0 / n_cand}
    if task_type == "speaker":
        # extra collapses only meaningful when candidates ARE the 4 speakers
        hemi_p = np.array([HEMISPHERE[p] for p in pred]); hemi_t = np.array([HEMISPHERE[t] for t in true])
        io_p = np.array([INNER_OUTER[p] for p in pred]); io_t = np.array([INNER_OUTER[t] for t in true])
        out[f"{prefix}acc_4class"] = acc
        out[f"{prefix}acc_hemisphere"] = float((hemi_p == hemi_t).mean())
        out[f"{prefix}acc_inner_outer"] = float((io_p == io_t).mean())
    return out


@dataclass
class RunContext:
    device: str = "cpu"
    wandb: Optional[WandbLogger] = None
    model_dir: Optional[Path] = None
    split_name: str = ""
    global_step: list = field(default_factory=lambda: [0])  # mutable shared counter

    def log(self, metrics: dict) -> None:
        if self.wandb is not None:
            self.global_step[0] += 1
            self.wandb.log(metrics, step=self.global_step[0])


class AbstractModel(ABC):
    is_neural: bool = False
    name: str = "abstract"

    @abstractmethod
    def fit(self, train_view: AADView, val_view: Optional[AADView], ctx: RunContext) -> dict:
        ...

    @abstractmethod
    def predict(self, view: AADView, ctx: RunContext, present_override=None) -> np.ndarray:
        ...

    def evaluate(self, view: AADView, ctx: RunContext, prefix: str = "test/",
                 present_override=None) -> dict:
        pred = self.predict(view, ctx, present_override=present_override)
        true = view.as_numpy()["attended"]
        fd = getattr(self, "fd", {}) or {}
        return compute_aad_metrics(pred, true, prefix=prefix,
                                   task_type=fd.get("task_type", "speaker"),
                                   n_cand=fd.get("n_candidates", 4))

    def save(self, path: Path) -> None:  # optional override
        pass

    def load(self, path: Path) -> None:  # optional override
        pass


# ----------------------------------------------------------------------------
# Classical base
# ----------------------------------------------------------------------------
class ClassicalModel(AbstractModel):
    is_neural = False

    @abstractmethod
    def fit_numpy(self, data: dict) -> None: ...

    @abstractmethod
    def predict_numpy(self, data: dict) -> np.ndarray: ...

    def fit(self, train_view, val_view, ctx) -> dict:
        log.info("[%s] fitting on %d windows", self.name, len(train_view))
        self.fit_numpy(train_view.as_numpy())
        out = {}
        if val_view is not None and len(val_view):
            m = self.evaluate(val_view, ctx, prefix="val/")
            ctx.log({f"{self.name}/{ctx.split_name}/{k}": v for k, v in m.items()})
            out = m
        return out

    def predict(self, view, ctx, present_override=None) -> np.ndarray:
        return self.predict_numpy(view.as_numpy())


# ----------------------------------------------------------------------------
# Neural base (shared training loop)
# ----------------------------------------------------------------------------
class TorchModel(AbstractModel):
    is_neural = True

    def __init__(self, cfg, feature_dims: dict):
        self.cfg = cfg
        self.fd = feature_dims
        self.module = None
        self.train_cfg = cfg.get("train", {})
        self.batch_size = int(self.train_cfg.get("batch_size", 64))
        self.epochs = int(self.train_cfg.get("epochs", 30))
        self.lr = float(self.train_cfg.get("lr", 1e-3))
        self.weight_decay = float(self.train_cfg.get("weight_decay", 1e-4))
        self.num_workers = int(self.train_cfg.get("num_workers", 0))
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))
        self.log_every = int(self.train_cfg.get("log_every", 20))

    # ---- subclass API --------------------------------------------------------
    @abstractmethod
    def build_module(self): ...

    @abstractmethod
    def compute_loss(self, batch: dict): ...   # -> (loss tensor, logs dict)

    @abstractmethod
    def predict_logits(self, batch: dict, present_override=None): ...  # -> (N,6)

    # ---- helpers -------------------------------------------------------------
    def _to_device(self, batch: dict, device: str) -> dict:
        import torch
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # ---- AbstractModel -------------------------------------------------------
    def fit(self, train_view, val_view, ctx) -> dict:
        import torch
        from tqdm import tqdm

        device = ctx.device
        if self.module is None:
            self.module = self.build_module()
        self.module.to(device)
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.epochs))
        loader = train_view.as_torch_loader(self.batch_size, shuffle=True,
                                            num_workers=self.num_workers)
        n_params = sum(p.numel() for p in self.module.parameters())
        log.info("[%s] %s: %.2fM params | %d train windows | %d epochs on %s",
                 self.name, ctx.split_name, n_params / 1e6, len(train_view),
                 self.epochs, device)

        best = {"val/acc_4class": -1.0}
        best_state = None
        for ep in range(self.epochs):
            self.module.train()
            agg, nb = {}, 0
            pbar = tqdm(loader, desc=f"{self.name} {ctx.split_name} ep{ep}",
                        leave=False, unit="batch")
            for batch in pbar:
                batch = self._to_device(batch, device)
                opt.zero_grad()
                loss, logs = self.compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
                opt.step()
                nb += 1
                for k, v in logs.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                if nb % self.log_every == 0:
                    pbar.set_postfix({k: f"{v:.3f}" for k, v in logs.items()})
                    ctx.log({f"{self.name}/{ctx.split_name}/step_{k}": float(v)
                             for k, v in logs.items()})
            sched.step()
            ep_logs = {f"{self.name}/{ctx.split_name}/ep_{k}": v / max(nb, 1)
                       for k, v in agg.items()}
            ep_logs[f"{self.name}/{ctx.split_name}/lr"] = sched.get_last_lr()[0]
            if val_view is not None and len(val_view):
                vm = self.evaluate(val_view, ctx, prefix="val/")
                ep_logs.update({f"{self.name}/{ctx.split_name}/{k}": v for k, v in vm.items()})
                if vm.get("val/acc_4class", 0) > best["val/acc_4class"]:
                    best = vm
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.module.state_dict().items()}
                    if ctx.model_dir is not None:
                        self.save(ctx.model_dir / f"{ctx.split_name}_best.pt")
            ctx.log(ep_logs)
            log.info("[%s] %s ep%d: %s", self.name, ctx.split_name, ep,
                     {k.split("/")[-1]: round(v, 4) for k, v in ep_logs.items()
                      if "ep_" in k or "val/" in k})
        # restore best-on-val weights so test uses the selected model, not the last epoch.
        if best_state is not None:
            self.module.load_state_dict(best_state)
            log.info("[%s] %s: restored best val model (val/acc_4class=%.4f)",
                     self.name, ctx.split_name, best["val/acc_4class"])
        return best

    def predict(self, view, ctx, present_override=None) -> np.ndarray:
        import torch

        self.module.eval()
        device = ctx.device
        loader = view.as_torch_loader(self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch, device)
                logits = self.predict_logits(batch, present_override=present_override)
                mask = batch["cand_mask"].bool()        # per-task valid candidates
                logits = logits.masked_fill(~mask, float("-inf"))
                preds.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], int)

    def save(self, path: Path) -> None:
        import torch
        if self.module is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": self.module.state_dict(),
                        "feature_dims": self.fd}, path)

    def load(self, path: Path) -> None:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        if self.module is None:
            self.module = self.build_module()
        self.module.load_state_dict(ckpt["state_dict"])
