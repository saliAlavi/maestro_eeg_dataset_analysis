"""transfer_ssl runner: whole-trial attended-source decoding with cross-subject
transfer + masked-channel SSL pretraining. Selected via `runner.name=transfer_ssl`.

Unlike the per-window AAD runner, this needs ALL subjects' whole-trial data at once
(for pooled/SSL pretraining), so it reads `dm.by_subject` directly rather than the
per-window split views. Reports per-subject + cohort mean (multi-seed) to wandb and
saves the SSL encoder + combo classifiers to the scratch run dir.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..data.factory import build_datamodule
from ..models.transfer_ssl.nets import Clf, acc, ssl_pretrain, train_clf, zs

log = logging.getLogger("runner.transfer_ssl")


def _gaze_feats(gz):                                          # (N,T,3) -> (N,8) x/y stats
    g = gz[:, :, :2]
    return np.concatenate([g.mean(1), g.std(1), np.percentile(g, 10, 1), np.percentile(g, 90, 1)], 1)


class TransferSSLRunner:
    def __init__(self, cfg, paths, wandb, device):
        self.cfg, self.paths, self.wandb, self.device = cfg, paths, wandb, device
        self.run_name = wandb.name
        self.m = cfg.model

    def _device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self, dm):
        D = {}
        for s, recs in dm.by_subject.items():
            eeg = zs(np.stack([r.eeg for r in recs]).astype(np.float32), 2)     # (N,32,T)
            gz = np.stack([r.gaze for r in recs]).astype(np.float32)            # (N,T,3)
            y = np.array([r.attended - 1 for r in recs], int)
            D[s] = (eeg, zs(_gaze_feats(gz).astype(np.float32), 0), y)
        T = min(D[s][0].shape[-1] for s in D)
        return {s: (e[:, :, :T], g, y) for s, (e, g, y) in D.items()}

    def _run_seed(self, D, subs, gdim, dev, seed):
        import torch
        torch.manual_seed(seed); np.random.seed(seed)
        h = int(self.m.get("h", 64)); ep_ft = int(self.m.get("finetune_epochs", 25))
        ep_sc = int(self.m.get("scratch_epochs", 60)); ep_pre = int(self.m.get("pretrain_epochs", 25))
        allE = np.concatenate([D[s][0] for s in subs])
        ssl_state = ssl_pretrain(allE, dev, epochs=int(self.m.get("ssl_epochs", 60)), h=h,
                                 mask_frac=float(self.m.get("mask_frac", 0.3))).state_dict()
        res = {k: {} for k in ("scratch", "transfer", "ssl", "combo")}
        for s in subs:
            Xe, Xg, y = D[s]
            oE = np.concatenate([D[o][0] for o in subs if o != s])
            oG = np.concatenate([D[o][1] for o in subs if o != s])
            oy = np.concatenate([D[o][2] for o in subs if o != s])
            pre = train_clf(Clf(gdim, h=h), oE, oG, oy, ep_pre, dev)
            cpre = Clf(gdim, h=h); cpre.enc.load_state_dict(ssl_state)
            cpre = train_clf(cpre, oE, oG, oy, ep_pre, dev)
            a = {k: [] for k in res}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xe, y):
                a["scratch"].append(acc(train_clf(Clf(gdim, h=h), Xe[tr], Xg[tr], y[tr], ep_sc, dev), Xe[te], Xg[te], y[te], dev))
                mt = Clf(gdim, h=h); mt.load_state_dict(pre.state_dict())
                a["transfer"].append(acc(train_clf(mt, Xe[tr], Xg[tr], y[tr], ep_ft, dev, lr=5e-4), Xe[te], Xg[te], y[te], dev))
                ms = Clf(gdim, h=h); ms.enc.load_state_dict(ssl_state)
                a["ssl"].append(acc(train_clf(ms, Xe[tr], Xg[tr], y[tr], ep_sc, dev), Xe[te], Xg[te], y[te], dev))
                mc = Clf(gdim, h=h); mc.load_state_dict(cpre.state_dict())
                a["combo"].append(acc(train_clf(mc, Xe[tr], Xg[tr], y[tr], ep_ft, dev, lr=5e-4), Xe[te], Xg[te], y[te], dev))
            for k in res:
                res[k][s] = float(np.mean(a[k]))
            log.info("seed%d S%d combo=%.3f (sc=%.3f tr=%.3f ssl=%.3f)", seed, s,
                     res["combo"][s], res["scratch"][s], res["transfer"][s], res["ssl"][s])
        return res, ssl_state

    def run(self) -> pd.DataFrame:
        import torch
        dm = build_datamodule(self.cfg.data); dm.prepare()
        D = self._load(dm); subs = sorted(D); gdim = D[subs[0]][1].shape[1]
        dev = self._device()
        seeds = list(self.m.get("seeds", [0, 1, 2]))
        per = {k: [] for k in ("scratch", "transfer", "ssl", "combo")}
        rows = []; last_ssl = None
        for sd in seeds:
            res, last_ssl = self._run_seed(D, subs, gdim, dev, sd)
            for k in per:
                per[k].append(np.mean([res[k][s] for s in subs]))
            for s in subs:
                rows.append(dict(seed=sd, subject=s, **{k: res[k][s] for k in res}))
            self.wandb.log({f"seed{sd}/{k}": per[k][-1] for k in per})
            log.info("seed%d means: %s", sd, {k: round(per[k][-1], 3) for k in per})
        summary = {}
        for k in per:
            a = np.array(per[k]); summary[f"{k}/mean"] = float(a.mean()); summary[f"{k}/std"] = float(a.std())
            log.info("%-9s = %.3f ± %.3f", k, a.mean(), a.std())
        self.wandb.log(summary)
        # save artifacts to scratch run dir (run_name already carries the project prefix)
        out = Path(f"/fs/scratch/PAS2301/alialavi/projects/{self.run_name}")
        out.mkdir(parents=True, exist_ok=True)
        if last_ssl is not None:
            torch.save(last_ssl, out / "ssl_encoder.pt")
        df = pd.DataFrame(rows); df.to_csv(out / "per_subject.csv", index=False)
        log.info("saved artifacts -> %s", out)
        return df
