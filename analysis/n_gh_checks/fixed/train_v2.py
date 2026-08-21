"""Ablation ladder for the fixed AAD model, with the brain-shuffle null wired
into BOTH the objective and the model-selection criterion.

Ladder (each row changes exactly one thing from the row above):

  A0   upstream encoder + upstream head + raw candidates + CE        (repro)
  A0b  same, with upstream's own Adam/lr=1e-4                        (repro)
  A1   + v2 encoder      (RF 1 s, GroupNorm, no final ReLU, centred)
  A2   + correlation head (time-centred; no constant-brain solution)
  A3   + anti-shortcut loss (CLIP over the brain axis, null hinges,
         anti-collapse, audio-only adversary)
  A4   + quantile-matched candidates   (marginal shortcut removed)
  A5   + same-source time-shifted negatives (shortcut removed by design)

Multimodal:
  M-gaze/M-imu/M-video/M-eeg-spatial  spatial head only, NO audio input at all
  M-full                              EEG coupling + behaviour spatial fusion

Every config is scored with the same battery: test accuracy, brain-shuffle
null, margin + permutation p, zeros-brain ablation, decision-flip rate and
embedding-collapse statistics.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

UP = "/fs/scratch/PAS2301/alialavi/MAESTRO_upstream"
sys.path.insert(0, os.path.join(UP, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import (build_dataset, load_official_splits,            # noqa: E402
                        get_official_split_windows, carve_inner_val,
                        carve_inner_val_content, compute_global_content_holdout)
from model_v2 import AADModelV2                                          # noqa: E402
from data_v2 import (AADDatasetV2, collate_v2, SubjectBatchSampler,      # noqa: E402
                     make_audio_bank, subject_per_window, content_per_window,
                     position_in_trial)
from losses_v2 import total_loss                                         # noqa: E402
from candidates_v2 import audio_only_probe                               # noqa: E402

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANTI = dict(anti_shortcut=True, w_clip=1.0, w_null=0.5, w_vic=0.1,
            w_adv=0.3, margin=0.5, smoothing=0.1)
PLAIN = dict(anti_shortcut=False, smoothing=0.1)

LADDER = {
    "A0":  dict(mods=("eeg",), encoder="legacy", head="legacy", cand="raw",
                anti=False, lr=1e-3, opt="adamw"),
    "A0b": dict(mods=("eeg",), encoder="legacy", head="legacy", cand="raw",
                anti=False, lr=1e-4, opt="adam"),
    "A1":  dict(mods=("eeg",), encoder="v2", head="legacy", cand="raw",
                anti=False),
    "A2":  dict(mods=("eeg",), encoder="v2", head="corr", cand="raw",
                anti=False),
    "A3":  dict(mods=("eeg",), encoder="v2", head="corr", cand="raw",
                anti=True),
    "A4":  dict(mods=("eeg",), encoder="v2", head="corr", cand="qmatch",
                anti=True),
    # Same-source negatives need hop-disjoint windows: at window/hop = 2 a trial
    # yields 5 windows, which supports at most 2 non-overlapping imposters.
    # A5 is therefore 3-way (chance 1/3) and A6 the literature-standard binary
    # match-mismatch (chance 1/2) — an overlapping "negative" would be partly
    # correct, so we shrink K rather than allow one.
    "A5":  dict(mods=("eeg",), encoder="v2", head="corr", cand="shifted_qm",
                anti=True, K=3),
    "A6":  dict(mods=("eeg",), encoder="v2", head="corr", cand="shifted_qm",
                anti=True, K=2),
    "A5r": dict(mods=("eeg",), encoder="v2", head="corr", cand="shifted",
                anti=True, K=3),
    # Lag-band controls. RF = 63 samples ~ 0.98 s, so with an 8-sample (125 ms)
    # shift a "future"-directed encoder sees the EEG only in +125..+1109 ms
    # relative to the audio sample it scores (the neural direction), and a
    # "past"-directed encoder with the mirrored shift sees only -1109..-125 ms,
    # which no stimulus-evoked response can occupy.  A margin that is as large
    # in the acausal direction is a shared artifact, not brain tracking.
    "A4f": dict(mods=("eeg",), encoder="v2", head="corr", cand="qmatch",
                anti=True, brain_dir="future", lag_samples=8),
    "A4b": dict(mods=("eeg",), encoder="v2", head="corr", cand="qmatch",
                anti=True, brain_dir="past", lag_samples=-8),
    "A6f": dict(mods=("eeg",), encoder="v2", head="corr", cand="shifted_qm",
                anti=True, K=2, brain_dir="future", lag_samples=8),
    "A6b": dict(mods=("eeg",), encoder="v2", head="corr", cand="shifted_qm",
                anti=True, K=2, brain_dir="past", lag_samples=-8),
    # All M-* configs are BUILT from the merged four-modality dataset so they are
    # evaluated on identical windows; `mods` selects which encoders the model
    # actually instantiates.  The spatial head takes no audio input at all.
    "M-gaze":  dict(mods=("gaze",), spatial=("gaze",)),
    "M-imu":   dict(mods=("imu",), spatial=("imu",)),
    "M-video": dict(mods=("video",), spatial=("video",)),
    "M-eeg-spatial": dict(mods=("eeg",), spatial=("eeg",)),
    "M-behav": dict(mods=("gaze", "imu", "video"), spatial=("gaze", "imu", "video"),
                    w_aux=0.3, modality_dropout=0.3),
    "M-full": dict(mods=("eeg", "gaze", "imu", "video"), head="corr",
                   spatial=("eeg", "gaze", "imu", "video"), cand="qmatch",
                   anti=True, w_aux=0.3, modality_dropout=0.3),
}
BUILD_ALL = ("eeg", "gaze", "imu", "video")
for _n, _c in LADDER.items():
    if _n.startswith("M-"):
        _c.setdefault("encoder", "v2"); _c.setdefault("head", "none")
        _c.setdefault("cand", "qmatch"); _c.setdefault("anti", False)
        _c.setdefault("w_aux", 0.0); _c["build_mods"] = BUILD_ALL


# ── evaluation battery ─────────────────────────────────────────────────────────

class Evaluator:
    """Caches per-window modality embeddings and audio embeddings once, then
    re-scores under arbitrary brain permutations without re-encoding."""

    def __init__(self, model, loader, device, chunk=512, strata=None):
        self.model, self.device, self.chunk = model, device, chunk
        if strata is None:
            self.strata = {}
        elif isinstance(strata, dict):
            self.strata = {k: np.asarray(v) for k, v in strata.items()}
        else:
            self.strata = {"pos": np.asarray(strata)}
        model.eval()
        embs = {m: [] for m in model.modalities}
        auds, labels, perms, spks, zero_e = [], [], [], [], {}
        with torch.no_grad():
            for mods, audio, lab, spk_of_slot, att, subj in loader:
                mods = {m: v.to(device) for m, v in mods.items()}
                e = model.encode(mods)
                for m, v in e.items():
                    embs[m].append(v.cpu())
                if model.couple_mod is not None:
                    a = torch.stack([model.audio_encoder(x.to(device))
                                     for x in audio], 1)          # (B,K,T,D)
                    auds.append(a.cpu())
                labels.append(lab); perms.append(spk_of_slot); spks.append(att)
                if not zero_e:
                    z = model.encode({m: torch.zeros_like(v) for m, v in mods.items()})
                    zero_e = {m: v[:1].cpu() for m, v in z.items()}
        self.embs = {m: torch.cat(v) for m, v in embs.items() if v}
        self.aud = torch.cat(auds) if auds else None
        self.labels = torch.cat(labels)
        self.perms = torch.cat(perms)
        self.spks = torch.cat(spks)
        self.zero_e = zero_e
        self.N = len(self.labels)

    @torch.no_grad()
    def logits(self, perm=None, zero=False):
        m0 = self.model
        idx = torch.arange(self.N) if perm is None else torch.as_tensor(perm)
        out = []
        for s in range(0, self.N, self.chunk):
            sl = slice(s, min(s + self.chunk, self.N))
            n = sl.stop - sl.start
            if zero:
                e = {m: self.zero_e[m].expand(n, -1, -1).to(self.device)
                     for m in self.embs}
            else:
                e = {m: self.embs[m][idx[sl]].to(self.device) for m in self.embs}
            lg = None
            if m0.couple_mod is not None and self.aud is not None:
                a = [self.aud[sl, k].to(self.device) for k in range(self.aud.shape[1])]
                lg = m0.head(e[m0.couple_mod], a)
            if m0.spatial_mods:
                sp = {m: m0.spatial_heads[m](e[m]) for m in m0.spatial_mods if m in e}
                if len(m0.spatial_mods) > 1:
                    sp["fused"] = m0.spatial_fuse(
                        torch.cat([e[m] for m in m0.spatial_mods], -1))
                key = "fused" if "fused" in sp else m0.spatial_mods[0]
                slot = torch.gather(sp[key], 1, self.perms[sl].to(self.device))
                lg = slot if lg is None else lg + slot
            out.append(lg.cpu())
        return torch.cat(out)

    def acc(self, lg):
        return float((lg.argmax(1) == self.labels).float().mean())

    def _shuffle(self, rng, key=None):
        """Global permutation, or one restricted to windows sharing a stratum.

        `pos`   — same position-in-trial: rules out a slow drift shared by brain
                  and envelope being what the margin reads.
        `trial` — same trial: the strictest control. It asks whether the model
                  matches this *window's* stimulus segment, or merely recognises
                  which trial the recording came from (all windows of a trial
                  share a listener, an attended talker and a stimulus set, so a
                  trial-level match would survive a global shuffle and inflate
                  the margin without being envelope tracking).
        """
        if key is None or key not in self.strata:
            return rng.permutation(self.N)
        st = self.strata[key]
        p = np.arange(self.N)
        for g in np.unique(st):
            m = np.where(st == g)[0]
            p[m] = m[rng.permutation(len(m))]
        return p

    def battery(self, n_shuffle=20, seed=1000):
        real_lg = self.logits()
        real = self.acc(real_lg)
        nulls, flips = [], []
        snulls = {k: [] for k in self.strata}
        for k in range(n_shuffle):
            rng = np.random.default_rng(seed + k)
            lg = self.logits(perm=self._shuffle(rng))
            nulls.append(self.acc(lg))
            flips.append(float((lg.argmax(1) != real_lg.argmax(1)).float().mean()))
            for nm in self.strata:
                snulls[nm].append(self.acc(self.logits(
                    perm=self._shuffle(np.random.default_rng(5000 + k), nm))))
        nulls = np.array(nulls)
        zero_acc = self.acc(self.logits(zero=True))
        # embedding-collapse stats on the coupling modality (or the first one)
        m = self.model.couple_mod or self.model.modalities[0]
        E = self.embs[m]
        sel = torch.randperm(len(E))[:512]

        def _offdiag_cos(V):
            V = F.normalize(V, dim=-1)
            return float(((V @ V.t()).sum() - len(V)) / (len(V) * (len(V) - 1)))

        # pooled cosine — comparable with the 0.9995 we measured upstream
        cos = _offdiag_cos(E[sel].mean(1))
        # time-centred cosine — the quantity the correlation head actually
        # consumes; 1.0 means every window has the same temporal profile
        Ec = E[sel] - E[sel].mean(1, keepdim=True)
        Ec = Ec / (Ec.norm(dim=1, keepdim=True) + 1e-6)
        cos_c = _offdiag_cos(Ec.reshape(len(sel), -1))
        K = self.aud.shape[1] if self.aud is not None else self.perms.shape[1]
        return dict(
            acc=real, null_mean=float(nulls.mean()), null_std=float(nulls.std()),
            margin=float(real - nulls.mean()),
            p_perm=float((np.sum(nulls >= real) + 1) / (n_shuffle + 1)),
            zeros_acc=zero_acc, flip_rate=float(np.mean(flips)),
            **{f"null_{nm}": float(np.mean(v)) for nm, v in snulls.items()},
            **{f"margin_{nm}": float(real - np.mean(v)) for nm, v in snulls.items()},
            emb_cos=cos, emb_cos_centered=cos_c,
            emb_temporal_std=float(E.std(1).mean()),
            chance=1.0 / K, n=self.N)


# ── training ───────────────────────────────────────────────────────────────────

def build_model(name, cfg):
    return AADModelV2(
        modalities=cfg["mods"], encoder=cfg.get("encoder", "v2"),
        head=cfg.get("head", "corr"), spatial=cfg.get("spatial", ()),
        adversary=cfg.get("anti", False) and cfg.get("head", "corr") == "corr",
        modality_dropout=cfg.get("modality_dropout", 0.0),
        lag_samples=cfg.get("lag_samples", 0),
        brain_dir=cfg.get("brain_dir")).to(DEVICE)


def run_epoch(model, loader, opt, lcfg, train=True):
    model.train(train)
    tot, nb = 0.0, 0
    with torch.set_grad_enabled(train):
        for mods, audio, lab, spk_of_slot, att, subj in loader:
            mods = {m: v.to(DEVICE) for m, v in mods.items()}
            audio = [a.to(DEVICE) for a in audio]
            lab, att = lab.to(DEVICE), att.to(DEVICE)
            spk_of_slot, subj = spk_of_slot.to(DEVICE), subj.to(DEVICE)
            out = model(mods, audio if model.couple_mod else None, spk_of_slot)
            loss, _ = total_loss(out, lab, subj, lcfg, spk_label=att)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += float(loss.detach()); nb += 1
    return tot / max(nb, 1)


def train_one(name, cfg, data, bank, tr_idx, vl_idx, te_idx, args, fold):
    te_strata = {"pos": position_in_trial(data)[te_idx],
                 "trial": data["trial_ids"][te_idx]}
    torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
    model = build_model(name, cfg)
    lcfg = dict(ANTI if cfg.get("anti") else PLAIN)
    lcfg["w_aux"] = cfg.get("w_aux", 0.3 if cfg.get("spatial") else 0.0)
    if getattr(model, "head", None) is not None:
        lcfg["head_module"] = model.head

    K = cfg.get("K", 4)
    mk = lambda idx, tr: AADDatasetV2(data, idx, bank, cfg["mods"], train=tr, n_cand=K)
    tr_ds, vl_ds, te_ds = mk(tr_idx, True), mk(vl_idx, False), mk(te_idx, False)
    subj_tr = subject_per_window(data)[tr_idx]
    tr_loader = DataLoader(tr_ds, batch_sampler=SubjectBatchSampler(
        subj_tr, args.batch_size, seed=SEED + fold), collate_fn=collate_v2)
    vl_loader = DataLoader(vl_ds, args.batch_size, shuffle=False, collate_fn=collate_v2)
    te_loader = DataLoader(te_ds, args.batch_size, shuffle=False, collate_fn=collate_v2)

    opt = (torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
           if cfg.get("opt") == "adam" else
           torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-4))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=5, min_lr=1e-6)
    best = {"margin": (-9e9, None), "acc": (-9e9, None)}
    bad = 0
    for ep in range(1, args.epochs + 1):
        tl = run_epoch(model, tr_loader, opt, lcfg, True)
        ev = Evaluator(model, vl_loader, DEVICE)
        b = ev.battery(n_shuffle=args.val_shuffles, seed=7)
        sch.step(b["margin"])
        if b["margin"] > best["margin"][0]:
            best["margin"] = (b["margin"], {k: v.detach().cpu().clone()
                                            for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if b["acc"] > best["acc"][0]:
            best["acc"] = (b["acc"], {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()})
        if ep % 5 == 0 or ep == 1:
            print(f"    ep{ep:03d} loss={tl:.4f} val_acc={b['acc']:.4f} "
                  f"val_null={b['null_mean']:.4f} margin={b['margin']:+.4f} "
                  f"cosC={b['emb_cos_centered']:.4f}", flush=True)
        if bad >= args.patience:
            print(f"    early stop @ep{ep}", flush=True); break

    res = {}
    for sel in ("margin", "acc"):
        if best[sel][1] is None:
            continue
        model.load_state_dict(best[sel][1])
        res[f"sel_{sel}"] = Evaluator(model, te_loader, DEVICE,
                                      strata=te_strata).battery(
            n_shuffle=args.test_shuffles)
    res["n_params"] = sum(p.numel() for p in model.parameters())
    res["rf_samples"] = getattr(model.encoders[cfg["mods"][0]], "receptive_field", None)
    return res


# ── driver ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_path", default="/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset")
    ap.add_argument("--cache_root", default="/fs/scratch/PAS2301/alialavi/cache")
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--hop_sec", type=float, default=5.0)
    ap.add_argument("--split_setting", default="within", choices=["within", "loso"])
    ap.add_argument("--configs", default="A0,A0b,A1,A2,A3,A4,A5")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_shuffles", type=int, default=3)
    ap.add_argument("--test_shuffles", type=int, default=20)
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_names = args.configs.split(",")
    bmods = lambda c: tuple(LADDER[c].get("build_mods", LADDER[c]["mods"]))
    mod_sets = sorted({bmods(c) for c in cfg_names}, key=len)
    print(f"Device: {DEVICE} | configs: {cfg_names}")

    subjects = "all" if args.subjects == "all" else \
        [int(s) for s in args.subjects.split(",")]

    # one dataset build per distinct modality set, memoised as a single npz so
    # repeat jobs start in seconds instead of re-extracting 6400 FLAC envelopes
    datasets = {}
    for ms in mod_sets:
        mode = "_".join(ms)
        cache = os.path.join(args.cache_root, f"n_gh_newrepo__{mode}")
        dc = os.path.join(args.cache_root, f"n_gh_fixed_data__{mode}"
                          f"_w{args.window_sec:g}_h{args.hop_sec:g}"
                          f"_{'all' if subjects == 'all' else len(subjects)}.npz")
        if os.path.exists(dc):
            print(f"\n[build] mode={mode} <- cached {dc}", flush=True)
            z = np.load(dc, allow_pickle=False)
            d = {k: z[k] for k in z.files if not k.startswith("audio_")}
            d["audio"] = [z[f"audio_{i}"] for i in range(4)]
            for k in ("eeg", "video", "gaze", "imu"):
                d.setdefault(k, None)
            datasets[ms] = d
            continue
        print(f"\n[build] mode={mode} cache={cache}")
        d = build_dataset(local_path=args.local_path, mode=mode,
                          subjects=subjects, cache_dir=cache,
                          window_sec=args.window_sec, hop_sec=args.hop_sec)
        datasets[ms] = d
        save = {k: v for k, v in d.items()
                if k != "audio" and v is not None}
        save["trial_meta_tid"] = save["trial_meta_tid"].astype("<U24")
        save.update({f"audio_{i}": a for i, a in enumerate(d["audio"])})
        tmp = dc + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, **save); os.replace(tmp, dc)
        print(f"[build] cached -> {dc}", flush=True)

    # candidate banks (split-independent) + the audio-only acceptance probe
    banks, probes = {}, {}
    for ms in mod_sets:
        data = datasets[ms]
        for cm, K in sorted({(LADDER[c]["cand"], LADDER[c].get("K", 4))
                             for c in cfg_names if bmods(c) == ms}):
            key = (ms, cm, K)
            print(f"[bank] {'_'.join(ms)} / {cm} / K={K}")
            banks[key] = make_audio_bank(data, cm, args.window_sec, args.hop_sec,
                                         n_cand=K)
            pk = f"{cm}_K{K}"
            if pk not in probes:
                probes[pk] = audio_only_probe(banks[key]["A"], banks[key]["pos"],
                                              content_per_window(data))
                print(f"  [acceptance] audio-only shape probe on '{cm}' K={K} "
                      f"candidates: {probes[pk]:.4f} (chance {1/K:.4f})", flush=True)

    results = {"probes": probes, "args": vars(args), "configs": {}}
    for name in cfg_names:
        cfg = LADDER[name]
        bm = bmods(name)
        data = datasets[bm]
        bank = banks[(bm, cfg["cand"], cfg.get("K", 4))]
        folds = load_official_splits(os.path.join(args.local_path, "splits"),
                                     args.split_setting)
        if args.folds != "all":
            keep = {int(f) for f in args.folds.split(",")}
            folds = [f for f in folds if f["fold"] in keep]
        if args.split_setting == "loso":
            tr_c, ho_c = compute_global_content_holdout(data, 0.2, SEED)

        per_fold = {}
        for fi in folds:
            t0 = time.time()
            tr_idx, te_idx = get_official_split_windows(data, fi)
            if args.split_setting == "loso":
                wc = content_per_window(data)
                tr_idx = tr_idx[np.isin(wc[tr_idx], list(tr_c))]
                te_idx = te_idx[np.isin(wc[te_idx], list(ho_c))]
                itr, ivl = carve_inner_val(data, tr_idx, 0.2, SEED + fi["fold"])
            else:
                itr, ivl = carve_inner_val_content(data, tr_idx, 0.2, SEED + fi["fold"])
            print(f"\n=== {name} | fold {fi['fold']} | train {len(itr)} "
                  f"val {len(ivl)} test {len(te_idx)} ===", flush=True)
            per_fold[fi["fold"]] = train_one(name, cfg, data, bank, itr, ivl,
                                             te_idx, args, fi["fold"])
            m = per_fold[fi["fold"]]["sel_margin"]
            print(f"  -> fold {fi['fold']}: acc={m['acc']:.4f} null={m['null_mean']:.4f} "
                  f"margin={m['margin']:+.4f} p={m['p_perm']:.3f} "
                  f"cosC={m['emb_cos_centered']:.4f}  [{time.time()-t0:.0f}s]", flush=True)

        agg = {}
        for sel in ("sel_margin", "sel_acc"):
            vals = [v[sel] for v in per_fold.values() if sel in v]
            if not vals:
                continue
            agg[sel] = {k: [float(np.mean([v[k] for v in vals])),
                            float(np.std([v[k] for v in vals]))]
                        for k in vals[0] if isinstance(vals[0][k], float)}
        results["configs"][name] = {"cfg": {k: str(v) for k, v in cfg.items()},
                                    "folds": per_fold, "mean": agg}
        a = agg["sel_margin"]
        print(f"\n##### {name}: acc={a['acc'][0]:.4f}+-{a['acc'][1]:.4f} "
              f"null={a['null_mean'][0]:.4f} margin={a['margin'][0]:+.4f} "
              f"cosC={a['emb_cos_centered'][0]:.4f}", flush=True)

        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "results", "fixed",
                                       f"ladder_{args.split_setting}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
