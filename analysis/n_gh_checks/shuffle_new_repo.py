"""EEG/gaze/IMU/video-shuffle null on the UPDATED (2026-08-12) ASPIRE-OSU/MAESTRO repo.

The repo fixed the split/eval leakage (held-out test, content-disjoint LOSO, RMS
loudness-equalization, per-window candidate permutation) but the MODEL is
architecturally identical (only a docstring changed): it still ships a learned
per-candidate audio encoder + learned similarity head = an audio->label path. It
also ships trained checkpoints for both official protocols. This script loads
THEIR checkpoint, rebuilds THEIR exact test set, reproduces THEIR reported
accuracy, then re-runs with the ACTIVE BRAIN MODALITY shuffled across the fold's
test windows (each window keeps its own audio candidates + label; only which
brain recording sits next to them is permuted). If accuracy is unchanged -> the
decision ignores the brain modality -> non-neural (audio-only floor).

Eval-only, no training. Mirrors train_aad.run_official_splits() exactly for BOTH
the loso protocol (official subject split + global content holdout) and the
within/intra protocol (official content split).
"""
import os, sys, json, argparse
import numpy as np
import torch

UPSTREAM = "/fs/scratch/PAS2301/alialavi/MAESTRO_upstream"
LOCAL    = "/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset"
sys.path.insert(0, os.path.join(UPSTREAM, "scripts"))

from dataloader import (build_dataset, AADDataset, collate_fn,          # noqa: E402
                        load_official_splits, get_official_split_windows,
                        compute_global_content_holdout, N_SPEAKERS)
from model_classification import AADModel                                # noqa: E402
from torch.utils.data import DataLoader                                  # noqa: E402
from scipy import stats                                                  # noqa: E402

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tag -> (window_sec, hop_sec); matches results_aad_{loso,within}_<tag>
WINDOWS = {
    "w5_h2.5":  (5.0,  2.5),
    "w10_h5":   (10.0, 5.0),
    "w15_h7.5": (15.0, 7.5),
    "w20_h10":  (20.0, 10.0),
    "w30_h15":  (30.0, 15.0),
}


def materialize(data, idx):
    ds = AADDataset(data, idx, train=False)
    ld = DataLoader(ds, batch_size=len(idx), shuffle=False, collate_fn=collate_fn)
    eeg, video, gaze, imu, audio, labels = next(iter(ld))
    return dict(eeg=eeg, video=video, gaze=gaze, imu=imu, audio=audio, labels=labels)


@torch.no_grad()
def batched_acc(model, T, mode, batch=32, brain_override=None):
    """Batched-mean accuracy (matches evaluate_test averaging). brain_override,
    if given, replaces the ACTIVE brain modality tensor (for the shuffle null).
    Audio candidates + labels are always the window's own, unchanged."""
    model.eval()
    N = T["labels"].shape[0]
    accs = []
    for s in range(0, N, batch):
        e = slice(s, min(s + batch, N))
        def pick(name):
            t = brain_override if (brain_override is not None and name == mode) else T[name]
            return t[e].to(DEVICE) if t is not None else None
        eeg = pick("eeg"); video = pick("video"); gaze = pick("gaze"); imu = pick("imu")
        audio = [a[e].to(DEVICE) for a in T["audio"]]
        labels = T["labels"][e].to(DEVICE)
        probs = model(eeg, video, gaze, imu, audio)
        accs.append((probs.argmax(1) == labels.argmax(1)).float().mean().item())
    return float(np.mean(accs))


def eval_protocol(data, mode, protocol, tag, n_shuffle):
    """protocol in {loso, within}. Returns (rows, summary)."""
    setting = "within" if protocol == "within" else "loso"
    folds = load_official_splits(os.path.join(LOCAL, "splits"), setting)
    ckpt_dir = os.path.join(UPSTREAM, "results", f"results_aad_{protocol}_{tag}")

    if protocol == "loso":
        train_c, heldout_c = compute_global_content_holdout(
            data, held_out_content_frac=0.2, seed=SEED)
        win_content = data["trial_meta_tid"][
            np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]
        is_heldout = np.isin(win_content, list(heldout_c))

    rows = []
    for fi in folds:
        fold = fi["fold"]
        _, te_idx = get_official_split_windows(data, fi)
        if protocol == "loso":
            te_idx = te_idx[is_heldout[te_idx]]     # exactly train_aad's restriction
        if len(te_idx) == 0:
            continue
        ckpt = os.path.join(ckpt_dir, f"fold_{fold}_{mode}_{protocol}.pt")
        if not os.path.exists(ckpt):
            print(f"    [{protocol} {tag} {mode}] fold {fold}: ckpt missing, skip")
            continue
        model = AADModel(mode=mode).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=True)

        T = materialize(data, te_idx)
        real = batched_acc(model, T, mode)
        brain = T[mode]; N = brain.shape[0]
        nulls = [batched_acc(model, T, mode,
                             brain_override=brain[np.random.default_rng(1000 + k).permutation(N)])
                 for k in range(n_shuffle)]
        null = float(np.mean(nulls))
        rows.append(dict(fold=fold, n_test=int(N), real=real, null=null, margin=real - null))

    reals = np.array([r["real"] for r in rows]); nulls = np.array([r["null"] for r in rows])
    margins = reals - nulls
    t, p = stats.ttest_rel(reals, nulls) if len(rows) > 1 else (float("nan"), float("nan"))
    # their own reported number, if present
    rep = None
    rj = os.path.join(ckpt_dir, f"results_{mode}_{protocol}.json")
    if os.path.exists(rj):
        try: rep = json.load(open(rj)).get("mean_accuracy")
        except Exception: pass
    summary = dict(mode=mode, protocol=protocol, tag=tag, n_folds=len(rows),
                   reported=rep, real=float(reals.mean()), real_std=float(reals.std()),
                   null=float(nulls.mean()), null_std=float(nulls.std()),
                   margin=float(margins.mean()), margin_std=float(margins.std()),
                   t=float(t), p=float(p), chance=1.0 / N_SPEAKERS, folds=rows)
    rep_s = f"{rep:.4f}" if rep is not None else "  ?  "
    print(f"  [{protocol:6s} {tag:8s} {mode:5s}] reported={rep_s} real={reals.mean():.4f} "
          f"null={nulls.mean():.4f} margin={margins.mean():+.4f} p={p:.3g} (n={len(rows)})")
    return summary


def run(mode, tags, protocols, n_shuffle, out):
    print(f"[cfg] mode={mode} tags={tags} protocols={protocols} device={DEVICE}")
    cache_dir = f"/fs/scratch/PAS2301/alialavi/cache/n_gh_newrepo__{mode}"
    all_summaries = []
    for tag in tags:
        w, h = WINDOWS[tag]
        print(f"\n=== build mode={mode} {tag} (window={w}s hop={h}s) ===")
        data = build_dataset(local_path=LOCAL, mode=mode, cache_dir=cache_dir,
                             window_sec=w, hop_sec=h)
        for proto in protocols:
            all_summaries.append(eval_protocol(data, mode, proto, tag, n_shuffle))
        del data
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(all_summaries, open(out, "w"), indent=2)
    print(f"\nsaved {len(all_summaries)} summaries -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eeg")
    ap.add_argument("--tags", default=",".join(WINDOWS.keys()))
    ap.add_argument("--protocols", default="loso,within")
    ap.add_argument("--n_shuffle", type=int, default=20)
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t]
    protocols = [p for p in a.protocols.split(",") if p]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "shuffle_new", f"{a.mode}_full.json")
    run(a.mode, tags, protocols, a.n_shuffle, out)
