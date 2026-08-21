"""Diagnostics to rule out alternative explanations for real==null on the
updated MAESTRO checkpoints. Answers, empirically, per modality:

 (Q collapse)   Are the brain embeddings near-identical across windows, so any
                shuffle is a no-op?  -> pairwise-cosine diversity of brain_enc.
 (Q gentle)     Is the within-fold shuffle 'too gentle' (same participant, and
                embeddings cluster by subject)?  -> add CROSS-SUBJECT shuffle and
                three input ablations maximally unlike the real brain:
                CONSTANT (mean), ZEROS, GAUSSIAN NOISE. If accuracy holds under
                these, the brain input is irrelevant regardless of clustering.
 (Q decision)   Does the per-trial 4-way argmax actually change when the brain
                changes?  -> decision-flip rate real-vs-shuffle / real-vs-const.
 (Q preproc)    Did preprocessing discard the neural signal so the model *cannot*
                use it?  -> POSITIVE CONTROL: decode SUBJECT identity from the
                model's own brain_enc (trial-disjoint). High acc => the pipeline
                preserved real per-subject structure that the attention decision
                simply ignores.

All eval-only, on their committed checkpoints; brain manipulations are at the
INPUT (fed through their encoder), the honest test.
"""
import os, sys, json, argparse
import numpy as np
import torch

UP = "/fs/scratch/PAS2301/alialavi/MAESTRO_upstream"
LOCAL = "/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset"
sys.path.insert(0, os.path.join(UP, "scripts"))
from dataloader import (build_dataset, AADDataset, collate_fn,          # noqa: E402
                        load_official_splits, get_official_split_windows,
                        compute_global_content_holdout, N_SPEAKERS)
from model_classification import AADModel                                # noqa: E402
from torch.utils.data import DataLoader                                  # noqa: E402
SEED = 42
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOWS = {"w10_h5": (10.0, 5.0)}


def materialize(data, idx):
    ds = AADDataset(data, idx, train=False)
    e, v, g, i, a, y = next(iter(DataLoader(ds, batch_size=len(idx), shuffle=False,
                                            collate_fn=collate_fn)))
    return dict(eeg=e, video=v, gaze=g, imu=i, audio=a, labels=y)


@torch.no_grad()
def predict(model, T, mode, brain, batch=64):
    """per-window 4-way argmax + prob, with brain (input tensor) fed as `mode`."""
    model.eval(); N = T["labels"].shape[0]; out = []
    for s in range(0, N, batch):
        e = slice(s, min(s + batch, N))
        kw = dict(eeg=None, video=None, gaze=None, imu=None)
        kw[mode] = brain[e].to(DEV)
        audio = [a[e].to(DEV) for a in T["audio"]]
        p = model(kw["eeg"], kw["video"], kw["gaze"], kw["imu"], audio)
        out.append(p.cpu())
    return torch.cat(out)


def acc_of(probs, labels):
    return (probs.argmax(1) == labels.argmax(1)).float().mean().item()


@torch.no_grad()
def brain_emb(model, mode, x, batch=64):
    """pooled brain_enc = mean_t({mode}_proj({mode}_encoder(x)))  -> (N, D)."""
    enc = getattr(model, f"{mode}_encoder"); proj = getattr(model, f"{mode}_proj")
    outs = []
    for s in range(0, x.shape[0], batch):
        xb = x[s:s + batch].to(DEV)
        outs.append(proj(enc(xb)).mean(1).cpu())
    return torch.cat(outs)


def make_brain(T, mode, kind, seed, subj=None):
    b = T[mode]; N = b.shape[0]; rng = np.random.default_rng(seed)
    if kind == "real":     return b
    if kind == "shuffle":  return b[rng.permutation(N)]
    if kind == "constant": return b.mean(0, keepdim=True).expand(N, -1, -1).contiguous()
    if kind == "zeros":    return torch.zeros_like(b)
    if kind == "noise":
        mu = b.mean((0, 1), keepdim=True); sd = b.std((0, 1), keepdim=True)
        return mu + sd * torch.randn(b.shape, generator=torch.Generator().manual_seed(seed))
    if kind == "shuffle_xsubj":
        perm = np.arange(N)
        for s in np.unique(subj):
            same = np.where(subj == s)[0]; diff = np.where(subj != s)[0]
            perm[same] = rng.choice(diff, size=len(same), replace=True)
        return b[perm]
    raise ValueError(kind)


def battery(model, T, mode, kinds, subj=None, n_seed=8):
    y = T["labels"]; res = {}
    real_pred = predict(model, T, mode, make_brain(T, mode, "real", 0)).argmax(1)
    for k in kinds:
        seeds = range(n_seed) if k in ("shuffle", "noise", "shuffle_xsubj") else [0]
        accs, flips = [], []
        for sd in seeds:
            p = predict(model, T, mode, make_brain(T, mode, k, 1000 + sd, subj))
            accs.append(acc_of(p, y)); flips.append((p.argmax(1) != real_pred).float().mean().item())
        res[k] = dict(acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
                      flip_vs_real=float(np.mean(flips)))
    return res


def emb_diversity(model, T, mode):
    E = brain_emb(model, mode, T[mode]).numpy()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    # mean pairwise cosine on a random subset of pairs
    N = len(En); rng = np.random.default_rng(0)
    idx = rng.integers(0, N, size=(min(5000, N * (N - 1) // 2 if N > 1 else 1), 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    pc = float((En[idx[:, 0]] * En[idx[:, 1]]).sum(1).mean()) if len(idx) else float("nan")
    return dict(n=int(N), dim=int(E.shape[1]), mean_pairwise_cosine=pc,
                emb_norm_mean=float(np.linalg.norm(E, axis=1).mean()),
                emb_norm_cv=float(np.linalg.norm(E, axis=1).std() /
                                  (np.linalg.norm(E, axis=1).mean() + 1e-8)))


def subject_decode(model, T, mode, data, te_idx):
    """POSITIVE CONTROL: subject identity from pooled brain_enc, trial-disjoint."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    E = brain_emb(model, mode, T[mode]).numpy()
    subj = data["trial_meta_subject"][np.searchsorted(data["trial_meta_ids"],
                                                       data["trial_ids"][te_idx])]
    trial = data["trial_ids"][te_idx]
    uq_tr = np.unique(trial); rng = np.random.default_rng(0); rng.shuffle(uq_tr)
    cut = int(0.7 * len(uq_tr)); tr_tr = set(uq_tr[:cut].tolist())
    m = np.array([t in tr_tr for t in trial])
    if len(np.unique(subj[m])) < 2 or (~m).sum() < 5:
        return None
    sc = StandardScaler().fit(E[m])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(E[m]), subj[m])
    acc = float((clf.predict(sc.transform(E[~m])) == subj[~m]).mean())
    return dict(subject_decode_acc=acc, chance=1.0 / len(np.unique(subj)),
                n_subjects=int(len(np.unique(subj))), n_test=int((~m).sum()))


def run(mode, tag, do_within):
    w, h = WINDOWS[tag]
    cache = f"/fs/scratch/PAS2301/alialavi/cache/n_gh_newrepo__{mode}"
    data = build_dataset(local_path=LOCAL, mode=mode, cache_dir=cache,
                         window_sec=w, hop_sec=h)
    win_subj = data["trial_meta_subject"][
        np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]

    report = {"mode": mode, "tag": tag}

    # ---- LOSO: accuracy battery + diversity, aggregated over 16 folds ----
    folds = load_official_splits(os.path.join(LOCAL, "splits"), "loso")
    tc, hc = compute_global_content_holdout(data, 0.2, SEED)
    is_hc = np.isin(data["trial_meta_tid"][
        np.searchsorted(data["trial_meta_ids"], data["trial_ids"])], list(hc))
    kinds = ["real", "shuffle", "constant", "zeros", "noise"]
    agg = {k: [] for k in kinds}; flips = {k: [] for k in kinds}; divs = []
    for fi in folds:
        _, te = get_official_split_windows(data, fi); te = te[is_hc[te]]
        if len(te) == 0: continue
        m = AADModel(mode=mode).to(DEV)
        m.load_state_dict(torch.load(os.path.join(
            UP, "results", f"results_aad_loso_{tag}", f"fold_{fi['fold']}_{mode}_loso.pt"),
            map_location=DEV), strict=True)
        T = materialize(data, te)
        b = battery(m, T, mode, kinds)
        for k in kinds: agg[k].append(b[k]["acc"]); flips[k].append(b[k]["flip_vs_real"])
        divs.append(emb_diversity(m, T, mode))
    report["loso"] = {k: dict(acc=float(np.mean(agg[k])), acc_std=float(np.std(agg[k])),
                              flip_vs_real=float(np.mean(flips[k]))) for k in kinds}
    report["loso"]["embedding_diversity_mean"] = {
        "mean_pairwise_cosine": float(np.mean([d["mean_pairwise_cosine"] for d in divs])),
        "emb_norm_cv": float(np.mean([d["emb_norm_cv"] for d in divs]))}

    # ---- WITHIN fold 0: + cross-subject shuffle + subject-decode control ----
    if do_within:
        wf = load_official_splits(os.path.join(LOCAL, "splits"), "within")[0]
        _, te = get_official_split_windows(data, wf)
        m = AADModel(mode=mode).to(DEV)
        m.load_state_dict(torch.load(os.path.join(
            UP, "results", f"results_aad_within_{tag}", f"fold_0_{mode}_within.pt"),
            map_location=DEV), strict=True)
        T = materialize(data, te)
        subj = win_subj[te]
        kinds_w = ["real", "shuffle", "shuffle_xsubj", "constant", "zeros", "noise"]
        report["within_fold0"] = {"battery": battery(m, T, mode, kinds_w, subj=subj),
                                  "embedding_diversity": emb_diversity(m, T, mode),
                                  "subject_decode": subject_decode(m, T, mode, data, te)}
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eeg")
    ap.add_argument("--tag", default="w10_h5")
    ap.add_argument("--within", type=int, default=1)
    a = ap.parse_args()
    rep = run(a.mode, a.tag, bool(a.within))
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                      "diagnostics", f"{a.mode}_{a.tag}.json")
    os.makedirs(os.path.dirname(od), exist_ok=True)
    json.dump(rep, open(od, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\nsaved -> {od}")
