"""Preprocessing positive control: does the repo's preprocessed EEG still carry
decodable neural/individual structure, or did preprocessing discard it?

Decode (a) SUBJECT identity and (b) ATTENDED speaker directly from the RAW
preprocessed EEG (their exact 64 Hz pipeline), with a simple linear probe on
per-channel log-variance + band powers, trial-disjoint. Compare subject-decode
from raw EEG vs from the model's collapsed brain_enc. If raw EEG >> brain_enc,
the *encoder* collapsed the signal (not preprocessing); the info was available
and the attention head ignored it.
"""
import os, sys, json
import numpy as np
import torch
from scipy.signal import welch

UP = "/fs/scratch/PAS2301/alialavi/MAESTRO_upstream"
LOCAL = "/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset"
sys.path.insert(0, os.path.join(UP, "scripts"))
from dataloader import (build_dataset, load_official_splits,             # noqa: E402
                        get_official_split_windows)
from model_classification import AADModel                                # noqa: E402
from sklearn.linear_model import LogisticRegression                      # noqa: E402
from sklearn.preprocessing import StandardScaler                         # noqa: E402
FS = 64


def raw_feats(eeg):  # eeg: (N, T, 32) preprocessed
    N, T, C = eeg.shape
    logvar = np.log(eeg.var(1) + 1e-8)                     # (N,32)
    f, P = welch(eeg, fs=FS, axis=1, nperseg=min(256, T))  # (N?,) -> P (Nf per axis)
    # welch over axis=1 gives (N, Nf, C)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return np.log(P[:, m, :].mean(1) + 1e-8)           # (N,32)
    return np.concatenate([logvar, band(1, 4), band(4, 8),
                           band(8, 13), band(13, 30)], axis=1)  # (N,160)


def trial_disjoint_decode(X, labels, trial, seed=0, chance_note=""):
    uq = np.unique(trial); rng = np.random.default_rng(seed); rng.shuffle(uq)
    cut = int(0.7 * len(uq)); tr_set = set(uq[:cut].tolist())
    m = np.array([t in tr_set for t in trial])
    if len(np.unique(labels[m])) < 2 or (~m).sum() < 5:
        return None
    sc = StandardScaler().fit(X[m])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[m]), labels[m])
    return float((clf.predict(sc.transform(X[~m])) == labels[~m]).mean())


@torch.no_grad()
def brain_emb(model, eeg, batch=64):
    outs = []
    for s in range(0, eeg.shape[0], batch):
        xb = torch.from_numpy(eeg[s:s+batch])
        outs.append(model.eeg_proj(model.eeg_encoder(xb)).mean(1).numpy())
    return np.concatenate(outs)


def main():
    data = build_dataset(local_path=LOCAL, mode="eeg",
                         cache_dir="/fs/scratch/PAS2301/alialavi/cache/n_gh_newrepo__eeg",
                         window_sec=10.0, hop_sec=5.0)
    wf = load_official_splits(os.path.join(LOCAL, "splits"), "within")[0]
    _, te = get_official_split_windows(data, wf)
    eeg = data["eeg"][te]                                   # (N,640,32) preprocessed
    subj = data["trial_meta_subject"][np.searchsorted(
        data["trial_meta_ids"], data["trial_ids"][te])]
    att = data["att_idxs"][te]
    trial = data["trial_ids"][te]

    Xraw = raw_feats(eeg)
    nsub = len(np.unique(subj))

    # (a) subject from RAW preprocessed EEG
    s_raw = np.mean([trial_disjoint_decode(Xraw, subj, trial, seed=k) for k in range(5)])
    # (b) attended (4-way) from RAW preprocessed EEG
    a_raw = np.mean([trial_disjoint_decode(Xraw, att, trial, seed=k) for k in range(5)])
    # (c) subject from the model's collapsed brain_enc
    m = AADModel(mode="eeg")
    m.load_state_dict(torch.load(os.path.join(
        UP, "results", "results_aad_within_w10_h5", "fold_0_eeg_within.pt"),
        map_location="cpu"), strict=True)
    Xemb = brain_emb(m, eeg)
    s_emb = np.mean([trial_disjoint_decode(Xemb, subj, trial, seed=k) for k in range(5)])

    out = dict(n_windows=int(len(te)), n_subjects=int(nsub),
               subject_chance=1.0/nsub, attended_chance=0.25,
               subject_from_RAW_eeg=float(s_raw),
               subject_from_collapsed_brain_enc=float(s_emb),
               attended_from_RAW_eeg=float(a_raw))
    print(json.dumps(out, indent=2))
    print(f"\nSubject decode: RAW EEG={s_raw:.3f} vs collapsed brain_enc={s_emb:.3f} "
          f"(chance {1.0/nsub:.3f})")
    print(f"Attended(4-way) decode from RAW EEG={a_raw:.3f} (chance 0.25)")
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                      "diagnostics", "raw_probe_eeg.json")
    json.dump(out, open(od, "w"), indent=2)
    print(f"saved -> {od}")


if __name__ == "__main__":
    main()
