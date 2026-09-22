"""Data layer for MERIT: windows, certified candidate sets, protocol splits.

The model layer never sees a file path.  This module owns everything upstream of
the decoder:

  * **windows** -- 64 Hz, per-channel standardised, ``W`` s long at hop ``H`` s,
    for EEG (32 ch), gaze (6), head IMU (6) and scene optical flow (4), plus the
    four co-present talkers' speech envelopes.  Materialised once per
    (modality-set, W, H) into a single ``.npz`` under the scratch cache.
  * **frozen video embeddings** -- V-JEPA 2 ViT-L scene and gaze-centred *fovea*
    vectors (1024-d each), extracted offline per 5 s sub-window and pooled onto
    whatever window grid is requested.  Frozen: no gradient ever reaches pixels.
  * **candidate constructions** -- the four of them, with the model-independent
    audio-only acceptance probe that certifies a construction *before* any
    decoder is trained (``audio_only_probe``).
  * **splits** -- content-disjoint within-subject folds and leave-one-listener-out,
    both with an inner validation carve taken from the training partition only.

The candidate machinery (``quantile_match``, ``build_shifted_imposters``,
``shape_features_batch``, ``audio_only_probe``) is carried over from the
leakage audit in ``analysis/n_gh_checks/fixed/candidates_v2.py``; it is
reproduced here so the released ``src/`` tree is self-contained.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

log = logging.getLogger("data.credit")

FS = 64.0                                     # common sampling grid (Hz)
N_SPEAKERS = 4                                # attendable loudspeakers
MODALITY_CH = {"eeg": 32, "gaze": 6, "imu": 6, "video": 4}
STATIC_DIM = 1024                             # V-JEPA 2 ViT-L embedding width
ORIENT_DIM = 23                               # per-trial gaze/head orienting descriptor
STATIC_DIMS = {"fovea": STATIC_DIM, "scene": STATIC_DIM, "orient": ORIENT_DIM}
TIME_MODS = ("eeg", "gaze", "imu", "video")
STATIC_MODS = ("fovea", "scene", "orient")


# ---------------------------------------------------------------------------
# candidate constructions
# ---------------------------------------------------------------------------
def _zscore_last(x, eps=1e-8):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + eps)


def quantile_match(A, chunk=1000):
    """``A``: (N,K,T) -> (N,K,T).  Force the K candidates of each window onto a
    common marginal (their average order statistics) while preserving each
    candidate's own temporal ordering.

    Per-candidate standardisation fixes only the first two moments, so every
    affine-invariant statistic -- kurtosis, skew, sparsity, quantile ratios,
    relative band powers -- passes through it untouched.  Matching the full
    marginal equalises all of them by construction, and leaves exactly the
    property a neural response tracks: the order in time.
    """
    N, K, T = A.shape
    out = np.empty_like(A, dtype=np.float32)
    ar = np.arange(T)
    for s in range(0, N, chunk):
        a = A[s:s + chunk]
        n = a.shape[0]
        order = np.argsort(a, axis=2, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(ar, (n, K, T)), axis=2)
        ref = np.sort(a, axis=2).mean(axis=1, keepdims=True)              # (n,1,T)
        matched = np.take_along_axis(np.broadcast_to(ref, (n, K, T)), ranks, axis=2)
        out[s:s + chunk] = _zscore_last(matched).astype(np.float32)
    return out


def build_shifted_imposters(trial_ids, window_sec, hop_sec, n_neg=3, seed=0):
    """Same-talker temporal negatives: for every window pick ``n_neg`` windows of
    the *same trial* whose time spans do not overlap it.  Same voice, same
    recording, different segment -- so the candidates are exchangeable and the
    audio-only Bayes accuracy is exactly 1/K.

    Negatives are drawn uniformly among the admissible windows.  Taking the
    temporally furthest one instead biases them toward trial edges, whose onset
    and offset statistics are distinctive; that alone lifted the audio-only
    probe to 0.60 on a chance-0.50 task.
    """
    gap = max(1, int(np.ceil(window_sec / hop_sec)))
    out = np.full((len(trial_ids), n_neg), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n_fallback = 0
    for t in np.unique(trial_ids):
        idx = np.where(trial_ids == t)[0]                  # contiguous, in time order
        n = len(idx)
        for p in range(n):
            valid = [q for q in range(n) if abs(q - p) >= gap]
            rng.shuffle(valid)
            if len(valid) < n_neg:
                n_fallback += 1
                extra = [q for q in range(n) if q != p and q not in valid]
                valid = valid + list(rng.permutation(extra))
            if not valid:                                  # single-window trial
                out[idx[p]] = idx[p]
                continue
            pick = (valid * n_neg)[:n_neg]
            out[idx[p]] = idx[np.asarray(pick)]
    assert (out >= 0).all(), "imposter index not assigned"
    return out, n_fallback


def shape_features_batch(E):
    """Eight scale-free shape statistics of standardised envelopes.  (M,T)->(M,8).

    Every one of them is invariant under ``x -> ax+b`` and therefore survives the
    per-candidate standardisation that is standard practice in this literature.
    """
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew
    M, T = E.shape
    p5, p95 = np.percentile(E, [5, 95], axis=1)
    f, P = welch(E, fs=FS, nperseg=min(256, T), axis=1)
    tot = P.sum(1) + 1e-12

    def band(lo, hi):
        return P[:, (f >= lo) & (f < hi)].sum(1) / tot

    a = np.sort(np.abs(E), axis=1)
    w = np.arange(1, T + 1)
    gini = 2 * (a * w).sum(1) / (T * a.sum(1) + 1e-12) - (T + 1) / T
    return np.stack([kurtosis(E, axis=1), skew(E, axis=1), p95 - p5,
                     (E < -0.5).mean(1), band(0.5, 4), band(4, 8),
                     band(8, 20), gini], axis=1).astype(np.float64)


def audio_only_probe(cands, labels, groups, n_splits=5, seed=0):
    """The acceptance test for a candidate construction, run before any training.

    ``cands`` (N,K,T) are the candidate envelopes the model will actually be fed;
    a logistic classifier on level-free shape features alone tries to name the
    attended one, on content-disjoint folds.  A construction free of acoustic
    confounding must land at 1/K.  This bounds what a *linear* reader can take
    from the candidates -- see the discussion of its limits in the paper.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    N, K, T = cands.shape
    X = shape_features_batch(cands.reshape(N * K, T))
    y = np.zeros(N * K, dtype=np.int64)
    y[np.arange(N) * K + labels] = 1
    g = np.repeat(groups, K)

    oof = np.zeros(N * K)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    pred = oof.reshape(N, K).argmax(1)
    return float((pred == labels).mean())


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------
class _WindowDataset(Dataset):
    """One decision window: the time-series streams, the static descriptors, the
    K candidate envelopes in a shuffled slot order, and the bookkeeping a
    permutation-based evaluation needs.

    When the store holds whole trials (``crop_len`` shorter than the stored
    length) the window is a *crop*. Training draws the offset uniformly at
    random, which multiplies the effective training set instead of reusing five
    fixed positions; evaluation walks a deterministic grid, so a crop-trained
    model is scored on exactly the windows a grid-built cache would have given.
    Candidates are distribution-matched \emph{after} cropping, so the marginals
    the model sees are matched on the segment it is actually shown.
    """

    def __init__(self, store, idx, bank, modalities, static_mods, train, K,
                 seed=0, crop_len=None, n_crops=5, cand_mode="qmatch",
                 audio_feats="env"):
        self.g = np.asarray(idx)
        self.train, self.K, self.seed = train, K, seed
        self.cand_mode, self.audio_feats = cand_mode, audio_feats
        self.mods = tuple(m for m in modalities if store.arrays.get(m) is not None)
        self.static = tuple(m for m in static_mods if store.static.get(m) is not None)
        self.x = {m: torch.from_numpy(store.arrays[m][self.g]) for m in self.mods}
        self.s = {m: torch.from_numpy(store.static[m][self.g]) for m in self.static}
        self.att = store.att_idx[self.g].astype(np.int64)
        self.subj = store.subject[self.g].astype(np.int64)
        self.trial = store.trial_ids[self.g].astype(np.int64)
        self.A = bank["A"][self.g]                                    # (n,K,T) numpy
        self.pos = bank["pos"][self.g]
        self.spk_meaningful = bank["spk_meaningful"]
        T = self.A.shape[-1]
        self.T = T
        self.crop_len = int(crop_len) if crop_len and crop_len < T else None
        self.n_crops = int(n_crops) if self.crop_len else 1
        if self.crop_len:
            span = T - self.crop_len
            self.grid = [int(round(k * span / max(1, self.n_crops - 1)))
                         for k in range(self.n_crops)]

    def __len__(self):
        return len(self.g) * self.n_crops

    def _perm(self, key):
        # Slots are shuffled per window, so slot index carries no information. At
        # evaluation the shuffle is seeded by the sample's global key and is
        # therefore identical across every configuration compared.
        if self.train:
            return torch.randperm(self.K)
        return torch.from_numpy(
            np.random.default_rng(int(key) + self.seed).permutation(self.K))

    def _same_trial_negatives(self, i, off, rng):
        """Same-talker temporal negatives drawn as non-overlapping crops of this
        trial. Whole-trial storage gives many more admissible offsets than a
        five-window grid, and the offset is drawn uniformly, so negatives are not
        biased toward trial edges."""
        L, T = self.crop_len, self.T
        att = self.A[i, int(self.att[i])]
        lo, hi = 0, T - L
        cand = [att[off:off + L]]
        tries, picked = 0, []
        while len(picked) < self.K - 1 and tries < 200:
            o = int(rng.integers(lo, hi + 1))
            if abs(o - off) >= L and all(abs(o - q) >= L for q in picked):
                picked.append(o)
            tries += 1
        while len(picked) < self.K - 1:                    # degenerate fallback
            picked.append((off + L) % max(1, hi + 1))
        cand += [att[o:o + L] for o in picked]
        return np.stack(cand, 0), 0

    def __getitem__(self, k):
        i = k // self.n_crops
        key = int(self.g[i]) * 16 + (k % self.n_crops)
        if self.crop_len:
            rng = np.random.default_rng(key + self.seed + (0 if not self.train else
                                                           np.random.randint(1 << 30)))
            off = (int(rng.integers(0, self.T - self.crop_len + 1)) if self.train
                   else self.grid[k % self.n_crops])
            if self.cand_mode.startswith("shifted"):
                A, pos = self._same_trial_negatives(i, off, rng)
            else:
                A, pos = self.A[i][:, off:off + self.crop_len], int(self.pos[i])
            if self.cand_mode.endswith("qm") or self.cand_mode == "qmatch":
                A = quantile_match(A[None])[0]
            else:
                A = _zscore_last(A).astype(np.float32)
            xs = {m: self.x[m][i, off:off + self.crop_len] for m in self.mods}
        else:
            A, pos = np.asarray(self.A[i], np.float32), int(self.pos[i])
            xs = {m: self.x[m][i] for m in self.mods}
        A = np.ascontiguousarray(A, np.float32)
        if self.audio_feats == "env_onset":
            # Half-wave-rectified envelope derivative. It is a deterministic
            # function of the envelope, so it adds no information a decoder could
            # read from the audio alone -- the acceptance probe is unchanged --
            # but the cortical response tracks onsets more strongly than the
            # envelope itself, so it is easier for the coupling branch to find.
            d = np.diff(A, axis=-1, prepend=A[:, :1])
            on = np.maximum(d, 0.0)
            on = (on - on.mean(-1, keepdims=True)) / (on.std(-1, keepdims=True) + 1e-8)
            A = np.stack([A, on.astype(np.float32)], axis=-1)        # (K,L,2)
        else:
            A = A[..., None]                                          # (K,L,1)
        A = torch.from_numpy(np.ascontiguousarray(A, np.float32))

        perm = self._perm(key)
        cands = A[perm]
        att_pos = int((perm == pos).nonzero()[0].item())
        spk_of_slot = (perm.clone() if self.spk_meaningful else
                       torch.full((self.K,), int(self.att[i]), dtype=torch.long))
        item = dict(xs)
        item.update({m: self.s[m][i] for m in self.static})
        return (item, cands.contiguous(), att_pos, spk_of_slot,
                int(self.att[i]), int(self.subj[i]), int(self.trial[i]))


def collate(batch):
    mods = {m: torch.stack([b[0][m] for b in batch]) for m in batch[0][0]}
    cands = torch.stack([b[1] for b in batch])                             # (B,K,T,1)
    audio = [cands[:, k] for k in range(cands.shape[1])]
    return (mods, audio,
            torch.tensor([b[2] for b in batch], dtype=torch.long),
            torch.stack([b[3] for b in batch]),
            torch.tensor([b[4] for b in batch], dtype=torch.long),
            torch.tensor([b[5] for b in batch], dtype=torch.long),
            torch.tensor([b[6] for b in batch], dtype=torch.long))


class SubjectBatchSampler(Sampler):
    """Every batch comes from one listener.

    The contrastive term's in-batch negatives must be within-listener: raw
    preprocessed EEG identifies the listener at 0.90 in a sixteen-way test
    against 0.0625 chance, so a cross-listener batch is solved by listener
    identity alone and teaches nothing about attention.
    """

    def __init__(self, subjects, batch_size, shuffle=True, seed=0, min_batch=4):
        self.groups = [np.where(subjects == s)[0] for s in np.unique(subjects)]
        self.bs, self.shuffle, self.seed, self.min_batch = \
            batch_size, shuffle, seed, min_batch
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for g in self.groups:
            g = g.copy()
            if self.shuffle:
                rng.shuffle(g)
            for s in range(0, len(g), self.bs):
                b = g[s:s + self.bs]
                if len(b) >= self.min_batch:
                    batches.append(b.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(max(0, len(g) // self.bs) for g in self.groups)


@dataclass
class MeritView:
    """The constant data->model interface: a materialisable set of windows."""
    store: "WindowStore"
    bank: dict
    idx: np.ndarray
    modalities: tuple
    static_mods: tuple
    K: int
    train: bool = False
    seed: int = 0
    crop_len: Optional[int] = None
    n_crops: int = 1
    cand_mode: str = "qmatch"
    audio_feats: str = "env"

    @property
    def _reps(self):
        T = self.bank["A"].shape[-1]
        return self.n_crops if (self.crop_len and self.crop_len < T) else 1

    def __len__(self):
        return len(self.idx) * self._reps

    def as_dataset(self):
        return _WindowDataset(self.store, self.idx, self.bank, self.modalities,
                              self.static_mods, self.train, self.K, self.seed,
                              crop_len=self.crop_len, n_crops=self.n_crops,
                              cand_mode=self.cand_mode, audio_feats=self.audio_feats)

    def as_torch_loader(self, batch_size, subject_batches=False, num_workers=0):
        ds = self.as_dataset()
        if subject_batches:
            subj = np.repeat(self.store.subject[self.idx], self._reps)
            sampler = SubjectBatchSampler(subj, batch_size, seed=self.seed)
            return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                              num_workers=num_workers)
        return DataLoader(ds, batch_size=batch_size, shuffle=self.train,
                          collate_fn=collate, num_workers=num_workers)

    def strata(self):
        """Groupings used for stratified permutation nulls (see the evaluator)."""
        r = self._reps
        return {"pos": np.tile(np.arange(r), len(self.idx)),
                "trial": np.repeat(self.store.trial_ids[self.idx], r)}


# ---------------------------------------------------------------------------
# store + module
# ---------------------------------------------------------------------------
@dataclass
class WindowStore:
    arrays: dict                      # modality -> (N,T,C) float32, or None
    static: dict                      # "fovea"/"scene" -> (N,1024) float32, or None
    audio: list                       # K arrays (N,T,1)
    att_idx: np.ndarray               # (N,) attended loudspeaker 0..3
    trial_ids: np.ndarray             # (N,)
    subject: np.ndarray               # (N,)
    content: np.ndarray               # (N,) stimulus-content id (e.g. "eval_007")
    position: np.ndarray              # (N,) index of the window within its trial
    window_sec: float
    hop_sec: float

    @property
    def n(self):
        return len(self.att_idx)


def _position_in_trial(trial_ids):
    _, first = np.unique(trial_ids, return_index=True)
    start = np.zeros(len(trial_ids), dtype=np.int64)
    start[first] = first
    np.maximum.accumulate(start, out=start)
    return np.arange(len(trial_ids)) - start


class MeritDataModule:
    """Builds/loads windows, joins the frozen video embeddings, assembles the
    candidate bank, and yields protocol splits.  Nothing below this line knows
    about models; nothing above it knows about files."""

    name = "merit"

    def __init__(self, subjects="all", cache_root="/fs/scratch/PAS2301/alialavi/cache",
                 dataset_path="/fs/scratch/PAS2301/alialavi/maestro-eeg-dataset",
                 vjepa_dir="/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa",
                 modalities=("eeg", "gaze", "imu", "video"), static_mods=("fovea",),
                 window_sec=10.0, hop_sec=5.0, cand_mode="qmatch", K=4,
                 n_folds=5, val_frac=0.2, content_holdout=0.2, seed=0,
                 crop_len=None, n_train_crops=5, n_eval_crops=5,
                 euclidean_align=False, orient_feats=False, audio_feats="env",
                 limit_subjects=None, limit_content=None,
                 builder_path="/fs/scratch/PAS2301/alialavi/MAESTRO_upstream/scripts"):
        self.subjects = subjects
        self.cache_root = cache_root
        self.dataset_path = dataset_path
        self.vjepa_dir = vjepa_dir
        self.modalities = tuple(modalities)
        self.static_mods = tuple(static_mods)
        self.window_sec, self.hop_sec = float(window_sec), float(hop_sec)
        self.cand_mode, self.K = cand_mode, int(K)
        self.n_folds, self.val_frac = int(n_folds), float(val_frac)
        self.content_holdout, self.seed = float(content_holdout), int(seed)
        self.builder_path = builder_path
        # smoke-test knobs: subset the materialised store instead of rebuilding
        # a separate cache for a handful of listeners/stimuli.
        # whole-trial storage + random crops: more training windows from the same
        # recordings, and an evaluation grid that reproduces the fixed-window cache.
        self.crop_len = crop_len
        self.n_train_crops, self.n_eval_crops = int(n_train_crops), int(n_eval_crops)
        # label-free per-listener covariance whitening of the EEG
        self.euclidean_align = bool(euclidean_align)
        # per-trial orienting descriptors recovered from the RAW gaze/IMU, which the
        # per-trial standardisation in the window pipeline destroys
        self.orient_feats = bool(orient_feats)
        # "env" or "env_onset": the second channel is the rectified derivative
        self.audio_feats = str(audio_feats)
        self.limit_subjects = limit_subjects
        self.limit_content = limit_content
        self.store: Optional[WindowStore] = None
        self.bank: Optional[dict] = None
        self.probe: Optional[float] = None

    # -- build / load ------------------------------------------------------
    def _cache_file(self):
        mode = "_".join(m for m in TIME_MODS if m in self.modalities)
        tag = "all" if self.subjects == "all" else str(len(self.subjects))
        return os.path.join(
            self.cache_root,
            f"n_gh_fixed_data__{mode}_w{self.window_sec:g}_h{self.hop_sec:g}_{tag}.npz")

    def _build_windows(self):
        """Materialise the window arrays.  Cached as one npz per (mods, W, H)."""
        path = self._cache_file()
        if os.path.exists(path):
            log.info("windows <- cache %s", path)
            z = np.load(path, allow_pickle=False)
            d = {k: z[k] for k in z.files if not k.startswith("audio_")}
            d["audio"] = [z[f"audio_{i}"] for i in range(N_SPEAKERS)]
            return d
        log.info("windows: cache miss, building %s", path)
        import sys
        if self.builder_path not in sys.path:
            sys.path.insert(0, self.builder_path)
        from dataloader import build_dataset            # noqa: E402  (offline builder)
        mode = "_".join(m for m in TIME_MODS if m in self.modalities)
        d = build_dataset(local_path=self.dataset_path, mode=mode,
                          subjects=self.subjects,
                          cache_dir=os.path.join(self.cache_root,
                                                 f"n_gh_newrepo__{mode}"),
                          window_sec=self.window_sec, hop_sec=self.hop_sec)
        save = {k: v for k, v in d.items() if k != "audio" and v is not None}
        save["trial_meta_tid"] = save["trial_meta_tid"].astype("<U24")
        save.update({f"audio_{i}": a for i, a in enumerate(d["audio"])})
        tmp = path + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, **save)
        os.replace(tmp, path)
        log.info("windows -> cached %s", path)
        return d

    def _load_vjepa(self, subject, trial_k, position):
        """Pool the 5 s V-JEPA 2 sub-window embeddings onto the requested grid.

        Frozen encoder, extracted offline; the window at position ``p`` starts at
        ``p*hop`` s and we average every 5 s sub-window fully inside it.
        """
        vj = [m for m in self.static_mods if m in ("fovea", "scene")]
        out = {m: np.zeros((len(subject), STATIC_DIM), np.float32) for m in vj}
        for s in np.unique(subject):
            f = os.path.join(self.vjepa_dir, f"s{int(s)}_win5.npz")
            if not os.path.exists(f):
                log.warning("V-JEPA cache missing for subject %d (zero-filled)", s)
                continue
            z = np.load(f, allow_pickle=True)
            tk, ws = z["trial_k"], z["win_start"].astype(np.float64)
            sel = np.where(subject == s)[0]
            for m in vj:
                emb = z[m]
                for i in sel:
                    lo = position[i] * self.hop_sec
                    hi = lo + self.window_sec - 5.0 + 1e-6
                    k = np.where((tk == trial_k[i]) & (ws >= lo - 1e-6) & (ws <= hi))[0]
                    if len(k) == 0:                      # window shorter than 5 s
                        k = np.where(tk == trial_k[i])[0]
                        if len(k) == 0:
                            continue
                        k = k[np.argmin(np.abs(ws[k] - lo))][None]
                    out[m][i] = emb[k].mean(0)
        return out

    def _euclidean_align(self, eeg, subject):
        """Label-free per-listener whitening: x <- R^(-1/2) x with R the listener's
        mean channel covariance. Standard for cross-subject EEG transfer and
        admissible under leave-one-listener-out, since no label is consulted."""
        from scipy.linalg import fractional_matrix_power
        out = eeg
        for sb in np.unique(subject):
            sel = np.where(subject == sb)[0]
            X = eeg[sel]                                   # (n,T,C)
            C = np.einsum("ntc,ntd->cd", X, X) / (X.shape[0] * X.shape[1])
            C += 1e-6 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
            W = np.real(fractional_matrix_power(C, -0.5)).astype(np.float32)
            out[sel] = np.einsum("ntc,cd->ntd", X, W)
        log.info("euclidean alignment applied to %d listeners", len(np.unique(subject)))
        return out

    def _orient_features(self, subject, trial_k):
        """Per-trial orienting descriptors from the RAW gaze and head-motion
        recordings.

        The window pipeline standardises every stream per trial, which removes
        each trial's mean gaze direction -- precisely the quantity that indicates
        which loudspeaker the listener turned toward. We recover it here from the
        source parquet, summarise each trial, and standardise *within listener*,
        which is label-free and therefore admissible under LOSO.
        """
        import pandas as pd
        root = os.path.join(self.dataset_path, "data")
        cache = os.path.join(self.cache_root, "merit_orient_feats.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=False)
            key = np.char.add(z["subject"].astype(str), np.char.add("_", z["trial"].astype(str)))
            want = np.char.add(subject.astype(str), np.char.add("_", trial_k.astype(str)))
            idx = {k: i for i, k in enumerate(key)}
            F = np.stack([z["F"][idx[k]] if k in idx else np.zeros(z["F"].shape[1], np.float32)
                          for k in want])
            log.info("orienting features <- cache %s (%d dims)", cache, F.shape[1])
            return F.astype(np.float32)

        rows, keys = [], []
        for sb in range(1, 17):
            for tk in range(1, 101):
                g = os.path.join(root, "gaze", f"subject=S{sb:02d}", f"trial=eval_{tk:03d}.parquet")
                m = os.path.join(root, "imu", f"subject=S{sb:02d}", f"trial=eval_{tk:03d}.parquet")
                f = []
                if os.path.exists(g):
                    d = pd.read_parquet(g)
                    x, y = d["gaze2d_x"].to_numpy(), d["gaze2d_y"].to_numpy()
                    v = np.isfinite(x) & np.isfinite(y)
                    gx, gy, gz = (d[c].to_numpy() for c in ("gaze3d_x", "gaze3d_y", "gaze3d_z"))
                    az = np.arctan2(gx, np.abs(gz) + 1e-6)
                    el = np.arctan2(gy, np.abs(gz) + 1e-6)
                    sp = np.hypot(np.diff(x, prepend=x[:1]), np.diff(y, prepend=y[:1]))
                    f += [np.nanmean(x), np.nanstd(x), np.nanmean(y), np.nanstd(y),
                          np.nanmean(az), np.nanstd(az), np.nanmean(el), np.nanstd(el),
                          np.nanmean(d["L_pupil"].to_numpy()), np.nanmean(d["R_pupil"].to_numpy()),
                          float(v.mean()), float(np.isfinite(d["L_pupil"].to_numpy()).mean()),
                          float(np.isfinite(d["R_pupil"].to_numpy()).mean()),
                          float(np.nanmean(sp)), float(np.nanpercentile(sp[np.isfinite(sp)], 95)
                                                       if np.isfinite(sp).any() else 0.0)]
                else:
                    f += [0.0] * 15
                if os.path.exists(m):
                    d = pd.read_parquet(m)
                    a = d[["ax", "ay", "az"]].to_numpy()
                    w = d[["gx", "gy", "gz"]].to_numpy()
                    f += list(np.nanmean(w, 0)) + list(np.nanstd(w, 0)) + \
                         [float(np.nanmean(np.linalg.norm(a, axis=1))),
                          float(np.nanstd(np.linalg.norm(a, axis=1)))]
                else:
                    f += [0.0] * 8
                rows.append(np.nan_to_num(np.array(f, np.float32)))
                keys.append((sb, tk))
        F = np.stack(rows)
        sb_all = np.array([k[0] for k in keys]); tk_all = np.array([k[1] for k in keys])
        for sb in np.unique(sb_all):                       # per-listener standardisation
            m = sb_all == sb
            F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-6)
        np.savez(cache, F=F, subject=sb_all, trial=tk_all)
        log.info("orienting features -> %s (%d trials x %d dims)", cache, *F.shape)
        idx = {(a, b): i for i, (a, b) in enumerate(keys)}
        return np.stack([F[idx[(int(a), int(b))]] for a, b in zip(subject, trial_k)])

    def prepare(self):
        d = self._build_windows()
        order = np.argsort(d["trial_meta_ids"])
        meta_ids = d["trial_meta_ids"][order]
        j = np.searchsorted(meta_ids, d["trial_ids"])
        subject = d["trial_meta_subject"][order][j]
        content = d["trial_meta_tid"][order][j]
        position = _position_in_trial(d["trial_ids"])
        trial_k = np.array([int(str(c).split("_")[-1]) for c in content])

        keep = np.ones(len(subject), bool)
        if self.limit_subjects:
            keep &= np.isin(subject, list(self.limit_subjects))
        if self.limit_content:
            keep &= np.isin(trial_k, list(range(1, int(self.limit_content) + 1)))
        sel = np.where(keep)[0]
        if len(sel) < len(subject):
            log.info("subsetting store: %d -> %d windows", len(subject), len(sel))
            for k in TIME_MODS:
                if d.get(k) is not None:
                    d[k] = d[k][sel]
            d["audio"] = [a[sel] for a in d["audio"]]
            d["att_idxs"] = d["att_idxs"][sel]
            d["trial_ids"] = d["trial_ids"][sel]
            subject, content, position, trial_k = (subject[sel], content[sel],
                                                   position[sel], trial_k[sel])

        if self.euclidean_align and d.get("eeg") is not None:
            d["eeg"] = self._euclidean_align(d["eeg"], subject)
        arrays = {m: (d[m] if d.get(m) is not None else None) for m in TIME_MODS}
        static = {}
        if any(m in ("fovea", "scene") for m in self.static_mods):
            static = self._load_vjepa(subject, trial_k, position)
        if "orient" in self.static_mods:
            static["orient"] = self._orient_features(subject, trial_k)
        self.store = WindowStore(
            arrays=arrays, static=static, audio=d["audio"],
            att_idx=d["att_idxs"].astype(np.int64), trial_ids=d["trial_ids"],
            subject=subject.astype(np.int64), content=content, position=position,
            window_sec=self.window_sec, hop_sec=self.hop_sec)
        self.bank = self._make_bank()
        log.info("prepared %d windows | mods=%s static=%s | cand=%s K=%d",
                 self.store.n, self.modalities, self.static_mods,
                 self.cand_mode, self.K)
        return self

    # -- candidates --------------------------------------------------------
    def _make_bank(self):
        st, K = self.store, self.K
        if self.crop_len:
            # Cropping defers both distribution matching and negative selection to
            # __getitem__, so the bank carries the raw per-speaker envelopes.
            A = np.stack([st.audio[k][:, :, 0] for k in range(N_SPEAKERS)], axis=1)
            spk = not self.cand_mode.startswith("shifted")
            return {"mode": self.cand_mode, "A": np.ascontiguousarray(A, np.float32),
                    "pos": st.att_idx.astype(np.int64), "spk_meaningful": spk}
        if self.cand_mode in ("raw", "qmatch"):
            A = np.stack([st.audio[k][:, :, 0] for k in range(K)], axis=1)
            if self.cand_mode == "qmatch":
                A = quantile_match(A)
            return {"mode": self.cand_mode, "A": np.ascontiguousarray(A, np.float32),
                    "pos": st.att_idx.astype(np.int64), "spk_meaningful": True}
        if self.cand_mode in ("shifted", "shifted_qm"):
            att = np.stack([st.audio[a][i, :, 0]
                            for i, a in enumerate(st.att_idx)]).astype(np.float32)
            imp, n_fb = build_shifted_imposters(st.trial_ids, self.window_sec,
                                                self.hop_sec, n_neg=K - 1,
                                                seed=self.seed)
            A = np.stack([att] + [att[imp[:, j]] for j in range(K - 1)], axis=1)
            if self.cand_mode == "shifted_qm":
                A = quantile_match(A)
            log.info("same-talker negatives: %.1f%% needed the overlap fallback",
                     100 * n_fb / max(st.n, 1))
            return {"mode": self.cand_mode, "A": np.ascontiguousarray(A, np.float32),
                    "pos": np.zeros(st.n, np.int64), "spk_meaningful": False}
        raise ValueError(self.cand_mode)

    def certify(self, n_splits=5):
        """Run the audio-only acceptance probe on the assembled candidate bank."""
        if self.probe is None:
            self.probe = audio_only_probe(self.bank["A"], self.bank["pos"],
                                          self.store.content, n_splits=n_splits)
            log.info("audio-only probe on '%s' K=%d: %.4f (chance %.4f)",
                     self.cand_mode, self.K, self.probe, 1 / self.K)
        return self.probe

    # -- splits ------------------------------------------------------------
    def _view(self, idx, train=False):
        return MeritView(self.store, self.bank, np.asarray(idx), self.modalities,
                          self.static_mods, self.K, train=train, seed=self.seed,
                          crop_len=self.crop_len,
                          n_crops=(self.n_train_crops if train else self.n_eval_crops),
                          cand_mode=self.cand_mode, audio_feats=self.audio_feats)

    def splits(self, protocol="within"):
        """Yield ``(name, train_view, val_view, test_view)``.

        ``within``: trials partitioned by *stimulus content*, so no stimulus heard
        during training reappears at test; all listeners appear on both sides.
        ``loso``: one listener held out per fold, with a global content holdout
        layered on top, so the held-out listener is novel in identity *and* in
        content.  The inner validation set is always carved from the training
        partition -- content-disjoint in the first protocol, listener-disjoint in
        the second -- and the test partition is scored exactly once.
        """
        st = self.store
        rng = np.random.default_rng(self.seed)
        contents = np.unique(st.content)
        if protocol == "within":
            perm = rng.permutation(len(contents))
            folds = np.array_split(perm, self.n_folds)
            for f, hold in enumerate(folds):
                te_c = set(contents[hold])
                te = np.where(np.isin(st.content, list(te_c)))[0]
                rest_c = np.array([c for c in contents if c not in te_c])
                n_val = max(1, int(round(self.val_frac * len(rest_c))))
                val_c = set(rng.choice(rest_c, n_val, replace=False))
                vl = np.where(np.isin(st.content, list(val_c)))[0]
                tr = np.where(~np.isin(st.content, list(te_c | val_c)))[0]
                yield f"within_f{f}", self._view(tr, True), self._view(vl), self._view(te)
        elif protocol == "loso":
            n_hold = max(1, int(round(self.content_holdout * len(contents))))
            hold_c = set(rng.choice(contents, n_hold, replace=False))
            held = np.isin(st.content, list(hold_c))
            subs = np.unique(st.subject)
            for s in subs:
                te = np.where((st.subject == s) & held)[0]
                pool = np.where((st.subject != s) & ~held)[0]
                # inner validation is listener-disjoint from the training set too
                other = np.array([u for u in subs if u != s])
                n_val = max(1, int(round(self.val_frac * len(other))))
                val_s = set(rng.choice(other, n_val, replace=False))
                vl = pool[np.isin(st.subject[pool], list(val_s))]
                tr = pool[~np.isin(st.subject[pool], list(val_s))]
                yield f"loso_s{int(s)}", self._view(tr, True), self._view(vl), self._view(te)
        else:
            raise ValueError(protocol)


def role_separation(cands, labels, names=None):
    """How far apart are target and masker candidates in *shape*?

    ``cands`` (N,K,T) are the standardised envelopes the model is fed.  Every
    statistic here is invariant under ``x -> ax+b`` and therefore survives
    per-candidate standardisation untouched, so a separation reported here is a
    cue the decoder can read without ever consulting the recording.  Reports the
    AUC of a target-vs-masker ranking and Cohen's *d*, per statistic.
    """
    names = names or ["kurtosis", "skew", "p95-p5", "silence frac",
                      "rel. power 0.5-4 Hz", "rel. power 4-8 Hz",
                      "rel. power 8-20 Hz", "Gini sparsity"]
    N, K, T = cands.shape
    X = shape_features_batch(cands.reshape(N * K, T)).reshape(N, K, -1)
    is_target = np.zeros((N, K), bool)
    is_target[np.arange(N), labels] = True
    rows = []
    for j, nm in enumerate(names):
        t = X[:, :, j][is_target]
        m = X[:, :, j][~is_target]
        # AUC = P(target statistic > masker statistic), by rank
        order = np.argsort(np.concatenate([t, m]), kind="stable")
        ranks = np.empty(len(order), float)
        ranks[order] = np.arange(1, len(order) + 1)
        auc = (ranks[:len(t)].sum() - len(t) * (len(t) + 1) / 2) / (len(t) * len(m))
        sd = np.sqrt((t.var(ddof=1) + m.var(ddof=1)) / 2) + 1e-12
        rows.append({"statistic": nm, "auc": float(auc),
                     "cohen_d": float((t.mean() - m.mean()) / sd),
                     "target_mean": float(t.mean()), "masker_mean": float(m.mean())})
    # peak-to-RMS of the standardised envelope: a pure dynamics measure that no
    # gain can alter, reported in dB
    crest = 20 * np.log10(np.abs(cands).max(-1) / (np.sqrt((cands ** 2).mean(-1)) + 1e-12)
                          + 1e-12)
    t, m = crest[is_target], crest[~is_target]
    sd = np.sqrt((t.var(ddof=1) + m.var(ddof=1)) / 2) + 1e-12
    rows.append({"statistic": "envelope crest (dB)", "auc": float(np.nan),
                 "cohen_d": float((t.mean() - m.mean()) / sd),
                 "target_mean": float(t.mean()), "masker_mean": float(m.mean())})
    return rows


__all__ = ["MeritDataModule", "role_separation", "MeritView", "WindowStore", "quantile_match",
           "build_shifted_imposters", "audio_only_probe", "shape_features_batch",
           "MODALITY_CH", "STATIC_DIM", "STATIC_DIMS", "ORIENT_DIM", "TIME_MODS",
           "STATIC_MODS", "N_SPEAKERS"]
