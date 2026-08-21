"""Dataset / samplers for the fixed model.

Differences from upstream `AADDataset`:
  * supports the three candidate constructions (raw / qmatch / shifted);
  * returns the candidate PERMUTATION, so a spatial head that predicts the
    fixed loudspeaker index can be mapped into slot order;
  * returns the SUBJECT id, so the contrastive term's negatives can be
    restricted to within-subject (raw EEG decodes subject identity at 0.90 —
    a cross-subject contrastive batch is solved by subject ID alone and
    teaches the encoder nothing about attention).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from candidates_v2 import quantile_match, build_shifted_imposters

MODALITIES = ("eeg", "video", "gaze", "imu")


def subject_per_window(data):
    return data["trial_meta_subject"][
        np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]


def position_in_trial(data):
    """Index of each window within its trial (windows of a trial are contiguous
    and in temporal order).  Used to build a position-stratified shuffle null."""
    tr = data["trial_ids"]
    _, first = np.unique(tr, return_index=True)
    start = np.zeros(len(tr), dtype=np.int64)
    start[first] = first
    np.maximum.accumulate(start, out=start)
    return np.arange(len(tr)) - start


def content_per_window(data):
    return data["trial_meta_tid"][
        np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]


def make_audio_bank(data, cand_mode, window_sec, hop_sec, n_cand=4, seed=0):
    """Builds the candidate bank once for the whole dataset (split-independent).

    Returns {"mode", "A": (N,K,T), "pos": (N,), "spk_meaningful": bool} where
    A[:, k] is candidate k and pos[i] is the index of the correct one.

      raw / qmatch          -> candidates are the K co-present talkers, so
                               candidate index == loudspeaker index
      shifted / shifted_qm  -> candidates are the attended talker at K different
                               non-overlapping times, so candidate index carries
                               no spatial meaning
    """
    N = len(data["trial_ids"])
    if cand_mode in ("raw", "qmatch"):
        A = np.stack([data["audio"][k][:, :, 0] for k in range(n_cand)], axis=1)
        if cand_mode == "qmatch":
            A = quantile_match(A)
        return {"mode": cand_mode, "A": np.ascontiguousarray(A, dtype=np.float32),
                "pos": data["att_idxs"].astype(np.int64), "spk_meaningful": True}

    if cand_mode in ("shifted", "shifted_qm"):
        att = np.stack([data["audio"][a][i, :, 0]
                        for i, a in enumerate(data["att_idxs"])]).astype(np.float32)
        imp, n_fb = build_shifted_imposters(data["trial_ids"], window_sec, hop_sec,
                                            n_neg=n_cand - 1, seed=seed)
        A = np.stack([att] + [att[imp[:, j]] for j in range(n_cand - 1)], axis=1)
        if cand_mode == "shifted_qm":
            A = quantile_match(A)
        print(f"  shifted negatives: {n_fb} windows needed the overlap fallback "
              f"({100*n_fb/max(N,1):.1f}%)")
        return {"mode": cand_mode, "A": np.ascontiguousarray(A, dtype=np.float32),
                "pos": np.zeros(N, dtype=np.int64), "spk_meaningful": False}

    raise ValueError(cand_mode)


class AADDatasetV2(Dataset):
    def __init__(self, data, window_idx, bank, modalities=("eeg",),
                 train=True, n_cand=4, seed=0):
        self.gidx = np.asarray(window_idx)
        self.train = train
        self.K = n_cand
        self.bank = bank
        self.mods = tuple(m for m in modalities if data.get(m) is not None)
        self.x = {m: torch.from_numpy(data[m][self.gidx]) for m in self.mods}
        self.att = data["att_idxs"][self.gidx].astype(np.int64)
        self.subj = subject_per_window(data)[self.gidx].astype(np.int64)
        self.A = torch.from_numpy(bank["A"][self.gidx])              # (n,K,T)
        self.pos = bank["pos"][self.gidx]
        self.spk_meaningful = bank["spk_meaningful"]

    def __len__(self):
        return len(self.gidx)

    def _perm(self, i):
        if self.train:
            return torch.randperm(self.K)
        return torch.from_numpy(np.random.default_rng(int(self.gidx[i]))
                                .permutation(self.K))

    def __getitem__(self, i):
        perm = self._perm(i)
        cands = self.A[i][perm]                                      # (K,T)
        att_pos = int((perm == int(self.pos[i])).nonzero()[0].item())
        # perm[k] is the loudspeaker index sitting in slot k, where that is
        # meaningful; for same-source negatives every slot is the same talker
        spk_of_slot = (perm.clone() if self.spk_meaningful else
                       torch.full((self.K,), int(self.att[i]), dtype=torch.long))
        return (
            {m: self.x[m][i] for m in self.mods},
            cands.unsqueeze(-1).contiguous(),                        # (K,T,1)
            att_pos, spk_of_slot, int(self.att[i]), int(self.subj[i]),
        )


def collate_v2(batch):
    mods = {m: torch.stack([b[0][m] for b in batch]) for m in batch[0][0]}
    cands = torch.stack([b[1] for b in batch])                       # (B,K,T,1)
    audio = [cands[:, k] for k in range(cands.shape[1])]             # K x (B,T,1)
    return (mods, audio,
            torch.tensor([b[2] for b in batch], dtype=torch.long),
            torch.stack([b[3] for b in batch]),
            torch.tensor([b[4] for b in batch], dtype=torch.long),
            torch.tensor([b[5] for b in batch], dtype=torch.long))


class SubjectBatchSampler(Sampler):
    """Every batch comes from a single subject, so the contrastive term's
    in-batch negatives are within-subject."""

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
