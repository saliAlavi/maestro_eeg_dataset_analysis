"""The public two-talker corpora, presented through the same interface as MAESTRO.

Subclassing ``MeritDataModule`` and overriding only ``prepare`` means the
runner, the model, the permutation battery and the Shapley code all run
unchanged on KU~Leuven, DTU and NJU.  That is the claim being tested: the
coupling branch and the acceptance probe are corpus-independent, and only the
behavioural streams are not.

Each corpus reduces to ``K=2`` candidates -- the attended envelope and the
competing one -- so chance is ``0.5``, and the window store carries a single
EEG stream with whatever montage that corpus used.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from .merit_data import MeritDataModule, WindowStore, _position_in_trial
from .external_aad import FS, LOADERS

log = logging.getLogger("data.external")

CORPUS_CH = {"kul": 64, "dtu": 64, "nju": 32}


class ExternalDataModule(MeritDataModule):
    """One public corpus, windowed into the store the rest of the stack expects."""

    name = "external"

    def __init__(self, corpus="nju", **kw):
        kw.setdefault("modalities", ("eeg",))
        kw.setdefault("static_mods", ())
        kw.setdefault("K", 2)
        super().__init__(**kw)
        self.corpus = str(corpus)
        self.eeg_channels = CORPUS_CH[self.corpus]
        if self.K != 2:
            raise ValueError("the public corpora are two-talker; K must be 2")

    # -- build -------------------------------------------------------------
    def _cache_path(self):
        return os.path.join(
            self.cache_root,
            f"external__{self.corpus}_w{self.window_sec:g}_h{self.hop_sec:g}.npz")

    def prepare(self):
        p = self._cache_path()
        if os.path.exists(p):
            try:
                z = np.load(p, allow_pickle=False)
            except Exception as e:                    # half-written by a peer job
                log.warning("cache %s unreadable (%s); rebuilding", p, e)
                z = None
        else:
            z = None
        if z is not None:
            eeg, A = z["eeg"], z["audio"]
            att, tid = z["att_idx"], z["trial_ids"]
            sub, cont, pos = z["subject"], z["content"], z["position"]
            log.info("external/%s: loaded %d windows from cache", self.corpus, len(att))
        else:
            eeg, A, att, tid, sub, cont = self._build()
            pos = _position_in_trial(tid)
            os.makedirs(self.cache_root, exist_ok=True)
            # Within- and LOSO tasks for one corpus run concurrently and would
            # otherwise write the same file at the same time; build under a
            # per-process name and rename, which is atomic on this filesystem.
            # np.savez appends ".npz" unless the name already ends in it, so the
            # temp name must too or the rename below looks for the wrong file.
            tmp = p[:-4] + f".{os.getpid()}.tmp.npz"
            np.savez(tmp, eeg=eeg, audio=A, att_idx=att, trial_ids=tid,
                     subject=sub, content=cont, position=pos)
            os.replace(tmp, p)
            log.info("external/%s: built %d windows -> %s", self.corpus, len(att), p)

        self.store = WindowStore(
            arrays={"eeg": eeg}, static={},
            audio=[np.ascontiguousarray(A[:, k, :, None]) for k in range(self.K)],
            att_idx=att, trial_ids=tid, subject=sub, content=cont, position=pos,
            window_sec=self.window_sec, hop_sec=self.hop_sec)
        self.bank = self._make_bank()
        log.info("external/%s: %d windows | %d listeners | %d contents | %s K=%d",
                 self.corpus, self.store.n, len(np.unique(sub)),
                 len(np.unique(cont)), self.cand_mode, self.K)
        return self

    def _build(self):
        W, H = int(self.window_sec * FS), int(self.hop_sec * FS)
        rng = np.random.default_rng(self.seed)
        E, A, att, tid, sub, cont = [], [], [], [], [], []
        t_counter = 0
        for t in LOADERS[self.corpus](with_eeg=True):
            if t.eeg is None:
                continue
            n = min(t.eeg.shape[1], len(t.att), len(t.unatt))
            if n < W:
                continue
            for o in range(0, n - W + 1, H):
                e = t.eeg[:, o:o + W]
                a, u = t.att[o:o + W], t.unatt[o:o + W]
                if e.std() < 1e-8 or a.std() < 1e-6 or u.std() < 1e-6:
                    continue
                k = int(rng.integers(0, 2))        # which slot holds the target
                E.append(_z(e).T.astype(np.float32))        # (T,C)
                A.append(np.stack((a, u) if k == 0 else (u, a)))
                att.append(k)
                tid.append(t_counter)
                # the stack treats subject ids as integers; these corpora name
                # listeners "S02", so carry the number
                sub.append(int("".join(c for c in t.subject if c.isdigit())))
                cont.append(t.content)
            t_counter += 1
        if not E:
            raise RuntimeError(f"no windows built for {self.corpus}")
        eeg = np.stack(E)
        A = _z(np.stack(A).astype(np.float32))
        return (eeg, A, np.asarray(att, np.int64), np.asarray(tid, np.int64),
                np.asarray(sub, np.int64), np.asarray(cont))


    def splits(self, protocol="within"):
        """``within`` is inherited; ``loso`` is redefined for these corpora.
        MAESTRO plays the same hundred stimuli to every listener, so a global
        content holdout intersects every listener and the inherited split works.
        Here content is largely listener-specific -- in NJU only 28% of stimulus
        pairs are heard by more than one listener -- so intersecting a global
        holdout with a single held-out listener usually yields an *empty* test
        set.  Instead we hold out the whole listener and remove from training
        every content that listener heard, which is disjoint in both senses and
        cannot empty a fold.
        """
        if protocol != "loso":
            yield from super().splits(protocol)
            return
        st = self.store
        # KU Leuven and DTU play every stimulus to every listener, exactly as
        # MAESTRO does, so the inherited split (global content holdout
        # intersected with the held-out listener) is the right one there.
        # Only a corpus whose content is largely listener-specific needs the
        # override below; removing everything the held-out listener heard
        # would otherwise empty the training set of a shared-content corpus.
        per = [len(np.unique(st.subject[st.content == c]))
               for c in np.unique(st.content)]
        if np.median(per) > 1:
            yield from super().splits(protocol)
            return
        rng = np.random.default_rng(self.seed)
        subs = np.unique(st.subject)
        for s in subs:
            te = np.where(st.subject == s)[0]
            te_content = set(np.unique(st.content[te]))
            pool = np.where((st.subject != s) &
                            ~np.isin(st.content, list(te_content)))[0]
            if len(te) == 0 or len(pool) < 2:
                log.warning("loso fold s%s is empty; skipping", s)
                continue
            other = np.array([u for u in subs if u != s])
            n_val = max(1, int(round(self.val_frac * len(other))))
            val_s = set(rng.choice(other, n_val, replace=False))
            vl = pool[np.isin(st.subject[pool], list(val_s))]
            tr = pool[~np.isin(st.subject[pool], list(val_s))]
            if len(vl) == 0 or len(tr) == 0:
                log.warning("loso fold s%s has an empty partition; skipping", s)
                continue
            yield f"loso_s{s}", self._view(tr, True), self._view(vl), self._view(te)


def _z(x):
    return ((x - x.mean(-1, keepdims=True)) /
            (x.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


__all__ = ["ExternalDataModule", "CORPUS_CH"]
