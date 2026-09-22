"""Per-stream contribution and its exact Shapley decomposition.

Accuracy says how often the decoder is right.  It does not say *which input*
made it right, and in this task the difference is the whole question: a decoder
can reach the published number while reading a confound in the candidate audio,
or while riding one strong behavioural stream with its EEG pathway dead.  Both
are invisible in accuracy and survive every standard held-out protocol.

The measurement is a permutation.  Permute one stream across test windows while
every window keeps its own candidates and its own label; a decoder that uses the
stream loses accuracy, a decoder that ignores it does not.  It must be a
*feature* permutation and not a label permutation -- permuting labels would also
destroy the audio-label relationship and would hide the very shortcut being
tested for.

Write ``v(T)`` for the accuracy when the streams in ``T`` are intact and all the
others are permuted.  Then ``v(M)`` is the reported accuracy, ``v(empty)`` is
the null, and the **Shapley value**

    phi_m = sum_{T subset of M\\{m}} |T|!(|M|-|T|-1)!/|M|!  [v(T + m) - v(T)]

splits the total contribution ``v(M) - v(empty)`` into per-stream credit that is
*exactly additive*: ``sum_m phi_m = v(M) - v(empty)``.  With M streams this costs
``2^M`` re-scorings, and because descriptors and audio embeddings are cached once
the re-scoring never re-encodes anything -- at M = 5 the whole decomposition is
cheaper than one training epoch.

Also provided: stratified nulls (position-in-trial, within-trial), the zeros
ablation, the decision-flip rate and the embedding-collapse statistics that
distinguish "uses the input" from "has stopped depending on the input".
"""
from __future__ import annotations

import itertools
import logging
from math import factorial

N_SPEAKERS = 4

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("credit.attribution")


class ContributionEvaluator:
    """Caches stream descriptors and audio embeddings once, then re-scores under
    arbitrary per-stream permutations."""

    def __init__(self, model, loader, device, strata=None, chunk=512,
                 trial_level=False):
        self.model, self.device, self.chunk = model, device, chunk
        self.trial_level = trial_level
        self.strata = ({} if strata is None else
                       {k: np.asarray(v) for k, v in strata.items()})
        model.eval()
        z_acc, brains, auds = {}, [], []
        labels, perms, spks, trials, subjs = [], [], [], [], []
        with torch.no_grad():
            for mods, audio, lab, spk_of_slot, att, subj, trial in loader:
                mods = {m: v.to(device) for m, v in mods.items()}
                z, brain = model.embed(mods, subj.to(device))
                for m, v in z.items():
                    z_acc.setdefault(m, []).append(v.cpu())
                if brain is not None:
                    brains.append(brain.cpu())
                if model.couple_mod is not None:
                    a = torch.stack([model.audio_encoder(x.to(device))
                                     for x in audio], 1)            # (B,K,T,D)
                    auds.append(a.cpu())
                labels.append(lab); perms.append(spk_of_slot); spks.append(att)
                trials.append(trial); subjs.append(subj)
        self.z = {m: torch.cat(v) for m, v in z_acc.items()}
        self.trials = torch.cat(trials).numpy()
        self.subjs = torch.cat(subjs).numpy()
        self.brain = torch.cat(brains) if brains else None
        self.aud = torch.cat(auds) if auds else None
        self.labels = torch.cat(labels)
        self.perms = torch.cat(perms)
        self.spks = torch.cat(spks)
        self.N = len(self.labels)
        self.streams = tuple(self.z.keys())
        self.K = (self.aud.shape[1] if self.aud is not None else self.perms.shape[1])
        self._tinv = None
        self._ttrue = None

    # -- scoring -----------------------------------------------------------
    @torch.no_grad()
    def logits(self, perm_map=None, zero=(), only=None):
        """``perm_map``: {stream: index array} applied to that stream only.
        ``zero``: streams replaced by zeros instead.
        ``only``: score through one stream's own pathway alone -- its orientation
        head, plus the coupling head if it is the coupling stream. This is what a
        single-stream model would produce from these same weights, and the gap
        between it and the stream's contribution to the fused decision is what the
        hinge gate needs (a stream can be capable on its own and still be ignored
        by the fusion)."""
        perm_map = perm_map or {}
        zero = set(zero)
        out = []
        for s in range(0, self.N, self.chunk):
            sl = slice(s, min(s + self.chunk, self.N))
            zb, brain = {}, None
            for m in self.streams:
                idx = (torch.as_tensor(perm_map[m][sl]) if m in perm_map
                       else torch.arange(sl.start, sl.stop))
                v = self.z[m][idx]
                if m in zero:
                    v = torch.zeros_like(v)
                zb[m] = v.to(self.device)
            if self.brain is not None:
                cm = self.model.couple_mod
                idx = (torch.as_tensor(perm_map[cm][sl]) if cm in perm_map
                       else torch.arange(sl.start, sl.stop))
                brain = self.brain[idx].to(self.device)
                if cm in zero:
                    brain = torch.zeros_like(brain)
            aud = ([self.aud[sl, k].to(self.device) for k in range(self.K)]
                   if self.aud is not None else None)
            pm = self.perms[sl].to(self.device)
            if only is None:
                lg, _, _ = self.model.score(zb, brain, aud, pm)
            else:
                lg = None
                m0 = self.model
                if only == m0.couple_mod and brain is not None and aud is not None:
                    lg = m0.head(brain, aud)
                if only in m0.orient_heads:
                    sp = torch.gather(m0.orient_heads[only](zb[only]), 1, pm)
                    lg = sp if lg is None else lg + sp
                if lg is None:
                    lg = torch.zeros(n, self.K, device=self.device)
            out.append(lg.cpu())
        return torch.cat(out)

    def acc(self, lg):
        if not self.trial_level:
            return float((lg.argmax(1) == self.labels).float().mean())
        return self._trial_acc(lg)

    def _trial_acc(self, lg):
        """Map each window's slot scores back to loudspeaker order through its own
        shuffle, sum over the windows of a trial, and decide once per trial.

        This is 'train short, decide long': the decoder is fitted on short windows,
        of which a trial yields many, and the decision is taken on the whole trial.
        It makes the four-way number trial-level by construction, which is why the
        same-talker results, which stay window-level, are reported alongside it.
        """
        spk = torch.zeros(lg.shape[0], N_SPEAKERS)
        spk.scatter_(1, self.perms, lg.float())
        if self._tinv is None:
            u, inv = np.unique(self.trials, return_inverse=True)
            self._tinv = torch.as_tensor(inv)
            first = {int(t): i for i, t in reversed(list(enumerate(self.trials)))}
            self._ttrue = torch.as_tensor([int(self.spks[first[int(t)]]) for t in u])
        agg = torch.zeros(len(self._ttrue), N_SPEAKERS)
        agg.index_add_(0, self._tinv, spk)
        return float((agg.argmax(1) == self._ttrue).float().mean())

    def _shuffle(self, rng, key=None):
        """A global permutation, or one restricted to windows sharing a stratum.

        ``pos`` (same position in trial) rules out a slow drift shared by the
        recording and the envelope.  ``trial`` is stricter: all windows of a trial
        share a listener, an attended talker and a stimulus set, so a decoder that
        merely recognises *which trial* it is viewing survives a global
        permutation but not a within-trial one.
        """
        if key is None or key not in self.strata:
            return rng.permutation(self.N)
        st = self.strata[key]
        p = np.arange(self.N)
        for g in np.unique(st):
            m = np.where(st == g)[0]
            p[m] = m[rng.permutation(len(m))]
        return p

    # -- the battery -------------------------------------------------------
    def battery(self, n_shuffle=20, seed=1000, shapley=True, max_shapley_streams=6,
                shapley_draws=5):
        real_lg = self.logits()
        real = self.acc(real_lg)
        streams = list(self.streams)

        # common random numbers: the same permutation draw is reused for every
        # coalition, so differences between coalitions are not permutation noise.
        draws = [np.random.default_rng(seed + k).permutation(self.N)
                 for k in range(n_shuffle)]

        nulls, flips, per_stream = [], [], {m: [] for m in streams}
        coalition = {}
        for k, p in enumerate(draws):
            lg = self.logits({m: p for m in streams})
            nulls.append(self.acc(lg))
            flips.append(float((lg.argmax(1) != real_lg.argmax(1)).float().mean()))
            for m in streams:
                per_stream[m].append(self.acc(self.logits({m: p})))

        if shapley and 1 < len(streams) <= max_shapley_streams:
            # v(T) = accuracy with T intact and M\T permuted, averaged over draws.
            # The interior coalitions use a subset of the draws: 2^M re-scorings
            # per draw, and their differences are what matters, not their noise.
            sub = draws[:max(1, min(shapley_draws, n_shuffle))]
            for r in range(len(streams) + 1):
                for T in itertools.combinations(streams, r):
                    out = [m for m in streams if m not in T]
                    if not out:
                        coalition[T] = real
                    elif len(out) == len(streams):
                        coalition[T] = float(np.mean(nulls))
                    else:
                        coalition[T] = float(np.mean(
                            [self.acc(self.logits({m: p for m in out}))
                             for p in sub]))
            phi = self._shapley(streams, coalition)
        else:
            phi = {}

        # solo capability: what each stream reaches through its own pathway alone
        solo = {}
        for m in streams:
            a = self.acc(self.logits(only=m))
            n_ = float(np.mean([self.acc(self.logits({m: p}, only=m))
                                for p in draws[:max(1, min(5, n_shuffle))]]))
            solo[m] = (a, a - n_)

        nulls = np.array(nulls)
        zeros_acc = {m: self.acc(self.logits(zero=(m,))) for m in streams}
        snull = {}
        for nm in self.strata:
            vals = [self.acc(self.logits(
                {m: self._shuffle(np.random.default_rng(5000 + k), nm)
                 for m in streams})) for k in range(n_shuffle)]
            snull[nm] = float(np.mean(vals))

        res = dict(
            acc=real, null_mean=float(nulls.mean()), null_std=float(nulls.std()),
            contribution=float(real - nulls.mean()),
            p_perm=float((np.sum(nulls >= real) + 1) / (n_shuffle + 1)),
            flip_rate=float(np.mean(flips)),
            zeros_all=self.acc(self.logits(zero=streams)),
            chance=1.0 / self.K, n=self.N, n_streams=len(streams))
        for m in streams:
            res[f"acc_perm_{m}"] = float(np.mean(per_stream[m]))
            res[f"contrib_{m}"] = float(real - np.mean(per_stream[m]))
            res[f"acc_zero_{m}"] = zeros_acc[m]
            res[f"lomo_{m}"] = float(real - zeros_acc[m])
            res[f"solo_acc_{m}"] = float(solo[m][0])
            res[f"solo_contrib_{m}"] = float(solo[m][1])
            # how much of what this stream can do on its own the fusion discards
            res[f"unused_{m}"] = float(max(0.0, solo[m][1] - (real - np.mean(per_stream[m]))))
        for m, v in phi.items():
            res[f"shapley_{m}"] = float(v)
        if phi:
            res["shapley_sum"] = float(sum(phi.values()))
            res["min_shapley"] = float(min(phi.values()))
        for nm, v in snull.items():
            res[f"null_{nm}"] = v
            res[f"contribution_{nm}"] = float(real - v)
        res.update(self._collapse_stats())
        return res

    @staticmethod
    def _shapley(streams, coalition):
        M = len(streams)
        phi = {}
        for m in streams:
            others = [s for s in streams if s != m]
            acc = 0.0
            for r in range(len(others) + 1):
                w = factorial(r) * factorial(M - r - 1) / factorial(M)
                for T in itertools.combinations(others, r):
                    acc += w * (coalition[tuple(sorted(T + (m,), key=streams.index))]
                                - coalition[tuple(sorted(T, key=streams.index))])
            phi[m] = acc
        return phi

    def _collapse_stats(self):
        """Has the encoder stopped depending on its input?  ``emb_cos_centered``
        is the mean pairwise cosine of the *time-centred* embeddings -- the exact
        quantity the coupling score consumes.  A value near 1 means every window
        produces the same temporal pattern."""
        if self.brain is None:
            return {}
        sel = torch.randperm(len(self.brain))[:512]
        E = self.brain[sel]

        def _offdiag_cos(V):
            V = F.normalize(V, dim=-1)
            return float(((V @ V.t()).sum() - len(V)) / (len(V) * (len(V) - 1)))

        Ec = E - E.mean(1, keepdim=True)
        Ec = Ec / (Ec.norm(dim=1, keepdim=True) + 1e-6)
        return {"emb_cos": _offdiag_cos(E.mean(1)),
                "emb_cos_centered": _offdiag_cos(Ec.reshape(len(sel), -1)),
                "emb_temporal_std": float(E.std(1).mean())}


__all__ = ["ContributionEvaluator"]
