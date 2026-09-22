"""MERIT network: encoders, scoring heads and the assembled decoder.

Three structural properties, each one removing a failure mode rather than
discouraging it:

1. **The coupling head cannot be won without the recording.**  Its score is a
   *time-centred* correlation, so a physiological embedding that is constant
   over time is exactly zero after centring, correlates with nothing, ties every
   candidate and is pinned at 1/K.  The degenerate optimum is arithmetically
   unreachable, not merely penalised.
2. **The orientation heads cannot see audio.**  They classify the loudspeaker
   index from one modality's embedding alone, so no acoustic property can reach
   them and their permutation null is at chance by construction.
3. **Every stream is scored separately and re-scorable.**  ``embed`` and
   ``score`` are split so a trained model can be re-evaluated under an arbitrary
   per-modality permutation without re-encoding anything -- which is what makes
   per-modality contribution and the Shapley decomposition cheap enough to run
   as a standard metric rather than an occasional audit.

The encoder, correlation head and audio-only adversary follow the corrected
model of the leakage audit (``analysis/n_gh_checks/fixed/model_v2.py``); the
static V-JEPA 2 stream, the per-modality gating and the re-scoring interface are
new here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.merit_data import (MODALITY_CH, N_SPEAKERS, STATIC_DIM,
                                 STATIC_DIMS, TIME_MODS)


# ---------------------------------------------------------------------------
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return _GRL.apply(x, lam)


# ---------------------------------------------------------------------------
class DilatedEncoder(nn.Module):
    """Dilated temporal convolutions with a receptive field matched to the
    cortical response timescale.

    Doubling the dilation each layer gives ``RF = 1 + (k-1)*sum_i d_i`` = 63
    samples from five layers -- 0.98 s at 64 Hz, the span over which the response
    to a speech envelope unfolds.  Matching it is a constraint, not a default: an
    RF far larger than the window makes the deepest layers convolve mostly
    padding, which is identical in every window, so their output stops depending
    on the input at all.

    ``direction`` selects the lag band the encoder can see.  ``centred`` looks
    +-RF/2 around ``t`` (the default: the response to audio at ``t`` occurs at
    ``t+100..300 ms``, which a past-only encoder cannot reach).  ``past`` and
    ``future`` restrict it to one side, which -- combined with an integer lag on
    the EEG -- confines the model to a chosen lag band and separates a causally
    directed neural response from a time-symmetric artifact.

    GroupNorm follows every convolution; there is deliberately **no activation on
    the last layer**, because a rectified output has no negative entries, so any
    two embeddings would have a cosine confined to [0,1] and the correlation the
    head measures would be compressed into a narrow band near 1.
    """

    def __init__(self, in_channels, spatial_filters=8, dilation_filters=16,
                 layers=5, kernel_size=3, spatial=False, dropout=0.1,
                 direction="centred"):
        super().__init__()
        self.spatial, self.direction, self.n_layers = spatial, direction, layers
        if spatial:
            self.spatial_conv = nn.Conv1d(in_channels, spatial_filters, 1)
            ch = spatial_filters
        else:
            ch = in_channels
        self.convs, self.norms, self.trims = nn.ModuleList(), nn.ModuleList(), []
        for i in range(layers):
            d = 2 ** i
            full = d * (kernel_size - 1)
            pad = full if direction in ("past", "future") else full // 2
            self.convs.append(nn.Conv1d(ch, dilation_filters, kernel_size,
                                        dilation=d, padding=pad))
            self.norms.append(nn.GroupNorm(min(4, dilation_filters), dilation_filters))
            self.trims.append(pad if direction in ("past", "future") else 0)
            ch = dilation_filters
        self.drop = nn.Dropout(dropout)
        self.out_channels = dilation_filters
        self.receptive_field = 1 + sum(2 ** i * (kernel_size - 1) for i in range(layers))

    def forward(self, x):                                   # (B,T,C) -> (B,T,D)
        x = x.transpose(1, 2)
        if self.spatial:
            x = self.spatial_conv(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x)
            if self.trims[i] > 0:
                x = (x[:, :, :-self.trims[i]] if self.direction == "past"
                     else x[:, :, self.trims[i]:])
            x = norm(x)
            if i < self.n_layers - 1:
                x = self.drop(F.relu(x))
        return x.transpose(1, 2)


class LegacyEncoder(nn.Module):
    """The encoder of the published benchmark, reproduced so the ablation ladder
    can start from an exact copy of it under our own training loop.

    Dilations *triple* over seven layers, so the receptive field is 2187 samples
    -- 34 s against a 10 s window.  The deepest layers therefore convolve mostly
    zero padding, and because that padding is identical in every window their
    output stops depending on the input at all.  A rectifier follows every layer
    including the last, so embeddings have no negative entries and their cosine
    is confined to [0,1].  There is no normalisation.
    """

    def __init__(self, in_channels, spatial_filters=8, dilation_filters=16,
                 layers=7, kernel_size=3, spatial=False):
        super().__init__()
        self.spatial = spatial
        if spatial:
            self.spatial_conv = nn.Conv1d(in_channels, spatial_filters, 1)
            ch = spatial_filters
        else:
            ch = in_channels
        self.dil_convs, self.acts = nn.ModuleList(), nn.ModuleList()
        for i in range(layers):
            d = kernel_size ** i
            self.dil_convs.append(nn.Conv1d(ch, dilation_filters, kernel_size,
                                            dilation=d, padding=d * (kernel_size - 1)))
            self.acts.append(nn.ReLU())
            ch = dilation_filters
        self.out_channels = dilation_filters
        self.receptive_field = 1 + sum((kernel_size - 1) * kernel_size ** i
                                       for i in range(layers))

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.spatial:
            x = self.spatial_conv(x)
        for conv, act in zip(self.dil_convs, self.acts):
            x = conv(x)
            if conv.padding[0] > 0:
                x = x[:, :, :-conv.padding[0]]
            x = act(x)
        return x.transpose(1, 2)


class StaticEncoder(nn.Module):
    """Projection for a frozen per-window embedding (V-JEPA 2 fovea/scene).

    No gradient ever reaches pixels: sixteen listeners cannot fit a video model,
    and one trained here would memorise the room.  The encoder is a standardising
    linear map onto the shared descriptor width.
    """

    def __init__(self, in_dim=STATIC_DIM, dim=16, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden),
                                 nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2 * dim))

    def forward(self, v):                                   # (B,in_dim) -> (B,2D)
        return self.net(v)


# ---------------------------------------------------------------------------
class CouplingHead(nn.Module):
    """``score(b, a) = tau * w . corr_t(b, a)``, then centred across candidates.

    Centring ``b`` over time is what forbids the degenerate solution: a
    time-constant embedding becomes exactly zero, every candidate scores zero and
    the logits tie at chance.  Correlation is scale-free, so a candidate cannot
    be favoured for being louder; ``w`` carries no bias because a bias shifts
    every candidate equally and can never change the arg-max.
    """

    def __init__(self, D, init_tau=0.07):
        super().__init__()
        self.w = nn.Linear(D, 1, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))

    @staticmethod
    def corr(b, a, eps=1e-6):
        b = b - b.mean(1, keepdim=True)
        a = a - a.mean(1, keepdim=True)
        b = b / (b.norm(dim=1, keepdim=True) + eps)
        a = a / (a.norm(dim=1, keepdim=True) + eps)
        return (b * a).sum(1)                               # (B,D)

    def score_from_corr(self, c):
        return self.w(c).squeeze(-1) * self.log_tau.exp()

    def forward(self, brain, aud_encs):
        s = torch.stack([self.score_from_corr(self.corr(brain, a))
                         for a in aud_encs], dim=1)         # (B,K)
        return s - s.mean(1, keepdim=True)


class LegacyCosineHead(nn.Module):
    """The published scoring function: cosine of *un-centred* embeddings,
    averaged over time, then a linear read-out.

    Let the physiological embedding be the same unit vector at every time step.
    The time average factors and the score becomes ``w . (c * mean_t a_k) + b``
    -- a linear classifier on the time-averaged audio embedding, with exactly the
    capacity needed to read the shape signature of the candidate set.  The
    architecture therefore *contains* an audio-only decoder as a special case,
    reachable by emitting a constant embedding, and that optimum is easier to
    find than the intended one.
    """

    def __init__(self, D):
        super().__init__()
        self.sim_proj = nn.Linear(D, 1)

    @staticmethod
    def corr(b, a, eps=1e-6):                          # interface parity only
        return (F.normalize(b, dim=2) * F.normalize(a, dim=2)).mean(1)

    def score_from_corr(self, c):
        return self.sim_proj(c).squeeze(-1)

    def forward(self, brain, aud_encs):
        return torch.cat([self.sim_proj(
            (F.normalize(brain, dim=2) * F.normalize(a, dim=2)).mean(1))
            for a in aud_encs], dim=1)


class OrientationHead(nn.Module):
    """Loudspeaker index from one modality's pooled descriptor.  No audio in."""

    def __init__(self, in_dim, n_spk=N_SPEAKERS, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_spk))

    def forward(self, z):
        return self.net(z)


class AudioOnlyAdversary(nn.Module):
    """Names the attended candidate from the audio embeddings alone.

    Its own parameters train normally; the audio encoder receives the reversed
    gradient, so whatever acoustic shortcut this head can read is actively
    unlearned from the audio embedding rather than left for the decoder to find.
    """

    def __init__(self, D, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, aud_encs, lam=1.0):
        f = torch.stack([torch.cat([a.mean(1), a.std(1)], -1) for a in aud_encs], 1)
        return self.net(grad_reverse(f, lam)).squeeze(-1)   # (B,K)


# ---------------------------------------------------------------------------
class MeritNet(nn.Module):
    """The decoder.

    ``embed(batch)`` returns one pooled descriptor per stream (plus the EEG time
    embedding the coupling head needs); ``score(...)`` turns descriptors into
    slot logits.  Splitting them is what lets the evaluator permute any subset of
    streams and re-score without re-encoding, and lets the objective include a
    per-stream permutation term at negligible cost.
    """

    def __init__(self, time_mods=("eeg",), static_mods=(), D=16, spatial_filters=8,
                 layers=5, dropout=0.1, coupling=True, orientation=("eeg",),
                 adversary=True, modality_dropout=0.3, lag_samples=0,
                 brain_dir="centred", fuse_hidden=32, gates=True,
                 encoder="v2", head="corr", n_subjects=0, subject_rank=2,
                 audio_channels=1, eeg_channels=None):
        super().__init__()
        self.time_mods = tuple(time_mods)
        self.static_mods = tuple(static_mods)
        self.streams = self.time_mods + self.static_mods
        self.orientation_mods = tuple(m for m in orientation if m in self.streams)
        self.modality_dropout = modality_dropout
        self.lag_samples, self.brain_dir, self.D = lag_samples, brain_dir, D

        # Corpora differ in montage: 32 channels here, 64 on KU Leuven and DTU.
        ch = dict(MODALITY_CH)
        if eeg_channels:
            ch["eeg"] = int(eeg_channels)

        def _mk(m):
            if encoder == "legacy":
                return LegacyEncoder(ch[m], spatial_filters=spatial_filters,
                                     dilation_filters=D,
                                     layers=7 if m == "eeg" else
                                     {"gaze": 6, "imu": 6, "video": 4}[m],
                                     spatial=(m == "eeg"))
            return DilatedEncoder(ch[m], spatial_filters=spatial_filters,
                                  dilation_filters=D, layers=layers,
                                  spatial=(m == "eeg"), dropout=dropout,
                                  direction=(brain_dir if m == "eeg" else "centred"))

        self.encoder_kind, self.head_kind = encoder, head
        self.encoders = nn.ModuleDict({m: _mk(m) for m in self.time_mods})
        # Listener-conditioned spatial filtering. One shared 32->8 electrode mix
        # for sixteen differently shaped heads is the weakest part of the EEG
        # encoder; a rank-r per-listener correction is 16*(32+8)*r parameters and
        # falls back to the population mean for an unseen listener under LOSO.
        self.n_subjects, self.subject_rank = int(n_subjects), int(subject_rank)
        C = ch["eeg"]
        if self.n_subjects and "eeg" in self.time_mods:
            # x <- x + sum_r <x, a_r> b_r : a rank-r channel-mixing correction.
            # b is zero-initialised, so training starts from the shared filter and
            # index 0 is reserved for the population mean used on an unseen listener.
            self.subj_a = nn.Embedding(self.n_subjects + 1, C * subject_rank)
            self.subj_b = nn.Embedding(self.n_subjects + 1, C * subject_rank)
            nn.init.normal_(self.subj_a.weight, std=1.0 / C ** 0.5)
            nn.init.zeros_(self.subj_b.weight)
        self.projs = nn.ModuleDict({m: nn.Linear(D, D) for m in self.time_mods})
        self.static_encoders = nn.ModuleDict(
            {m: StaticEncoder(STATIC_DIMS.get(m, STATIC_DIM), D,
                              dropout=max(dropout, 0.2))
             for m in self.static_mods})

        self.couple_mod = "eeg" if (coupling and "eeg" in self.time_mods) else None
        if self.couple_mod is not None:
            self.audio_encoder = (
                LegacyEncoder(audio_channels, dilation_filters=D, layers=7, spatial=False)
                if encoder == "legacy" else
                DilatedEncoder(audio_channels, dilation_filters=D, layers=layers,
                               spatial=False, dropout=dropout, direction=brain_dir))
            self.head = CouplingHead(D) if head == "corr" else LegacyCosineHead(D)
            self.adversary = AudioOnlyAdversary(D) if adversary else None
        else:
            self.head, self.adversary = None, None

        self.orient_heads = nn.ModuleDict(
            {m: OrientationHead(2 * D) for m in self.orientation_mods})
        # a learned non-negative gate per stream keeps the additive part of the
        # decision readable; the fusion head then models what gating cannot.
        self.gates = (nn.ParameterDict({m: nn.Parameter(torch.zeros(()))
                                        for m in self.orientation_mods})
                      if gates else None)
        self.fuse = (OrientationHead(2 * D * len(self.orientation_mods),
                                     hidden=fuse_hidden)
                     if len(self.orientation_mods) > 1 else None)

    # -- pieces ------------------------------------------------------------
    def _lagged(self, x):
        """``x'[t] = x[t+L]``.  A positive lag reads the recording *after* the
        stimulus sample it is scored against -- the only direction a neural
        response can occupy; a negative lag reads it before, which isolates
        instantaneous artifacts."""
        L = self.lag_samples
        if not L:
            return x
        return (F.pad(x, (0, 0, 0, L))[:, L:] if L > 0
                else F.pad(x, (0, 0, -L, 0))[:, :L])

    def embed(self, batch, subj=None):
        """-> (descriptors {m: (B,2D)}, brain time-embedding (B,T,D) or None)."""
        z, brain = {}, None
        for m in self.time_mods:
            x = batch.get(m)
            if x is None:
                continue
            if m == "eeg" and self.n_subjects and subj is not None:
                sid = subj.clamp(0, self.n_subjects)
                a = self.subj_a(sid).view(-1, self.subject_rank, x.shape[-1])
                b = self.subj_b(sid).view(-1, self.subject_rank, x.shape[-1])
                x = x + torch.einsum("btr,brc->btc",
                                     torch.einsum("btc,brc->btr", x, a), b)
            e = self.projs[m](self.encoders[m](self._lagged(x) if m == self.couple_mod else x))
            z[m] = torch.cat([e.mean(1), e.std(1)], dim=-1)
            if m == self.couple_mod:
                brain = e
        for m in self.static_mods:
            v = batch.get(m)
            if v is not None:
                z[m] = self.static_encoders[m](v)
        return z, brain

    def encode_audio(self, audio):
        return [self.audio_encoder(a) for a in audio]

    def orientation_logits(self, z):
        """-> ({stream: (B,4)}, fused (B,4) or None). Missing streams are
        zero-filled so the fusion head always executes."""
        per = {m: self.orient_heads[m](z[m]) for m in self.orientation_mods if m in z}
        fused = None
        if self.fuse is not None and per:
            ref = next(iter(z.values()))
            cat = torch.cat([z[m] if m in z else torch.zeros_like(ref)
                             for m in self.orientation_mods], dim=-1)
            fused = self.fuse(cat)
        return per, fused

    def score(self, z, brain, aud_encs, perm):
        """Slot logits from descriptors.  ``perm[b,k]`` is the loudspeaker index
        sitting in slot ``k``."""
        logits = None
        if brain is not None and aud_encs is not None:
            logits = self.head(brain, aud_encs)
        per, fused = self.orientation_logits(z)
        spk = None
        if fused is not None:
            spk = fused
        if self.gates is not None and per:
            gated = sum(F.softplus(self.gates[m]) * per[m] for m in per)
            spk = gated if spk is None else spk + gated
        elif spk is None and per:
            spk = sum(per.values()) / len(per)
        if spk is not None and perm is not None:
            slot = torch.gather(spk, 1, perm)
            logits = slot if logits is None else logits + slot
        return logits, per, fused

    def forward(self, batch, audio=None, perm=None, adv_lam=1.0, subj=None):
        z, brain = self.embed(batch, subj)
        if self.training and self.modality_dropout > 0 and len(z) > 1:
            keep = [m for m in z if torch.rand(()) > self.modality_dropout]
            if not keep:
                keep = [self.couple_mod or self.streams[0]]
            if self.couple_mod is not None and self.couple_mod not in keep:
                brain = None
            z = {m: z[m] for m in keep}
        aud_encs = self.encode_audio(audio) if (audio is not None and
                                                self.couple_mod is not None) else None
        logits, per, fused = self.score(z, brain, aud_encs, perm)
        out = {"logits": logits, "z": z, "brain": brain, "aud_encs": aud_encs,
               "spk_logits": per, "spk_fused": fused}
        if self.adversary is not None and aud_encs is not None:
            out["adv_logits"] = self.adversary(aud_encs, adv_lam)
        return out


__all__ = ["MeritNet", "CouplingHead", "LegacyCosineHead", "OrientationHead",
           "DilatedEncoder", "LegacyEncoder", "StaticEncoder", "AudioOnlyAdversary",
           "grad_reverse"]
