"""Fixed AAD model (v2) — un-collapses the brain encoder and closes the
audio-only escape hatch in the scoring head.

Four mechanical causes of the collapse we measured in the upstream model
(`model_classification.py`), and what is changed here:

1. RECEPTIVE FIELD >> WINDOW.  Upstream: layers=7, kernel=3, dilation=3**i
   -> RF = 1 + 2*(1+3+9+27+81+243+729) = 2187 samples = 34.2 s @ 64 Hz,
   against windows of 320-1920 samples (5-30 s).  The deep layers convolve
   mostly zero-padding, so their output is dominated by a deterministic,
   input-independent padding transient.  Here: dilation=2**i over 5 layers
   -> RF = 63 samples ~= 1 s, which is the timescale of the speech TRF
   (0-400 ms) rather than 7x the window.

2. NO NORMALISATION + FINAL ReLU.  Upstream applies ReLU after every conv
   including the last and has no BatchNorm/LayerNorm anywhere, so every
   embedding lives in the positive orthant and any two of them have cosine
   ~ 1 by construction (we measured 0.9995-1.0000).  Here: GroupNorm after
   every conv, and no activation on the final layer.

3. THE SCORING HEAD CONTAINS A LINEAR AUDIO-ONLY CLASSIFIER.  Upstream
   scores with  sim_k = mean_t[ normalize(b) * normalize(a_k) ]  then
   Linear(sim_k).  With a brain embedding that is constant in time (unit
   vector c), this reduces exactly to  logit_k = (w * c) . mean_t(a_hat_k),
   i.e. a linear classifier on the time-averaged audio embedding — with
   precisely the capacity needed to read the loudness/shape fingerprint.
   Here: `CouplingHead` time-centres both embeddings before correlating, so
   a time-constant brain gives 0 for every candidate and the model is pinned
   at chance.  The degenerate solution is removed structurally.

4. CAUSAL EEG BRANCH.  The neural response to audio at time t appears in EEG
   at t+100..300 ms, i.e. in the FUTURE relative to t.  A strictly causal EEG
   encoder cannot see it.  Here the brain encoder is centred (non-causal) by
   default, so a window at t sees +-0.5 s of context.

Also adds the two heads the modalities actually deserve: EEG keeps the
stimulus-coupling head, while gaze / IMU / head-IMU / scene-video get a
`SpatialHead` that takes NO audio input at all and therefore cannot use the
acoustic shortcut even in principle.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SPEAKERS = 4


# ── gradient reversal (for the audio-only adversary) ───────────────────────────

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


# ── encoders ───────────────────────────────────────────────────────────────────

class DilatedEncoderLegacy(nn.Module):
    """Byte-equivalent re-implementation of upstream `DilatedEncoder`.

    Kept so the ablation ladder can start from an exact reproduction of the
    published model (config A0) under our own training loop.
    """

    def __init__(self, in_channels, spatial_filters=8, dilation_filters=16,
                 layers=7, kernel_size=3, spatial=False):
        super().__init__()
        self.spatial = spatial
        if spatial:
            self.spatial_conv = nn.Conv1d(in_channels, spatial_filters, 1)
            first_in = spatial_filters
        else:
            first_in = in_channels
        self.dil_convs, self.acts = nn.ModuleList(), nn.ModuleList()
        ch_in = first_in
        for i in range(layers):
            d = kernel_size ** i
            self.dil_convs.append(nn.Conv1d(ch_in, dilation_filters, kernel_size,
                                            dilation=d, padding=d * (kernel_size - 1)))
            self.acts.append(nn.ReLU())
            ch_in = dilation_filters
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


class DilatedEncoderV2(nn.Module):
    """Dilated conv encoder with an RF matched to the TRF timescale, GroupNorm
    after every conv, no final activation, and an optionally centred (non-causal)
    receptive field.  See module docstring, points 1/2/4."""

    def __init__(self, in_channels, spatial_filters=8, dilation_filters=16,
                 layers=5, kernel_size=3, spatial=False, causal=False,
                 dropout=0.1, direction=None):
        """direction: "centred" (default) sees +-RF/2 around t; "past" sees only
        [t-RF, t]; "future" sees only [t, t+RF].  Restricting the direction, in
        combination with an integer lag on the EEG, confines the model to a
        chosen lag band -- which is how a genuine neural response (audio leads
        EEG by 100-300 ms) is distinguished from an instantaneous stimulus
        artifact bleeding into the recording."""
        super().__init__()
        direction = direction or ("past" if causal else "centred")
        self.spatial, self.direction, self.n_layers = spatial, direction, layers
        self.causal = direction == "past"
        if spatial:
            self.spatial_conv = nn.Conv1d(in_channels, spatial_filters, 1)
            first_in = spatial_filters
        else:
            first_in = in_channels

        self.convs, self.norms, self.trims = nn.ModuleList(), nn.ModuleList(), []
        ch = first_in
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

    def forward(self, x):                              # (B,T,C) -> (B,T,D)
        x = x.transpose(1, 2)
        if self.spatial:
            x = self.spatial_conv(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x)
            if self.trims[i] > 0:
                x = (x[:, :, :-self.trims[i]] if self.direction == "past"
                     else x[:, :, self.trims[i]:])
            x = norm(x)
            if i < self.n_layers - 1:                  # NO activation on the last layer
                x = self.drop(F.relu(x))
        return x.transpose(1, 2)


# ── scoring heads ──────────────────────────────────────────────────────────────

class LegacyCosineHead(nn.Module):
    """Upstream head: cosine of un-centred embeddings, then Linear -> scalar.
    Admits the audio-only solution described in point 3."""

    def __init__(self, D):
        super().__init__()
        self.sim_proj = nn.Linear(D, 1)

    def forward(self, brain, aud_encs):
        return torch.cat([
            self.sim_proj((F.normalize(brain, dim=2) * F.normalize(a, dim=2)).mean(1))
            for a in aud_encs], dim=1)


class CouplingHead(nn.Module):
    """Time-centred per-dimension correlation score.

        score(b, a) = tau * w . corr_t(b, a)      (w bias-free)

    Structural guarantee: if the brain embedding is constant over time its
    centred version is exactly 0, so every candidate scores 0, the logits tie
    and the model sits at chance.  There is no constant-brain solution to find.
    Candidate-centring (subtracting the mean over candidates) additionally
    removes any additive term shared by all candidates.
    """

    def __init__(self, D, learn_w=True, init_tau=0.07):
        super().__init__()
        self.w = nn.Linear(D, 1, bias=False) if learn_w else None
        self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))

    @staticmethod
    def corr(b, a, eps=1e-6):
        """Per-dimension Pearson correlation over time. b,a: (B,T,D) -> (B,D)."""
        b = b - b.mean(1, keepdim=True)
        a = a - a.mean(1, keepdim=True)
        b = b / (b.norm(dim=1, keepdim=True) + eps)
        a = a / (a.norm(dim=1, keepdim=True) + eps)
        return (b * a).sum(1)

    def score_from_corr(self, c):                      # c: (..., D) -> (...)
        return (self.w(c).squeeze(-1) if self.w is not None else c.mean(-1)) \
            * self.log_tau.exp()

    def forward(self, brain, aud_encs):
        s = torch.stack([self.score_from_corr(self.corr(brain, a))
                         for a in aud_encs], dim=1)    # (B,K)
        return s - s.mean(1, keepdim=True)             # candidate-centring


class SpatialHead(nn.Module):
    """Predicts the attended SPEAKER INDEX (a fixed loudspeaker azimuth)
    straight from a modality embedding.  Takes no audio input, so it is
    structurally incapable of using the acoustic shortcut — whatever it scores
    is genuinely orienting/lateralisation information."""

    def __init__(self, D, n_spk=N_SPEAKERS, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_spk))

    def forward(self, emb):                            # (B,T,D) -> (B,n_spk)
        return self.net(torch.cat([emb.mean(1), emb.std(1)], dim=-1))


class AudioOnlyAdversary(nn.Module):
    """Tries to name the attended candidate from the AUDIO embeddings alone.
    Its own parameters are trained normally; the audio encoder receives the
    reversed gradient, so any shortcut this adversary can read is actively
    unlearned by the encoder."""

    def __init__(self, D, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, aud_encs, lam=1.0):
        f = torch.stack([torch.cat([a.mean(1), a.std(1)], -1) for a in aud_encs], 1)
        return self.net(grad_reverse(f, lam)).squeeze(-1)          # (B,K)


# ── model ──────────────────────────────────────────────────────────────────────

MODALITY_CH = {"eeg": 32, "video": 4, "gaze": 6, "imu": 6}


class AADModelV2(nn.Module):
    """Configurable AAD model covering the whole ablation ladder.

    encoder : "v2" | "legacy"
    head    : "corr" | "legacy" | "none"   (stimulus-coupling head; EEG only)
    spatial : list of modalities that get a SpatialHead (no audio access)
    """

    def __init__(self, modalities=("eeg",), encoder="v2", head="corr",
                 spatial=(), D=16, D_common=16, spatial_filters=8, layers=5,
                 causal_brain=False, dropout=0.1, adversary=False,
                 modality_dropout=0.0, lag_samples=0, brain_dir=None):
        super().__init__()
        self.lag_samples = lag_samples
        self.brain_dir = brain_dir
        self.modalities = tuple(modalities)
        self.spatial_mods = tuple(spatial)
        self.head_kind = head
        self.modality_dropout = modality_dropout
        self.D_common = D_common

        def _mk(mod):
            ch = MODALITY_CH[mod]
            if encoder == "legacy":
                return DilatedEncoderLegacy(
                    ch, spatial_filters=spatial_filters, dilation_filters=D,
                    layers=7 if mod == "eeg" else {"gaze": 6, "imu": 6, "video": 4}[mod],
                    spatial=(mod == "eeg"))
            return DilatedEncoderV2(
                ch, spatial_filters=spatial_filters, dilation_filters=D,
                layers=layers, spatial=(mod == "eeg"), causal=causal_brain,
                dropout=dropout, direction=brain_dir)

        self.encoders = nn.ModuleDict({m: _mk(m) for m in self.modalities})
        self.projs = nn.ModuleDict({m: nn.Linear(D, D_common) for m in self.modalities})

        # stimulus-coupling branch (EEG only — it is the only modality with a
        # temporal coupling to the speech envelope; forcing gaze/IMU/video
        # through an envelope-matching head is what made them collapse too)
        self.couple_mod = "eeg" if ("eeg" in self.modalities and head != "none") else None
        if self.couple_mod is not None:
            if encoder == "legacy":
                self.audio_encoder = DilatedEncoderLegacy(
                    1, dilation_filters=D_common, layers=7, spatial=False)
            else:
                self.audio_encoder = DilatedEncoderV2(
                    1, dilation_filters=D_common, layers=layers, spatial=False,
                    causal=causal_brain, dropout=dropout)
            self.head = (CouplingHead(D_common) if head == "corr"
                         else LegacyCosineHead(D_common))
            self.adversary = AudioOnlyAdversary(D_common) if adversary else None
        else:
            self.adversary = None

        # spatial branches (no audio input)
        self.spatial_heads = nn.ModuleDict(
            {m: SpatialHead(D_common) for m in self.spatial_mods})
        if len(self.spatial_mods) > 1:
            self.spatial_fuse = SpatialHead(D_common * len(self.spatial_mods))

    # ── pieces ────────────────────────────────────────────────────────────────
    def encode(self, batch):
        """batch: dict of modality -> (B,T,C). Returns dict modality -> (B,T,D_common)."""
        out = {}
        for m in self.modalities:
            x = batch.get(m)
            if x is None:
                continue
            if self.lag_samples and m == self.couple_mod:
                # x'[t] = x[t + lag]: positive lag reads the brain AFTER the
                # stimulus sample it is matched against (the neural direction);
                # negative lag reads it BEFORE, which no neural response can
                # explain and which therefore isolates shared artifacts.
                L = self.lag_samples
                x = (F.pad(x, (0, 0, 0, L))[:, L:] if L > 0
                     else F.pad(x, (0, 0, -L, 0))[:, :L])
            out[m] = self.projs[m](self.encoders[m](x))
        return out

    def encode_audio(self, audio):
        return [self.audio_encoder(a) for a in audio]

    def forward(self, batch, audio=None, perm=None, brain_override=None,
                adv_lam=1.0):
        """
        batch : dict modality -> (B,T,C)
        audio : list of K tensors (B,T,1)   (None for spatial-only models)
        perm  : (B,K) long — perm[b,k] is the SPEAKER INDEX sitting in slot k
        Returns dict with logits (B,K) over slots, plus intermediates.
        """
        embs = self.encode(batch)
        if self.training and self.modality_dropout > 0 and len(embs) > 1:
            keep = [m for m in embs if torch.rand(()) > self.modality_dropout]
            if not keep:                       # never drop everything
                keep = [self.couple_mod or self.modalities[0]]
            embs = {m: embs[m] for m in keep}

        out = {"embs": embs, "logits": None, "couple_logits": None,
               "spk_logits": {}, "aud_encs": None, "brain": None}

        # coupling branch (skipped if modality dropout removed the coupling modality)
        brain = embs.get(self.couple_mod) if self.couple_mod else None
        if brain_override is not None:
            brain = brain_override
        if brain is not None and audio is not None:
            aud_encs = self.encode_audio(audio)
            out["brain"], out["aud_encs"] = brain, aud_encs
            out["couple_logits"] = self.head(brain, aud_encs)
            if self.adversary is not None:
                out["adv_logits"] = self.adversary(aud_encs, adv_lam)

        # spatial branches (logits over SPEAKER INDEX)
        for m in self.spatial_mods:
            if m in embs:
                out["spk_logits"][m] = self.spatial_heads[m](embs[m])
        if len(self.spatial_mods) > 1 and embs:
            # dropped modalities are zero-filled so the fusion head always runs
            ref = next(iter(embs.values()))
            cat = torch.cat([embs[m] if m in embs else torch.zeros_like(ref)
                             for m in self.spatial_mods], dim=-1)
            out["spk_logits"]["fused"] = self.spatial_fuse(cat)

        # assemble slot logits
        logits = out["couple_logits"]
        if out["spk_logits"] and perm is not None:
            key = ("fused" if "fused" in out["spk_logits"]
                   else next(iter(out["spk_logits"])))
            slot = torch.gather(out["spk_logits"][key], 1, perm)   # speaker -> slot
            logits = slot if logits is None else logits + slot
        out["logits"] = logits
        return out
