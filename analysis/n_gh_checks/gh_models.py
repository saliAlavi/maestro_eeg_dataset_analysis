"""Faithful port of the public ASPIRE-OSU/MAESTRO benchmark model architectures.

These are byte-faithful re-implementations of the github repo's
``model_classification.py`` (4-class AAD), ``model_spatial.py`` (2-stream
hemisphere / eccentricity — identical to the classification model but with two
output streams), ``model_reconstruction.py`` (linear backward envelope
reconstruction) and the ``LateFusionCombiner`` from ``late_fusion.py``.

Nothing about the *architecture* is changed here — the leakage fixes (proper
train/inner-val/test protocol, trial-level leakage-safe splits, loudness-matched
envelopes, 5 s / 0.5-overlap windows) live entirely in the data + training
layers (``gh_data.py`` / ``train_gh.py``). This file must reproduce the github
graph exactly so the only thing that moves the numbers is the honest protocol.

Reference: /tmp clone of github.com/ASPIRE-OSU/MAESTRO scripts/ (model_*.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel counts. EEG/IMU match the github repo exactly (32 / 6). GAZE is 3 here
# (our proper dataloader exposes [gaze2d_x, gaze2d_y, pupil]) vs the github repo's
# 6 (it also carried gaze3d x/y/z). VIDEO is intentionally absent — our dataloader
# does not materialise optical-flow video (windows.py leaves present_video=False),
# so the video modality is out of scope for this reproduction.
N_EEG_CH = 32
N_GAZE_CH = 3
N_IMU_CH = 6
N_VIDEO_CH = 4   # kept for signature compatibility; never instantiated here


class DilatedEncoder(nn.Module):
    """Causal dilated convolutional encoder — verbatim from the github repo.

    (model_classification.py:15-62). x: (B, T, C) -> (B, T, dilation_filters).
    Causality is enforced by right-trimming the symmetric padding.
    """

    def __init__(self, in_channels: int, spatial_filters: int = 8,
                 dilation_filters: int = 16, layers: int = 6,
                 kernel_size: int = 3, spatial: bool = False):
        super().__init__()
        self.spatial = spatial
        if spatial:
            self.spatial_conv = nn.Conv1d(in_channels, spatial_filters, kernel_size=1)
            first_in = spatial_filters
        else:
            first_in = in_channels

        self.dil_convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        ch_in = first_in
        for i in range(layers):
            dilation = kernel_size ** i
            padding = dilation * (kernel_size - 1)
            self.dil_convs.append(
                nn.Conv1d(ch_in, dilation_filters, kernel_size=kernel_size,
                          dilation=dilation, padding=padding))
            self.acts.append(nn.ReLU())
            ch_in = dilation_filters
        self.out_channels = dilation_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                      # (B, C, T)
        if self.spatial:
            x = self.spatial_conv(x)
        for conv, act in zip(self.dil_convs, self.acts):
            x = conv(x)
            if conv.padding[0] > 0:
                x = x[:, :, :-(conv.padding[0])]   # causal right-trim
            x = act(x)
        return x.transpose(1, 2)                   # (B, T, D)


class AADModel(nn.Module):
    """Match-mismatch AAD scorer — github model_classification.py:65-177.

    ``n_speakers`` parameterises the number of audio streams scored:
      * 4  -> the 4-class attended-speaker task (model_classification.py)
      * 2  -> hemisphere / eccentricity (model_spatial.py, which hardcodes
              N_SPEAKERS=2 but is otherwise byte-identical).

    ``modalities`` is the set of active "brain" encoders (subset of
    {eeg, gaze, imu}); the forward signature keeps the github order
    (eeg, video, gaze, imu, audio) for faithfulness — video is always None.
    """

    def __init__(self, modalities, n_speakers: int = 4, D: int = 16,
                 D_common: int = 16, spatial_filters: int = 8):
        super().__init__()
        self.modalities = list(modalities)
        self.n_speakers = n_speakers
        self.D_common = D_common
        use_eeg = "eeg" in self.modalities
        use_gaze = "gaze" in self.modalities
        use_imu = "imu" in self.modalities

        if use_eeg:
            self.eeg_encoder = DilatedEncoder(
                in_channels=N_EEG_CH, spatial_filters=spatial_filters,
                dilation_filters=D, layers=7, spatial=True)
            self.eeg_proj = nn.Linear(D, D_common)
        if use_gaze:
            self.gaze_encoder = DilatedEncoder(
                in_channels=N_GAZE_CH, dilation_filters=D, layers=6, spatial=False)
            self.gaze_proj = nn.Linear(D, D_common)
        if use_imu:
            self.imu_encoder = DilatedEncoder(
                in_channels=N_IMU_CH, dilation_filters=D, layers=6, spatial=False)
            self.imu_proj = nn.Linear(D, D_common)

        n_enc = sum([use_eeg, use_gaze, use_imu])
        if n_enc > 1:
            self.fusion = nn.Sequential(nn.Linear(n_enc * D_common, D_common), nn.ReLU())

        self.audio_encoder = DilatedEncoder(
            in_channels=1, dilation_filters=D_common, layers=7, spatial=False)
        self.sim_proj = nn.Linear(D_common, 1)

    @staticmethod
    def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # per-feature normalised product averaged over time -> (B, D_common)
        return (F.normalize(a, dim=2) * F.normalize(b, dim=2)).mean(dim=1)

    def forward(self, eeg, video, gaze, imu, audio):
        embeddings = []
        if eeg is not None and hasattr(self, "eeg_encoder"):
            embeddings.append(self.eeg_proj(self.eeg_encoder(eeg)))
        if gaze is not None and hasattr(self, "gaze_encoder"):
            embeddings.append(self.gaze_proj(self.gaze_encoder(gaze)))
        if imu is not None and hasattr(self, "imu_encoder"):
            embeddings.append(self.imu_proj(self.imu_encoder(imu)))
        if not embeddings:
            raise RuntimeError(f"no modality embeddings for {self.modalities}")

        if len(embeddings) == 1:
            brain_enc = embeddings[0]
        elif hasattr(self, "fusion"):
            brain_enc = self.fusion(torch.cat(embeddings, dim=2))
        else:
            brain_enc = torch.stack(embeddings, dim=0).mean(dim=0)

        logits = []
        for aud in audio:
            aud_enc = self.audio_encoder(aud)
            sim = self._cosine_sim(brain_enc, aud_enc)
            logits.append(self.sim_proj(sim))
        return F.softmax(torch.cat(logits, dim=1), dim=1)


# --------------------------------------------------------------------------- #
# Reconstruction (model_reconstruction.py)
# --------------------------------------------------------------------------- #
def pearson_r(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = y_true.squeeze(-1)
    y_pred = y_pred.squeeze(-1)
    yt = y_true - y_true.mean(dim=1, keepdim=True)
    yp = y_pred - y_pred.mean(dim=1, keepdim=True)
    num = (yt * yp).sum(dim=1)
    denom = torch.sqrt((yt ** 2).sum(dim=1) * (yp ** 2).sum(dim=1)) + 1e-8
    r = num / denom
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def pearson_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return -pearson_r(y_true, y_pred).mean()


class LinearModel(nn.Module):
    """Linear backward model — single causal Conv1d (model_reconstruction.py:65)."""

    def __init__(self, integration_window: int = 32, n_in_channels: int = N_EEG_CH):
        super().__init__()
        self.padding = integration_window - 1
        self.conv = nn.Conv1d(n_in_channels, 1, kernel_size=integration_window, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)               # (B, C, T)
        x = F.pad(x, (self.padding, 0))     # causal left-pad
        x = self.conv(x)                    # (B, 1, T)
        return x.transpose(1, 2)            # (B, T, 1)


N_CHANNELS = {"eeg": N_EEG_CH, "gaze": N_GAZE_CH, "imu": N_IMU_CH}


def n_channels_for(modalities) -> int:
    return sum(N_CHANNELS[m] for m in modalities)


# --------------------------------------------------------------------------- #
# Late fusion (late_fusion.py:103-120)
# --------------------------------------------------------------------------- #
class LateFusionCombiner(nn.Module):
    """One softmax-normalised scalar weight per frozen single-modality model."""

    def __init__(self, n_modalities: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_modalities))

    def forward(self, probs_list):
        w = F.softmax(self.logits, dim=0)
        stacked = torch.stack(probs_list, dim=0)
        combined = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        return combined / combined.sum(dim=1, keepdim=True)


if __name__ == "__main__":
    # tiny CPU smoke of the graphs
    B, T = 4, 320
    for mods, nspk in [(["eeg"], 4), (["gaze"], 2), (["eeg", "imu"], 2)]:
        m = AADModel(mods, n_speakers=nspk)
        eeg = torch.randn(B, T, N_EEG_CH) if "eeg" in mods else None
        gaze = torch.randn(B, T, N_GAZE_CH) if "gaze" in mods else None
        imu = torch.randn(B, T, N_IMU_CH) if "imu" in mods else None
        audio = [torch.randn(B, T, 1) for _ in range(nspk)]
        p = m(eeg, None, gaze, imu, audio)
        assert p.shape == (B, nspk) and torch.allclose(p.sum(1), torch.ones(B), atol=1e-5)
        print(f"AAD {mods} nspk={nspk}: {tuple(p.shape)} params={sum(x.numel() for x in m.parameters()):,}")
    lm = LinearModel(n_in_channels=n_channels_for(["eeg", "imu"]))
    x = torch.randn(B, T, n_channels_for(["eeg", "imu"]))
    print("Linear out:", tuple(lm(x).shape), "r:", float(pearson_r(torch.randn(B, T, 1), lm(x)).mean()))
    print("smoke ok")
