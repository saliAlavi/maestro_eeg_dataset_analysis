"""EEG preprocessing: MNE Raw construction, filtering, ICA, eye-gaze regression."""
from __future__ import annotations

import numpy as np
import mne

from .config import EEG_CHANNELS, EEG_SFREQ, EEG_MASTOIDS


# ADC clip magnitude observed in the pickles (~2^23 × LSB).
ADC_CLIP = 0.08388


def load_bad_channels_manifest():
    """Return the per-trial bad-channel manifest produced by
    ``scripts/scan_bad_channels.py``, or None if the scan hasn't been run.
    """
    from .config import RESULTS_DIR
    import pandas as pd
    p = RESULTS_DIR / "bad_channels_manifest.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def get_cached_bads(subject: int, trial: int) -> list[str] | None:
    """Return cached bad-channel list for a (subject, trial) or None if no cache."""
    m = load_bad_channels_manifest()
    if m is None:
        return None
    r = m[(m["subject"] == subject) & (m["trial"] == trial)]
    if not len(r):
        return None
    s = str(r.iloc[0]["bad_channels"] or "")
    return [c for c in s.split(";") if c]


def detect_bad_channels(
    raw: mne.io.Raw,
    *,
    flat_std_threshold: float = 1e-9,
    saturation_rate_threshold: float = 0.1,
    variance_z_threshold: float = 6.0,
    min_corr_with_neighbors: float = 0.05,
) -> list[str]:
    """Return channel names judged to be unusable.

    Criteria:
        - std of raw signal below ``flat_std_threshold`` (dead electrode),
        - proportion of samples at ADC clip ≥ ``saturation_rate_threshold``,
        - median-robust z-score of channel variance above ``variance_z_threshold``,
        - mean absolute correlation with other channels below
          ``min_corr_with_neighbors``.
    """
    data = raw.get_data()
    ch_names = list(raw.ch_names)
    stds = data.std(axis=1)
    sat = (np.abs(data) >= ADC_CLIP).mean(axis=1)

    # First pass: flat and saturated channels are unambiguously bad.
    prelim_bad = set()
    for i, ch in enumerate(ch_names):
        if stds[i] < flat_std_threshold:
            prelim_bad.add(ch)
        if sat[i] >= saturation_rate_threshold:
            prelim_bad.add(ch)

    # Robust variance z-score computed on *AC* content (successive-sample
    # differences) so per-channel DC offsets — common in the raw pickle —
    # don't corrupt the statistic, and on channels not already flagged
    # (otherwise a saturated channel with enormous variance dominates MAD).
    remaining_idx = [i for i, ch in enumerate(ch_names) if ch not in prelim_bad]
    if len(remaining_idx) >= 3:
        ac = np.diff(data[remaining_idx], axis=1)
        var_sub = np.var(ac, axis=1)
        med = np.median(var_sub)
        mad = np.median(np.abs(var_sub - med)) or 1e-30
        for j, i in enumerate(remaining_idx):
            z = (var_sub[j] - med) / (1.4826 * mad)
            if np.abs(z) > variance_z_threshold:
                prelim_bad.add(ch_names[i])

    # Neighbour correlation on the subset of not-yet-bad channels.
    good_idx = [i for i, ch in enumerate(ch_names) if ch not in prelim_bad]
    if len(good_idx) >= 3:
        with np.errstate(invalid="ignore"):
            sub = data[good_idx]
            # Avoid NaN rows when std is 0 (shouldn't happen at this stage).
            valid = sub.std(axis=1) > 0
            idx_valid = [good_idx[j] for j in range(len(good_idx)) if valid[j]]
            if len(idx_valid) >= 3:
                corr = np.corrcoef(data[idx_valid])
                mean_abs_corr = (np.abs(corr).sum(0) - 1) / max(1, corr.shape[0] - 1)
                for j, i in enumerate(idx_valid):
                    if mean_abs_corr[j] < min_corr_with_neighbors:
                        prelim_bad.add(ch_names[i])
    return sorted(prelim_bad)


def make_mne_info(ch_names: list[str] | None = None, sfreq: float = EEG_SFREQ) -> mne.Info:
    ch_names = list(ch_names or EEG_CHANNELS)
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(ch_names),
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def eeg_raw_to_mne(
    data: np.ndarray,
    *,
    ch_names: list[str] | None = None,
    sfreq: float = EEG_SFREQ,
    assume_volts: bool = True,
) -> mne.io.RawArray:
    """Build a Raw from the (n_times, 32) pickle array.

    ``data`` should contain only the 32 EEG channels (not the counter column).
    MNE expects (n_channels, n_times) in volts.
    """
    arr = np.asarray(data, dtype=float).T  # -> (32, n_times)
    if not assume_volts:
        # Treat as microvolts and convert.
        arr = arr * 1e-6
    info = make_mne_info(ch_names=ch_names, sfreq=sfreq)
    return mne.io.RawArray(arr, info, verbose="ERROR")


def _resolve_reference(
    raw: mne.io.Raw,
    reference,
    bads: list[str],
) -> tuple[str, list[str] | None]:
    """Pick an effective reference given bad-channel list.

    ``reference='auto'`` prefers linked mastoids, falls back to common-average
    over good channels if mastoids are bad. Returns (kind, ref_channels).
    """
    if reference is None:
        return "none", None
    if reference == "average":
        return "average", None
    # Treat tuple/list or 'auto' similarly.
    if reference == "auto":
        wanted = list(EEG_MASTOIDS)
    elif isinstance(reference, str):
        return "named", [reference]
    else:
        wanted = list(reference)
    good_ref = [c for c in wanted if c not in bads]
    if not good_ref:
        return "average", None
    return "named", good_ref


def preprocess_eeg(
    raw: mne.io.Raw,
    *,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch: float | None = 60.0,
    reference: str | tuple[str, ...] = "auto",
    detect_bads: bool = True,
    interpolate_bads: bool = True,
    apply_ica: bool = False,
    ica_n_components: float | int = 0.99,
    random_state: int = 0,
    return_info: bool = False,
):
    """Robust preprocessing pipeline.

    Order: copy → mark bads (pre-filter, catches flat/saturated) → notch →
    band-pass → (re-check bads post-filter) → re-reference with fallback →
    optionally interpolate bads → optional ICA.

    Parameters
    ----------
    reference : 'auto' | 'average' | str | tuple[str, ...] | None
        'auto' = linked mastoids if both good, else common-average over good
        channels (PREP-style).
    detect_bads : bool
        Run ``detect_bad_channels`` and add to ``raw.info['bads']``.
    interpolate_bads : bool
        After re-referencing, interpolate marked bads so subsequent analyses
        have 32 channels of valid data.
    return_info : bool
        If True, return ``(raw, info_dict)`` where ``info_dict`` contains
        {'bads': [...], 'ref': 'named'|'average', 'ref_channels': [...]}.
    """
    raw = raw.copy()

    # 1) Early bad-channel marking catches dead/saturated electrodes whose
    #    involvement in the reference would collapse the data.
    bads: list[str] = []
    if detect_bads:
        bads = detect_bad_channels(raw)
        raw.info["bads"] = list(set(list(raw.info.get("bads", [])) + bads))

    # 2) Filtering.
    if notch is not None:
        raw.notch_filter(freqs=notch, verbose="ERROR")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    # 3) Re-run bad detection after filtering in case saturation only manifests
    #    post-filter (e.g., edge rings from a clipping channel).
    if detect_bads:
        post_bads = detect_bad_channels(raw)
        new_bads = sorted(set(bads) | set(post_bads))
        raw.info["bads"] = list(set(list(raw.info.get("bads", [])) + new_bads))
        bads = new_bads

    # 4) Reference (respecting bads).
    kind, ref_channels = _resolve_reference(raw, reference, bads)
    if kind == "named":
        raw.set_eeg_reference(ref_channels=ref_channels, verbose="ERROR")
    elif kind == "average":
        # Average-reference excludes channels already in info['bads'].
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # 5) Interpolate bads via spherical splines (requires a montage).
    if interpolate_bads and raw.info.get("bads"):
        try:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        except Exception:
            pass  # leave bads flagged; skip interpolation on failure

    # 6) ICA.
    if apply_ica:
        ica = mne.preprocessing.ICA(
            n_components=ica_n_components,
            random_state=random_state,
            method="fastica",
            max_iter="auto",
        )
        try:
            ica.fit(raw, verbose="ERROR")
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"], verbose="ERROR")
            ica.exclude = eog_idx
            ica.apply(raw, verbose="ERROR")
        except RuntimeError:
            # Single-component data (still rank-deficient); skip ICA gracefully.
            pass

    if return_info:
        return raw, dict(bads=bads, ref=kind, ref_channels=ref_channels)
    return raw


def regress_out_gaze(
    eeg: np.ndarray,
    gaze: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Remove gaze-linked variance from EEG via ridge regression, channel-wise.

    Parameters
    ----------
    eeg : (n_times, n_channels) array — EEG to clean.
    gaze : (n_times, n_gaze_features) regressors sampled at the same rate as EEG
        (e.g. up-sampled gaze_x, gaze_y, and their derivatives).
    ridge : L2 penalty.

    Returns
    -------
    residual EEG, same shape.
    """
    X = np.asarray(gaze, dtype=float)
    Y = np.asarray(eeg, dtype=float)
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    XtX = X.T @ X
    XtX += ridge * np.eye(XtX.shape[0])
    B = np.linalg.solve(XtX, X.T @ Y)
    Y_hat = X @ B
    return Y - Y_hat
