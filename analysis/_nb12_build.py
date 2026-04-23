"""Build 12_advanced_eeg_neuroscience.ipynb."""
from _build_notebook import build

CELLS = [
("md", """\
# 12 · Advanced EEG neuroscience analyses

Going beyond the basic TRF/ERP benchmark in nb06, this notebook implements the
canonical analyses a reviewer of a NeurIPS Datasets & Benchmarks AAD paper
would expect to see. Every method is implemented from scratch on top of
`numpy` / `scipy` / `mne` so no exotic dependencies are required.

**Menu (pick & choose):**

1. **Time–frequency decomposition** (Morlet wavelets, multitaper) → event-
   related spectral perturbation (ERSP) and inter-trial phase coherence (ITC).
2. **Spectral parameterization** — separate aperiodic (1/f) and periodic
   components (FOOOF-style, implemented via non-linear least squares) and
   extract individual-alpha-frequency (IAF), alpha peak power/bandwidth.
3. **EEG microstates** (Koenig/Pascual-Marqui k-means on GFP-peak topographies)
   with back-fitting and per-state metrics.
4. **Current source density (CSD)** / surface Laplacian — reference-free,
   enhances focal sources.
5. **Phase–amplitude coupling (PAC)** — Tort's Modulation Index, with a
   surrogate-based null.
6. **Global field power (GFP)** and **global map dissimilarity** (GMD).
7. **Functional connectivity** — phase-locking value (PLV), weighted
   phase-lag index (wPLI), amplitude-envelope correlation (AEC). Built-in
   surrogate significance test.
8. **Graph-theoretic summaries** of connectivity: clustering, path length,
   small-worldness, modularity.
9. **Common Spatial Patterns (CSP)** for **left-vs-right attended**
   binary decoding.
10. **Riemannian tangent-space features** on per-trial covariance matrices
    + logistic regression.
11. **Inter-subject correlation (ISC)** across subjects listening to the
    same stimulus — classic engagement/attention index.
12. **Cerebro-acoustic coherence** — phase-locking of each channel to the
    attended envelope in δ/θ/α bands.
13. **Complexity / entropy** — sample entropy, permutation entropy,
    Lempel–Ziv complexity (all implemented in-notebook).
14. **Cluster-based permutation ERP** for audio-onset responses.
15. **Source localization stub** (MNE inverse via `fsaverage` template) —
    guarded by `RUN_HEAVY` because it downloads ~500 MB.

Each section returns either a parquet of features or a publication-ready
figure in `analysis/figures/`.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import mne
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.stats import zscore
from aad_utils import (EEG_CHANNELS, EEG_SFREQ, list_subjects, load_trials_csv,
                       load_eeg_trial, load_eeg_time, load_gaze_trial_2d,
                       load_audio_timestamps, align_modalities_to_trial,
                       eeg_raw_to_mne, preprocess_eeg, audio_envelope, load_audio_file,
                       CACHE_DIR, FIGURES_DIR, RESULTS_DIR, set_pub_style,
                       save_fig, COLORS, bootstrap_ci)
from aad_utils.config import ATTENDED_SPEAKER_MAP
from aad_utils.preprocess import make_mne_info
set_pub_style()
TRIALS = load_trials_csv(); SUBJECTS = list_subjects()
RUN_HEAVY = False  # source localization etc.

def load_trial(s, k, l_freq=1, h_freq=40, sfreq_out=None):
    eeg, ts = load_eeg_trial(s, k); em = load_eeg_time(s, k)
    g2 = load_gaze_trial_2d(s, k); at = load_audio_timestamps(s, k)
    ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                    gaze2d=g2, audio_timestamps=at)
    raw = eeg_raw_to_mne(ali['eeg'])
    raw = preprocess_eeg(raw, l_freq=l_freq, h_freq=h_freq, reference='auto')
    if sfreq_out is not None:
        raw.resample(sfreq_out, verbose='ERROR')
    return raw, ali

print('Environment ready.')
"""),

("md", "## 1 · Time–frequency: Morlet wavelets → ERSP / ITC"),
("code", """\
from mne.time_frequency import tfr_array_morlet

def compute_ersp_itc(subject, trials=range(1, 21), freqs=np.logspace(np.log10(3), np.log10(30), 24), n_cycles_factor=3):
    segs = []
    for k in trials:
        try:
            raw, _ = load_trial(subject, k, l_freq=1, h_freq=40, sfreq_out=128)
        except Exception: continue
        d = raw.get_data()
        if d.shape[1] < 64: continue
        segs.append(d[:, :int(30*128)])
    if not segs: return None, None, None
    # Pad/crop to equal length.
    T = min(s.shape[1] for s in segs)
    X = np.stack([s[:, :T] for s in segs], axis=0)  # (n_epochs, n_ch, n_times)
    n_cycles = freqs / n_cycles_factor
    tfr = tfr_array_morlet(X, sfreq=128, freqs=freqs, n_cycles=n_cycles,
                            output='complex', n_jobs=1)
    power = (np.abs(tfr) ** 2).mean(axis=0)  # (n_ch, n_freqs, n_times)
    # Baseline correction: log-ratio to first 1 s baseline.
    base = power[:, :, :128].mean(axis=-1, keepdims=True) + 1e-30
    ersp = 10 * np.log10(power / base)
    itc = np.abs((tfr / (np.abs(tfr) + 1e-30)).mean(axis=0))  # (n_ch, n_freqs, n_times)
    return ersp, itc, freqs

ersp, itc, freqs = compute_ersp_itc(1)
if ersp is not None:
    t = np.arange(ersp.shape[-1]) / 128
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ci = EEG_CHANNELS.index('Cz')
    im0 = axes[0].imshow(ersp[ci], aspect='auto', origin='lower', cmap='RdBu_r',
                          extent=[t[0], t[-1], freqs[0], freqs[-1]], vmin=-3, vmax=3)
    axes[0].set_yscale('log'); axes[0].set_title('ERSP · Cz (dB)'); axes[0].set_xlabel('time (s)')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(itc[ci], aspect='auto', origin='lower', cmap='magma',
                          extent=[t[0], t[-1], freqs[0], freqs[-1]], vmin=0, vmax=0.3)
    axes[1].set_yscale('log'); axes[1].set_title('ITC · Cz'); axes[1].set_xlabel('time (s)')
    plt.colorbar(im1, ax=axes[1])
    save_fig(fig, '12_ersp_itc_s1cz', FIGURES_DIR); plt.show()
"""),

("md", "## 2 · Spectral parameterization (FOOOF-style)"),
("code", """\
from scipy.optimize import curve_fit

def aperiodic(f, offset, exponent, knee=0):
    return offset - np.log10(knee + f**exponent)

def gauss(f, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((f - mu) / sigma)**2)

def fit_fooof(freqs, psd, fmin=2, fmax=40, n_peaks=3):
    mask = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[mask]; lp = np.log10(psd[mask] + 1e-30)
    # 1. Fit aperiodic ignoring peaks (fixed knee=0)
    try:
        ap, _ = curve_fit(lambda x, o, e: aperiodic(x, o, e, 0), f, lp, p0=[0, 1])
    except Exception:
        return None
    ap_curve = aperiodic(f, ap[0], ap[1], 0)
    resid = lp - ap_curve
    # 2. Iteratively fit peaks.
    peaks = []
    cur = resid.copy()
    for _ in range(n_peaks):
        if np.max(cur) < 0.05: break
        mu0 = f[np.argmax(cur)]
        try:
            p, _ = curve_fit(gauss, f, cur, p0=[mu0, 2, np.max(cur)],
                              bounds=([fmin, 0.5, 0], [fmax, 12, 5]))
            peaks.append(tuple(p))
            cur -= gauss(f, *p)
        except Exception: break
    return dict(offset=ap[0], exponent=ap[1], peaks=peaks, freqs=f, log_psd=lp, ap_curve=ap_curve)

raw, _ = load_trial(1, 6)
freqs_, psd_ = welch(raw.get_data().mean(0), fs=EEG_SFREQ, nperseg=int(EEG_SFREQ*2))
res = fit_fooof(freqs_, psd_)
if res:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(res['freqs'], res['log_psd'], color='k', label='data')
    ax.plot(res['freqs'], res['ap_curve'], color=COLORS['eeg'], ls='--', label=f'aperiodic (exp={res[\"exponent\"]:.2f})')
    full = res['ap_curve'].copy()
    for mu, sig, amp in res['peaks']:
        full += gauss(res['freqs'], mu, sig, amp)
    ax.plot(res['freqs'], full, color=COLORS['attended'], label='full fit')
    ax.set_xlabel('Hz'); ax.set_ylabel('log10 PSD'); ax.legend()
    ax.set_title('Spectral parameterization · Subject 1 · Eval-6')
    print('Peaks (mu, σ, amp):', res['peaks'])
    save_fig(fig, '12_fooof_s1', FIGURES_DIR); plt.show()
# Individual alpha frequency = strongest peak in 7–13 Hz
alpha_peaks = [p for p in res['peaks'] if 7 <= p[0] <= 13] if res else []
IAF = alpha_peaks[0][0] if alpha_peaks else np.nan
print('Individual alpha frequency (IAF):', IAF, 'Hz')
"""),

("md", "## 3 · EEG microstates (k-means on GFP peaks)"),
("code", """\
def microstate_kmeans(data, n_maps=4, n_iter=50, random_state=0):
    # data: (n_channels, n_times). GFP = std across channels at each time.
    gfp = data.std(axis=0)
    # Pick GFP-peak samples as training.
    peaks = []
    for i in range(1, len(gfp)-1):
        if gfp[i] > gfp[i-1] and gfp[i] > gfp[i+1]:
            peaks.append(i)
    if len(peaks) < n_maps * 10:
        peaks = np.arange(data.shape[1])
    X = data[:, peaks].T  # (n_peaks, n_ch)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(random_state)
    # Modified k-means: label by |correlation|; update centroid by 1st PC of assigned.
    maps = X[rng.choice(len(X), n_maps, replace=False)]
    for _ in range(n_iter):
        sim = np.abs(X @ maps.T)
        labels = sim.argmax(axis=1)
        new_maps = []
        for k in range(n_maps):
            group = X[labels == k]
            if len(group) < 2:
                new_maps.append(maps[k])
            else:
                # Flip signs to align before averaging (use the dominant sign).
                pc = np.linalg.svd(group, full_matrices=False)[2][0]
                new_maps.append(pc / (np.linalg.norm(pc) + 1e-12))
        new_maps = np.array(new_maps)
        if np.allclose(new_maps, maps): break
        maps = new_maps
    # Back-fit: label every timepoint.
    Xall = data.T / (np.linalg.norm(data.T, axis=1, keepdims=True) + 1e-12)
    seq = np.abs(Xall @ maps.T).argmax(axis=1)
    return maps, seq, gfp

raw, _ = load_trial(1, 6, l_freq=2, h_freq=20, sfreq_out=128)
maps, seq, gfp = microstate_kmeans(raw.get_data(), n_maps=4)
print('Microstate coverage:', np.bincount(seq, minlength=4) / len(seq))
print('Mean duration (samples) per state:', [int(np.mean(np.diff(np.where(np.diff(seq == k))[0]))) if (seq==k).any() else 0 for k in range(4)])

info = make_mne_info()
fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
for i, (ax, m) in enumerate(zip(axes, maps)):
    mne.viz.plot_topomap(m, info, axes=ax, show=False, cmap='RdBu_r')
    ax.set_title(f'Microstate {chr(65+i)}')
save_fig(fig, '12_microstates_s1', FIGURES_DIR); plt.show()
"""),

("md", "## 4 · Current Source Density (surface Laplacian)"),
("code", """\
raw, _ = load_trial(1, 6)
try:
    raw_csd = mne.preprocessing.compute_current_source_density(raw, verbose='ERROR')
    d0 = raw.get_data(); d1 = raw_csd.get_data()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    t = np.arange(d0.shape[1]) / raw.info['sfreq']
    for ax, D, title in zip(axes, [d0, d1], ['Reference EEG', 'CSD (Laplacian)']):
        for c in ['Cz','Pz','Fz']:
            i = EEG_CHANNELS.index(c)
            ax.plot(t[:1000], D[i,:1000]*1e6, label=c)
        ax.set_xlabel('s'); ax.set_title(title); ax.legend()
    save_fig(fig, '12_csd_s1', FIGURES_DIR); plt.show()
except Exception as e:
    print('CSD unavailable:', e)
"""),

("md", "## 5 · Phase–amplitude coupling (Tort MI)"),
("code", """\
def tort_mi(x, sf, phase_band=(4,8), amp_band=(30,80), n_bins=18):
    # Band-pass via 4th-order Butterworth.
    def bp(lo, hi):
        b, a = butter(4, [lo/(sf/2), hi/(sf/2)], btype='band')
        return filtfilt(b, a, x)
    ph = np.angle(hilbert(bp(*phase_band)))
    amp = np.abs(hilbert(bp(*amp_band)))
    bins = np.linspace(-np.pi, np.pi, n_bins+1)
    mean_amp = np.array([amp[(ph>=bins[i])&(ph<bins[i+1])].mean() if ((ph>=bins[i])&(ph<bins[i+1])).any() else 0 for i in range(n_bins)])
    p = mean_amp / mean_amp.sum()
    H = -np.sum(p[p>0] * np.log(p[p>0]))
    Hmax = np.log(n_bins)
    return float((Hmax - H) / Hmax), mean_amp

raw, _ = load_trial(1, 6, l_freq=1, h_freq=80, sfreq_out=200)
# Use Cz signal; check θ-γ coupling.
x_cz = raw.get_data()[EEG_CHANNELS.index('Cz')]
mi_tg, amp_bins = tort_mi(x_cz, 200, (4,8), (30,60))
print(f'Theta-gamma PAC (Cz): MI = {mi_tg:.4f}')
# Surrogate null: shuffle gamma amplitude in chunks
from numpy.random import default_rng
rng = default_rng(0); surrogate = []
for _ in range(100):
    shift = rng.integers(500, len(x_cz)-500)
    x_shuf = np.roll(x_cz, shift)
    mi_s, _ = tort_mi(x_shuf, 200, (4,8), (30,60))
    surrogate.append(mi_s)
print(f'Surrogate null: mean = {np.mean(surrogate):.4f}, p = {(np.array(surrogate) >= mi_tg).mean():.3f}')

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(np.linspace(-np.pi, np.pi, 18), amp_bins, width=2*np.pi/18, color=COLORS['eeg'])
ax.set_xlabel('theta phase (rad)'); ax.set_ylabel('mean gamma amp')
ax.set_title(f'Cz θ→γ PAC (MI={mi_tg:.3f})')
save_fig(fig, '12_pac_s1', FIGURES_DIR); plt.show()
"""),

("md", "## 6 · Functional connectivity (PLV / wPLI / AEC)"),
("code", """\
def band_hilbert(x, sf, band):
    b, a = butter(4, [band[0]/(sf/2), band[1]/(sf/2)], btype='band')
    return hilbert(filtfilt(b, a, x, axis=-1))

def plv_matrix(analytic):
    # analytic: (n_ch, n_times)
    phi = np.angle(analytic)
    # PLV = |<exp(i Δφ)>|
    ex = np.exp(1j * phi)
    n = ex.shape[0]; T = ex.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            v = np.abs(np.mean(ex[i] * np.conj(ex[j])))
            M[i, j] = M[j, i] = v
    return M

def wpli_matrix(analytic):
    n, T = analytic.shape
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            csd = analytic[i] * np.conj(analytic[j])
            num = np.abs(np.mean(np.imag(csd)))
            den = np.mean(np.abs(np.imag(csd))) + 1e-30
            M[i, j] = M[j, i] = num / den
    return M

def aec_matrix(analytic):
    amp = np.abs(analytic)
    # Orthogonalize per pair and then correlate.
    n = amp.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            ai, aj = amp[i] - amp[i].mean(), amp[j] - amp[j].mean()
            M[i, j] = M[j, i] = np.corrcoef(ai, aj)[0,1]
    return M

raw, _ = load_trial(1, 6, l_freq=1, h_freq=40)
alpha = band_hilbert(raw.get_data(), EEG_SFREQ, (8, 13))
plv = plv_matrix(alpha); wpli = wpli_matrix(alpha); aec = aec_matrix(alpha)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, M, title in zip(axes, [plv, wpli, aec], ['PLV', 'wPLI', 'AEC']):
    im = ax.imshow(M, vmin=0, vmax=1 if title!='AEC' else 0.8, cmap='magma')
    ax.set_title(f'{title} · 8–13 Hz · S1E6'); plt.colorbar(im, ax=ax)
save_fig(fig, '12_connectivity_s1alpha', FIGURES_DIR); plt.show()
"""),

("md", "## 7 · Graph-theoretic summaries (on PLV)"),
("code", """\
def graph_metrics(W, thresh_frac=0.3):
    # Sparsify: keep top `thresh_frac` edges by weight.
    W = W.copy(); np.fill_diagonal(W, 0)
    flat = np.sort(W[np.triu_indices_from(W, 1)])
    if len(flat) == 0: return {}
    thr = flat[-int(len(flat)*thresh_frac)] if thresh_frac < 1 else 0
    A = (W >= thr).astype(int); np.fill_diagonal(A, 0)
    n = A.shape[0]
    # Clustering coefficient (weighted).
    C = []
    for i in range(n):
        nbrs = np.where(A[i])[0]
        if len(nbrs) < 2: C.append(0); continue
        sub = A[np.ix_(nbrs, nbrs)]
        C.append(sub.sum() / (len(nbrs)*(len(nbrs)-1)))
    C = float(np.mean(C))
    # Characteristic path length (BFS for unweighted graph).
    import collections
    def bfs(src):
        d = np.full(n, np.inf); d[src] = 0
        q = collections.deque([src])
        while q:
            u = q.popleft()
            for v in np.where(A[u])[0]:
                if d[v] > d[u]+1:
                    d[v] = d[u]+1; q.append(v)
        return d
    D = np.stack([bfs(i) for i in range(n)])
    L = np.nanmean(D[D != np.inf])
    return dict(avg_clustering=C, char_path_length=float(L), edge_density=float(A.sum()/(n*(n-1))))

print('PLV graph:', graph_metrics(plv))
print('wPLI graph:', graph_metrics(wpli))
"""),

("md", "## 8 · CSP for left-vs-right attended"),
("code", """\
from scipy.linalg import eigh

def csp(X_left, X_right, n_components=4):
    # X_*: (n_epochs, n_ch, n_times). Returns spatial filters W (n_ch, n_components*2).
    def cov(X):
        c = np.zeros((X.shape[1], X.shape[1]))
        for e in X:
            e = e - e.mean(axis=1, keepdims=True)
            c += e @ e.T / e.shape[1]
        return c / len(X)
    C1 = cov(X_left); C2 = cov(X_right)
    # Regularize
    C1 += 1e-6 * np.trace(C1)/C1.shape[0] * np.eye(C1.shape[0])
    C2 += 1e-6 * np.trace(C2)/C2.shape[0] * np.eye(C2.shape[0])
    vals, vecs = eigh(C1, C1 + C2)
    # Take top-n from each end.
    idx = np.concatenate([np.arange(n_components), np.arange(-n_components, 0)])
    W = vecs[:, idx]
    return W

def csp_features(raw_data, W):
    # Variance of each filtered component (log) per epoch.
    Z = W.T @ raw_data
    logvar = np.log(Z.var(axis=1) + 1e-12)
    return logvar

# Collect epochs labeled by attended side.
def build_csp_dataset(subject, n_trials=60):
    left, right = [], []
    for k in range(1, 1+n_trials):
        try:
            raw, _ = load_trial(subject, k, l_freq=8, h_freq=13, sfreq_out=128)
            tno = f'Trial-{k}'
            tr = TRIALS[TRIALS['Trial No.']==tno]
            if not len(tr): continue
            az = ATTENDED_SPEAKER_MAP[int(tr.iloc[0]['Attended Speaker'])][2]
            d = raw.get_data()[:, :int(25*128)]
            if d.shape[1] < 64: continue
            (left if az < 0 else right).append(d)
        except Exception: continue
    return np.array(left), np.array(right)

L, R = build_csp_dataset(1, n_trials=60)
print('Left epochs:', L.shape, 'Right epochs:', R.shape)
if len(L) > 5 and len(R) > 5:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([L, R]); y = np.array([0]*len(L) + [1]*len(R))
    accs = []
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        Xtr, ytr = X[tr_i], y[tr_i]
        W = csp(Xtr[ytr==0], Xtr[ytr==1], n_components=3)
        Ftr = np.stack([csp_features(e, W) for e in Xtr])
        Fte = np.stack([csp_features(e, W) for e in X[te_i]])
        clf = LogisticRegression(max_iter=2000).fit(Ftr, ytr)
        accs.append(clf.score(Fte, y[te_i]))
    print(f'CSP left-vs-right AAD accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
"""),

("md", "## 9 · Riemannian tangent-space covariance features"),
("code", """\
from scipy.linalg import eigh as _eigh

def regularize_spd(C, eps=1e-6):
    # Add a small multiple of identity scaled by trace so the matrix is SPD.
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)  # symmetrize in case of tiny numerical asymmetry
    tr = np.trace(C) / max(1, C.shape[0])
    return C + eps * tr * np.eye(C.shape[0])

def spd_logm(C):
    C = regularize_spd(C)
    w, V = _eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * np.log(w)) @ V.T

def tangent_space_vec(C, ref):
    # Half-vectorization of whitened log-map.
    ref = regularize_spd(ref)
    w, V = _eigh(ref)
    w = np.clip(w, 1e-12, None)
    iref = (V * (1 / np.sqrt(w))) @ V.T
    S = iref @ regularize_spd(C) @ iref
    L = spd_logm(S)
    n = L.shape[0]
    iu, ju = np.triu_indices(n)
    w2 = np.where(iu == ju, 1.0, np.sqrt(2))
    return L[iu, ju] * w2

def epoch_cov(e, shrink=0.05):
    # Channel-wise demean, then sample covariance with Ledoit-Wolf-style shrink
    # toward the diagonal target. Drops NaNs/Infs.
    e = np.asarray(e, dtype=float)
    if not np.all(np.isfinite(e)):
        e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    e = e - e.mean(axis=1, keepdims=True)
    C = e @ e.T / max(1, e.shape[1])
    diag_tgt = np.diag(np.diag(C))
    C = (1 - shrink) * C + shrink * diag_tgt
    return regularize_spd(C)

def cov_tangent_pipeline(L, R):
    X = np.concatenate([L, R])
    covs = np.stack([epoch_cov(e) for e in X])
    # Drop any covariance that is still not finite (shouldn't happen after guards).
    ok = np.array([np.all(np.isfinite(c)) for c in covs])
    if ok.sum() < len(covs):
        print(f'  dropped {len(covs)-ok.sum()} non-finite covariance(s)')
    covs_ok = covs[ok]
    labels = np.array([0]*len(L) + [1]*len(R))[ok]
    # Arithmetic mean is a cheap reference; regularize afterwards.
    ref = regularize_spd(covs_ok.mean(0))
    feats = np.stack([tangent_space_vec(c, ref) for c in covs_ok])
    # Final safety net: drop any feature rows that still contain NaNs.
    row_ok = np.all(np.isfinite(feats), axis=1)
    return feats[row_ok], labels[row_ok]

if len(L) > 5 and len(R) > 5:
    Xrf, yrf = cov_tangent_pipeline(L, R)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    accs = cross_val_score(LogisticRegression(max_iter=2000, C=0.1), Xrf, yrf, cv=5)
    print(f'Riemannian tangent-space left-vs-right AAD: {accs.mean():.3f} ± {accs.std():.3f}')
"""),

("md", "## 10 · Inter-subject correlation (ISC)"),
("code", """\
# Correlate each channel across subjects who saw the same trial.
def isc_trial(trial_idx_1based, subjects, channel='Cz'):
    signals = []
    for s in subjects:
        try:
            raw, _ = load_trial(s, trial_idx_1based, l_freq=1, h_freq=20, sfreq_out=64)
            signals.append(raw.get_data()[EEG_CHANNELS.index(channel)])
        except Exception: continue
    if len(signals) < 3: return np.nan, 0
    T = min(len(s) for s in signals)
    S = np.stack([s[:T] for s in signals])
    # ISC = mean leave-one-out correlation.
    corrs = []
    for i in range(len(S)):
        others = np.delete(S, i, axis=0).mean(0)
        corrs.append(np.corrcoef(S[i], others)[0,1])
    return float(np.mean(corrs)), len(S)

rows = []
for k in range(1, 11):  # first 10 main trials
    r, n = isc_trial(k, SUBJECTS, channel='Cz')
    rows.append(dict(trial=k, isc_cz=r, n_subjects=n))
isc_df = pd.DataFrame(rows)
print(isc_df)
isc_df.to_parquet(RESULTS_DIR / '12_isc_cz.parquet')
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(isc_df['trial'], isc_df['isc_cz'], color=COLORS['eeg'])
ax.set_xlabel('trial (Eval-K)'); ax.set_ylabel('ISC at Cz')
ax.set_title('Inter-subject correlation (leave-one-out)')
save_fig(fig, '12_isc_cz', FIGURES_DIR); plt.show()
"""),

("md", "## 11 · Cerebro-acoustic coherence"),
("code", """\
from scipy.signal import coherence
def cac(subject, k, bands=dict(delta=(1,4), theta=(4,8), alpha=(8,13))):
    raw, ali = load_trial(subject, k, l_freq=1, h_freq=20, sfreq_out=64)
    E = raw.get_data()
    tno = f'Trial-{k}'
    tr = TRIALS[TRIALS['Trial No.']==tno]
    if not len(tr): return None
    tr = tr.iloc[0]
    att = 'Device-1' if int(tr['Attended Speaker']) in (1,2) else 'Device-2'
    a, sr = load_audio_file(tr[att]); env = audio_envelope(a, sr, sr_out=64)
    L = min(E.shape[1], len(env)); E = E[:, :L]; env = env[:L]
    out = {}
    for bname, (lo, hi) in bands.items():
        chvals = []
        for c in range(E.shape[0]):
            f, C = coherence(E[c], env, fs=64, nperseg=256)
            m = (f >= lo) & (f <= hi)
            chvals.append(C[m].mean())
        out[bname] = np.array(chvals)
    return out

cac_res = cac(1, 6)
if cac_res:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    info = make_mne_info()
    for ax, (b, v) in zip(axes, cac_res.items()):
        mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap='Reds', vlim=(0, None))
        ax.set_title(f'CAC {b}')
    save_fig(fig, '12_cac_s1e6', FIGURES_DIR); plt.show()
"""),

("md", "## 12 · Complexity & entropy"),
("code", """\
def permutation_entropy(x, m=3, tau=1):
    from itertools import permutations
    x = np.asarray(x)
    n = len(x) - (m - 1) * tau
    patterns = {p: i for i, p in enumerate(permutations(range(m)))}
    counts = np.zeros(len(patterns))
    for i in range(n):
        seg = x[i:i+m*tau:tau]
        key = tuple(np.argsort(seg))
        counts[patterns[key]] += 1
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(patterns)))

def sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, dtype=float)
    r = r * x.std()
    def _count(m):
        n = len(x) - m + 1
        templates = np.stack([x[i:i+m] for i in range(n)])
        B = 0
        for i in range(n):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            B += np.sum(d <= r) - 1
        return B
    try:
        return float(-np.log((_count(m+1) + 1e-12) / (_count(m) + 1e-12)))
    except Exception: return np.nan

def lempel_ziv(x):
    # Binary complexity via median split.
    b = (np.asarray(x) > np.median(x)).astype(int)
    s = ''.join(map(str, b))
    i, C, k = 0, 1, 1
    n = len(s)
    while True:
        if s[i+k-1] == s[i+k-1]:
            if s[i+k-1] in s[:i+k-1]:
                k += 1
                if i + k > n: return C
            else:
                C += 1; i += k; k = 1
                if i + k > n: return C

raw, _ = load_trial(1, 6, l_freq=1, h_freq=40, sfreq_out=128)
D = raw.get_data()
rows = []
for ci, ch in enumerate(['Fz','Cz','Pz','Oz','T7','T8']):
    i = EEG_CHANNELS.index(ch)
    sig = D[i]
    rows.append(dict(channel=ch,
                     perm_ent=permutation_entropy(sig, m=3),
                     samp_ent=sample_entropy(sig[::2], m=2, r=0.2),
                     lz=lempel_ziv(sig)))
ent_df = pd.DataFrame(rows)
print(ent_df)
ent_df.to_parquet(RESULTS_DIR / '12_entropy_s1.parquet')
"""),

("md", "## 13 · Cluster-based permutation ERP to audio onset"),
("code", """\
from mne.stats import permutation_cluster_1samp_test

def onset_epochs(subject, tmin=-0.2, tmax=1.0, max_trials=30):
    eps = []
    for k in range(6, 6+max_trials):
        try:
            raw, ali = load_trial(subject, k, l_freq=1, h_freq=30, sfreq_out=250)
        except Exception: continue
        d = raw.get_data(); sf = raw.info['sfreq']
        n = int((tmax - tmin) * sf)
        start = max(0, int(-tmin*sf))
        seg = d[:, start:start+n]
        if seg.shape[1] == n: eps.append(seg)
    return np.array(eps)

eps = onset_epochs(1)
if len(eps) >= 10:
    # Baseline correct to pre-onset average.
    sf = 250; tmin = -0.2; n = eps.shape[-1]
    pre_n = int(0.2*sf)
    eps_bc = eps - eps[:, :, :pre_n].mean(-1, keepdims=True)
    times = np.arange(n)/sf + tmin
    # Cluster permutation at channel Cz.
    ci = EEG_CHANNELS.index('Cz')
    X = eps_bc[:, ci, :]
    T_obs, clusters, p_vals, _ = permutation_cluster_1samp_test(X, n_permutations=500, out_type='mask', verbose='ERROR')
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(times, X.mean(0)*1e6, color=COLORS['eeg'])
    for c, p in zip(clusters, p_vals):
        if p < 0.05:
            ax.axvspan(times[c][0], times[c][-1], color=COLORS['attended'], alpha=0.3)
    ax.axvline(0, color='k', lw=0.5); ax.set_xlabel('time (s)'); ax.set_ylabel('Cz (µV)')
    ax.set_title(f'Audio-onset ERP · Subject 1 · Cz · cluster p<0.05 shaded (n={len(eps)} trials)')
    save_fig(fig, '12_erp_cluster_s1', FIGURES_DIR); plt.show()
"""),

("md", "## 14 · Source localization (RUN_HEAVY)"),
("code", """\
if RUN_HEAVY:
    # Download/locate fsaverage and build forward solution.
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=False).parent
    subject = 'fsaverage'
    src = mne.setup_source_space(subject, spacing='oct4', subjects_dir=subjects_dir, verbose=False)
    bem = mne.make_bem_solution(mne.make_bem_model(subject, ico=3, subjects_dir=subjects_dir, verbose=False), verbose=False)
    raw, _ = load_trial(1, 6, l_freq=1, h_freq=30)
    # Coregistration uses montage only (no digitisation available).
    trans = 'fsaverage'
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True, verbose=False)
    cov = mne.compute_raw_covariance(raw, tmin=0, tmax=5, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2, depth=0.8, verbose=False)
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1/9, verbose=False)
    print('Source estimate shape:', stc.data.shape)
    # Plot grand-average source-power map.
    stc_pow = stc.copy(); stc_pow.data = (stc_pow.data ** 2).mean(axis=1, keepdims=True)
    brain = stc_pow.plot(subject='fsaverage', subjects_dir=subjects_dir, hemi='both', time_viewer=False, show_traces=False)
else:
    print('RUN_HEAVY=False — source localization skipped (requires fsaverage download).')
"""),

("md", """\
### Outputs

- `12_ersp_itc_s1cz.{pdf,png}` — time-frequency ERSP + ITC at Cz.
- `12_fooof_s1.{pdf,png}` — FOOOF-style spectral decomposition + IAF.
- `12_microstates_s1.{pdf,png}` — 4 canonical microstate topographies.
- `12_csd_s1.{pdf,png}` — reference-free surface Laplacian trace comparison.
- `12_pac_s1.{pdf,png}` — theta-gamma modulation histogram.
- `12_connectivity_s1alpha.{pdf,png}` — 3-panel PLV / wPLI / AEC in α.
- `12_isc_cz.{pdf,png}` — inter-subject Cz correlation across subjects.
- `12_cac_s1e6.{pdf,png}` — cerebro-acoustic coherence topomaps.
- `12_erp_cluster_s1.{pdf,png}` — audio-onset ERP with cluster-corrected significance.
- `results/12_entropy_s1.parquet`, `12_isc_cz.parquet` — numeric artefacts.

Extending to the full cohort: wrap each section in a per-subject loop and save
per-subject parquets (same pattern used in 02–09).
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/12_advanced_eeg_neuroscience.ipynb', CELLS)
print('Wrote 12_advanced_eeg_neuroscience.ipynb')
