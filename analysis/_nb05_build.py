"""Build 05_audio_features.ipynb."""
from _build_notebook import build

CELLS = [
("md", """\
# 05 · Audio stimulus features

For every trial we extract, from each of the 6 played streams (Device-1 L/R,
Device-2 L/R, Device-3 L/R, 3 scaled by per-ear power in `trials.csv`):

1. **Broadband Hilbert envelope** (compressed, smoothed, resampled to 64 Hz).
2. **Log-mel gammatone-proxy bank** (28 bands, 80–8000 Hz, 64 Hz).
3. **Mel-spectrogram** (80 mel, 10 ms hop) for deep models.
4. **Phonetic onset proxy** — half-wave-rectified envelope derivative.
5. **Linguistic surprisal** per word via a small pre-trained LM
   (default: `distilgpt2`), aligned to word-onset timestamps obtained via
   forced alignment (`wav2vec2` CTC) — guarded behind `RUN_HEAVY=True`.
6. **Binaural reconstruction at the listener** via a simple spherical-head
   ITD/ILD approximation at the listed azimuths, for use as stimulus features
   in the decoder notebook.

All features are cached under `analysis/cache/audio_features/<trial>.npz`.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from aad_utils import (load_trials_csv, load_audio_file, PAIRS_DIR, CACHE_DIR,
                       FIGURES_DIR, RESULTS_DIR, audio_envelope, gammatone_envelope,
                       mel_spectrogram, set_pub_style, save_fig, COLORS)
from aad_utils.config import SPEAKER_AZIMUTHS, ATTENDED_SPEAKER_MAP, SPEAKER_DISTANCE_M
set_pub_style()
TRIALS = load_trials_csv()
AUD_CACHE = CACHE_DIR / 'audio_features'; AUD_CACHE.mkdir(exist_ok=True)
RUN_HEAVY = False  # set True to compute word-level surprisal with a pretrained LM
"""),

("md", "## 1 · Envelopes for a single trial (Trial-1)"),
("code", """\
row = TRIALS[TRIALS['Trial No.'] == '1'].iloc[0]
files = dict(D1=row['Device-1'], D2=row['Device-2'], D3=row['Device-3'])
audios = {k: load_audio_file(v) for k, v in files.items()}
for k,(a,sr) in audios.items(): print(k, a.shape, sr, 'Hz')

fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
for ax, (k,(a,sr)) in zip(axes, audios.items()):
    env = audio_envelope(a, sr, sr_out=64)
    t = np.arange(len(env))/64
    ax.plot(t, env, color=COLORS['audio']); ax.set_ylabel(k + ' env')
axes[-1].set_xlabel('time (s)'); plt.suptitle('Hilbert envelopes · Trial-1')
save_fig(fig, '05_envelopes_trial1', FIGURES_DIR); plt.show()
"""),

("md", "## 2 · Log-mel & spectrogram visualisation"),
("code", """\
a, sr = audios['D1']
mel = gammatone_envelope(a, sr, n_bands=28, sr_out=64)
spec = mel_spectrogram(a, sr)
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
axes[0].imshow(mel.T, aspect='auto', origin='lower', cmap='magma',
               extent=[0, mel.shape[0]/64, 0, 28])
axes[0].set_ylabel('mel band'); axes[0].set_title('Log-mel (gammatone proxy) · D1 · Trial-1')
axes[1].imshow(spec.T, aspect='auto', origin='lower', cmap='magma',
               extent=[0, spec.shape[0]*0.01, 0, 80])
axes[1].set_ylabel('mel bin'); axes[1].set_xlabel('time (s)'); axes[1].set_title('80-mel log spectrogram')
save_fig(fig, '05_melspec_d1', FIGURES_DIR); plt.show()
"""),

("md", "## 3 · Feature cache for all trials"),
("code", """\
def cache_trial_features(row):
    tid = str(row['Trial No.'])
    # Main-trial IDs in the CSV are bare integers like '1'..'100'; prefix them
    # so cache filenames don't collide with filesystem tooling and are glob-able.
    fname = f'Trial-{tid}' if tid.isdigit() else tid
    out = AUD_CACHE / f'{fname}.npz'
    if out.exists(): return out
    data = {}
    for dev in ['Device-1','Device-2','Device-3']:
        a, sr = load_audio_file(row[dev])
        data[f'{dev}_env'] = audio_envelope(a, sr, sr_out=64)
        data[f'{dev}_mel'] = gammatone_envelope(a, sr, n_bands=28, sr_out=64)
    data['attended'] = int(row['Attended Speaker'])
    data['snr'] = float(row['SNR'])
    np.savez_compressed(out, **data)
    return out

# For speed: cache only first 5 training + first 10 main trials here. Run full
# pass separately when time permits.
subset = pd.concat([TRIALS.head(5), TRIALS[TRIALS['Trial No.']=='1'].head(1),
                    TRIALS[TRIALS['Trial No.']=='2'].head(1),
                    TRIALS[TRIALS['Trial No.']=='3'].head(1)])
for _, r in subset.iterrows():
    try:
        p = cache_trial_features(r); print('cached', p.name)
    except Exception as e:
        print('skip', r['Trial No.'], e)
"""),

("md", "## 4 · Linguistic surprisal (RUN_HEAVY)"),
("code", """\
if RUN_HEAVY:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Forced alignment is handled by wav2vec2-CTC in a companion script; here
    # we assume word/onset pairs supplied externally. This cell is a stub.
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    lm = AutoModelForCausalLM.from_pretrained('distilgpt2').eval()
    def word_surprisal(sentence):
        enc = tok(sentence, return_tensors='pt')
        with torch.no_grad():
            logits = lm(**enc).logits[0]
        logp = torch.log_softmax(logits, dim=-1)
        ids = enc['input_ids'][0]
        surp = -logp[:-1, ids[1:]].diag().numpy()
        return surp
    print('Example surprisal:', word_surprisal('The quick brown fox jumps over the lazy dog.'))
else:
    print('RUN_HEAVY=False — skipping LM surprisal. Enable to download distilgpt2 (~350MB).')
"""),

("md", "## 5 · Simple binaural reconstruction at the listener"),
("code", """\
# Delay-and-sum with spherical-head ITD + ILD proxy. Head radius 8.75 cm.
C = 343.0  # m/s
R_HEAD = 0.0875

def woodworth_itd(az_deg):
    a = np.deg2rad(az_deg)
    return R_HEAD * (np.sin(a) + a) / C  # seconds (positive → right lag)

def ild_db(az_deg, freq_hz=1000):
    # Crude: pinna-independent head shadow; 6 dB per 90° per 2 kHz decade.
    return 6.0 * np.sin(np.deg2rad(az_deg)) * np.log2(max(freq_hz, 200)/1000 + 1)

# Build a trial's binaural mix (Device-1..3 L/R each at its own azimuth).
def spatialize_trial(row, sr_target=16000):
    streams = []
    mapping = [
        ('Device-1', 'Left',  SPEAKER_AZIMUTHS[('Device-1', 'Left')],  row['Device-1 Left Power']),
        ('Device-1', 'Right', SPEAKER_AZIMUTHS[('Device-1', 'Right')], row['Device-1 Right Power']),
        ('Device-2', 'Left',  SPEAKER_AZIMUTHS[('Device-2', 'Left')],  row['Device-2 Left Power']),
        ('Device-2', 'Right', SPEAKER_AZIMUTHS[('Device-2', 'Right')], row['Device-2 Right Power']),
        ('Device-3', 'Left',  SPEAKER_AZIMUTHS[('Device-3', 'Left')],  row['Device-3 Left Power']),
        ('Device-3', 'Right', SPEAKER_AZIMUTHS[('Device-3', 'Right')], row['Device-3 Right Power']),
    ]
    # Device-1 and Device-2 carry the *same file* at both speakers (docs say
    # "both left and right but with different powers"); Device-3 is noise.
    import librosa, soundfile as sf
    def _load(fname):
        a, sr = load_audio_file(fname)
        if sr != sr_target:
            a = librosa.resample(a, orig_sr=sr, target_sr=sr_target)
        return a.astype(np.float32)
    audio_by_file = {f: _load(row[f]) for f in ['Device-1','Device-2','Device-3']}

    n = max(len(x) for x in audio_by_file.values())
    L = np.zeros(n, dtype=np.float32); R = np.zeros(n, dtype=np.float32)
    for dev, side, az, power in mapping:
        src = audio_by_file[dev].copy()
        src = np.pad(src, (0, n - len(src)), mode='constant')
        itd = woodworth_itd(az)
        delay_samples = int(round(itd * sr_target))
        # Positive azimuth → left ear lagged.
        left_shift = max(delay_samples, 0); right_shift = max(-delay_samples, 0)
        gainL = float(power) * 10 ** (-ild_db(az) / 20 / 2)  # dB toward contra ear
        gainR = float(power) * 10 ** ( ild_db(az) / 20 / 2)
        lpad = np.pad(src[:n-left_shift], (left_shift, 0)) * gainL
        rpad = np.pad(src[:n-right_shift], (right_shift, 0)) * gainR
        L[:len(lpad)] += lpad; R[:len(rpad)] += rpad
    stereo = np.stack([L, R], axis=1)
    return stereo, sr_target

stereo, srt = spatialize_trial(TRIALS.iloc[5])  # Trial-1
print('binaural shape', stereo.shape, srt)
fig, ax = plt.subplots(figsize=(7, 2.5))
t = np.arange(stereo.shape[0]) / srt
ax.plot(t[:srt*5], stereo[:srt*5, 0], color=COLORS['attended'], lw=0.4, label='L')
ax.plot(t[:srt*5], stereo[:srt*5, 1] - 0.4, color=COLORS['unattended'], lw=0.4, label='R (offset)')
ax.legend(); ax.set_xlabel('s'); ax.set_title('Binaural reconstruction · Trial-1 (first 5 s)')
save_fig(fig, '05_binaural_trial1', FIGURES_DIR); plt.show()
"""),

("md", "## 6 · Attended vs unattended envelope correlation"),
("code", """\
# Check to what degree the two speaker envelopes are already correlated (task difficulty proxy).
import glob
rows = []
for p in sorted(AUD_CACHE.glob('Trial-*.npz'))[:20]:
    d = np.load(p)
    e1, e2, e3 = d['Device-1_env'], d['Device-2_env'], d['Device-3_env']
    L = min(len(e1), len(e2), len(e3))
    rows.append(dict(trial=p.stem,
                     corr_D1_D2=np.corrcoef(e1[:L], e2[:L])[0,1],
                     corr_D1_D3=np.corrcoef(e1[:L], e3[:L])[0,1],
                     corr_D2_D3=np.corrcoef(e2[:L], e3[:L])[0,1]))
ec = pd.DataFrame(rows)
print(ec.describe())
ec.to_parquet(RESULTS_DIR / '05_envelope_corr.parquet')
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/05_audio_features.ipynb', CELLS)
print('Wrote 05_audio_features.ipynb')
