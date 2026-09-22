"""Rich per-speaker audio features for content-based AAD matching.

For every main trial we extract, for the 4 *attendable* talkers (speakers 1-4 =
Device-1 L/R, Device-2 L/R; Device-3 is babble noise -> never attended), three
time-aligned feature streams on a shared 64 Hz grid spanning the full 30 s FLAC:

  * ``env``  -- 28-band gammatone envelope        (low-level, ~10 Hz)  [from windows]
  * ``w2v``  -- HuBERT-base layer-9 hidden states  (auditory / phonetic, ~5 Hz)
  * ``sem``  -- GPT-2 word surprisal + entropy + word-onset (semantic, ~2-3 Hz)

Rationale: envelope matching needs ms-precise EEG<->audio timing this corpus lacks
(software sync, per-trial jitter). HuBERT (slower) and especially GPT-2 surprisal
(N400-scale, ~400 ms broad response) degrade more gracefully under that jitter, so
a learned *mixture* over the three spaces is the best shot at content-based AAD.

Audio is shared across subjects (trials.csv is global) -> extract ONCE per trial
over the full FLAC; ``windows.build_trial`` slices each stream by the same
per-device anchor offset it uses for the envelope.

CLI:  python -m src.data.audio_features --trials 1-100 [--device cuda]
Cache: /fs/scratch/PAS2301/alialavi/cache/multimodal_aad__audiofeat/trial{K}.npz
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("audio_features")

SR_AUDIO = 16000
SR_OUT = 64.0
FLAC_DUR = 30.0
HUBERT_LAYER = 9                 # 0..12; mid layers are most brain-predictive
CACHE_DIR = Path("/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__audiofeat")
WHISPER_CACHE = "/fs/scratch/PAS2301/alialavi/cache/faster_whisper"   # models--Systran--faster-whisper-*
# speaker -> (device column, channel) ; ch0=L ch1=R. Only the 4 attendable talkers.
SPK_MAP = [("Device-1", 0), ("Device-1", 1), ("Device-2", 0), ("Device-2", 1)]


def _resample_time(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linear-resample (T_in, D) -> (n_out, D) along time."""
    T = x.shape[0]
    if T == n_out:
        return x.astype(np.float32)
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, n_out)
    return np.stack([np.interp(dst, src, x[:, d]) for d in range(x.shape[1])], 1).astype(np.float32)


class _Extractors:
    """Lazily-built, reused across trials."""

    def __init__(self, device: str):
        import torch
        import torchaudio
        from faster_whisper import WhisperModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        log.info("loading HuBERT_BASE ...")
        self.hubert = torchaudio.pipelines.HUBERT_BASE.get_model().to(device).eval()
        log.info("loading GPT-2 ...")
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
        log.info("loading faster-whisper (base) ...")
        wdev = "cuda" if device.startswith("cuda") else "cpu"
        ctype = "float16" if wdev == "cuda" else "int8"
        self.whisper = WhisperModel("base", device=wdev, compute_type=ctype,
                                    download_root=WHISPER_CACHE)

    # --- auditory: HuBERT layer-9 ---
    def hubert_feats(self, wav16: np.ndarray, n_out: int) -> np.ndarray:
        t = self.torch.from_numpy(wav16).float().to(self.device).unsqueeze(0)
        with self.torch.inference_mode():
            feats, _ = self.hubert.extract_features(t, num_layers=HUBERT_LAYER + 1)
        h = feats[HUBERT_LAYER][0].float().cpu().numpy()      # (T50, 768)
        return _resample_time(h, n_out)                       # (n_out, 768)

    # --- semantic: GPT-2 surprisal/entropy + word onsets, stepped onto 64 Hz ---
    def semantic_feats(self, wav16: np.ndarray, n_out: int) -> np.ndarray:
        torch = self.torch
        segs, _ = self.whisper.transcribe(wav16, word_timestamps=True, language="en")
        words = []                                            # (start, end, text)
        for s in segs:
            for w in (s.words or []):
                txt = w.word.strip()
                if txt:
                    words.append((float(w.start), float(w.end), txt))
        sem = np.zeros((n_out, 3), np.float32)                # [surprisal, entropy, onset]
        if not words:
            return sem
        text = " ".join(w[2] for w in words)
        enc = self.tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        offs = enc["offset_mapping"][0].tolist()
        with torch.inference_mode():
            logits = self.lm(ids).logits[0]                   # (L, V)
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        ent = (-(p * logp).sum(-1)).cpu().numpy()             # next-token entropy at each pos
        idl = ids[0].tolist()
        tok_nll = np.zeros(len(idl), np.float32)
        for t in range(1, len(idl)):
            tok_nll[t] = -float(logp[t - 1, idl[t]])          # surprisal of token t
        tok_ent = np.zeros(len(idl), np.float32)
        tok_ent[1:] = ent[:-1]
        # char-span of each word in `text`
        spans, c = [], 0
        for _, _, txt in words:
            c0 = text.find(txt, c)
            c0 = c if c0 < 0 else c0
            spans.append((c0, c0 + len(txt))); c = c0 + len(txt)
        for (st, en, _), (c0, c1) in zip(words, spans):
            sur = entr = 0.0
            for ti, (o0, o1) in enumerate(offs):
                if o1 > c0 and o0 < c1:                       # token overlaps this word
                    sur += tok_nll[ti]; entr = max(entr, tok_ent[ti])
            f0 = int(round(st * SR_OUT)); f1 = max(f0 + 1, int(round(en * SR_OUT)))
            f0 = min(f0, n_out - 1); f1 = min(f1, n_out)
            sem[f0:f1, 0] = sur                               # surprisal held over the word
            sem[f0:f1, 1] = entr
            sem[f0, 2] = 1.0                                  # onset impulse
        return sem


def extract_trial(K: int, ex: _Extractors, *, overwrite: bool = False) -> Path:
    import soundfile as sf
    from . import aad_compat as C

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"trial{K}.npz"
    if out.exists() and not overwrite:
        return out
    csv = C.load_trials_csv()
    row = csv[csv["Trial No."] == C.trial_name(K, "main")].iloc[0]
    n_out = int(round(FLAC_DUR * SR_OUT))
    w2v, sem = [], []
    flac_cache: dict[str, np.ndarray] = {}
    for dev, ch in SPK_MAP:
        fn = row[dev]
        if fn not in flac_cache:
            a, sr = sf.read(str(C.PAIRS_DIR / fn))
            assert sr == SR_AUDIO, f"expected 16k, got {sr}"
            flac_cache[fn] = np.asarray(a, np.float32)
        a = flac_cache[fn]
        wav = a[:, ch] if a.ndim == 2 else a
        w2v.append(ex.hubert_feats(wav, n_out))
        sem.append(ex.semantic_feats(wav, n_out))
    np.savez_compressed(
        out,
        w2v=np.stack(w2v).astype(np.float16),     # (4, n_out, 768)
        sem=np.stack(sem).astype(np.float32),     # (4, n_out, 3)
        sr=SR_OUT, dur=FLAC_DUR, layer=HUBERT_LAYER,
    )
    return out


def fit_pca(dim: int) -> Path:
    """Fit a HuBERT-768 -> `dim` PCA on frames sampled across all extracted trials."""
    import glob
    files = sorted(glob.glob(str(CACHE_DIR / "trial*.npz")))
    if not files:
        raise RuntimeError(f"no extracted trials in {CACHE_DIR}")
    rng = np.random.default_rng(0)
    frames = []
    for f in files:
        w = np.load(f)["w2v"].astype(np.float32).reshape(-1, 768)
        idx = rng.choice(len(w), size=min(500, len(w)), replace=False)
        frames.append(w[idx])
    X = np.concatenate(frames, 0)
    mean = X.mean(0); Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:dim]                                          # (dim, 768)
    out = CACHE_DIR / f"pca_w2v{dim}.npz"
    ev = float((S[:dim] ** 2).sum() / (S ** 2).sum())
    np.savez(out, components=comp.astype(np.float32), mean=mean.astype(np.float32))
    log.info("PCA %d: fit on %d frames from %d trials, explained var=%.3f -> %s",
             dim, len(X), len(files), ev, out.name)
    return out


def _parse_trials(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    os.environ.setdefault("HF_HOME", "/fs/scratch/PAS2301/alialavi/cache/hf")
    # NOTE: leave HF online-capable; this node has internet and all weights are cached.
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="1-100")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fit-pca", type=int, default=0, help="fit HuBERT PCA to this dim and exit")
    a = ap.parse_args()
    if a.fit_pca:
        fit_pca(a.fit_pca); return
    import torch
    from tqdm import tqdm
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s trials=%s", device, a.trials)
    ex = _Extractors(device)
    for K in tqdm(_parse_trials(a.trials), desc="trials"):
        p = extract_trial(K, ex, overwrite=a.overwrite)
        log.info("trial %d -> %s", K, p.name)


if __name__ == "__main__":
    main()
