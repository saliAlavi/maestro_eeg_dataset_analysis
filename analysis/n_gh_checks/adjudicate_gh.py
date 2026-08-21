"""Adjudicate the GitHub AADModel's strict-LOSO 4-class edge: is it neural?

Loads each held-out subject's trained strict-LOSO checkpoint (speaker4, EEG) and,
on that subject's test set, runs the same confound controls we apply to our own
baseline:
  * real accuracy (reproduces the reported ~0.356);
  * EEG-shuffle null -- permute the test EEG across trials so it no longer matches
    the trial's candidates; the learned audio encoder + similarity head are intact.
    If this stays at 0.25 the edge is neural; if it is elevated, the audio side is
    reading the attended talker's acoustic marking (not the brain);
  * causal-lag curve -- accuracy with the EEG shifted later than the audio
    (causal, audio leads EEG by ~100-250 ms) vs earlier (anti-causal). Genuine
    cortical tracking is causal.

Eval-only: no retraining. Writes results/adjudicate/s{S}.json + prints aggregate.
"""
import argparse, glob, json, os
import numpy as np
import torch
from scipy import stats

import gh_data as D
from gh_models import AADModel

CKPT = "/fs/scratch/PAS2301/alialavi/projects/n_gh_checks/ckpt/loso/speaker4"
OUT = "/fs/scratch/PAS2301/alialavi/projects/n_gh_checks/results/adjudicate"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAGS = [-16, -8, -4, 0, 4, 8, 16]                 # samples @64 Hz (-250..+250 ms)


def _to(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


@torch.no_grad()
def _acc(model, eeg, audio, labels, bs=256):
    correct = 0
    for i in range(0, len(labels), bs):
        e = _to(eeg[i:i + bs])
        au = [_to(a[i:i + bs]) for a in audio]
        pred = model(e, None, None, None, au).argmax(1).cpu().numpy()
        correct += int((pred == labels[i:i + bs]).sum())
    return correct / max(1, len(labels))


def _shift(eeg, audio, L):
    """L>0: EEG later than audio (causal, audio leads). L<0: anti-causal."""
    if L == 0:
        return eeg, audio
    if L > 0:
        return eeg[:, L:, :], [a[:, :-L, :] for a in audio]
    return eeg[:, :L, :], [a[:, -L:, :] for a in audio]


def run_subject(s):
    ck = f"{CKPT}/loso_test_s{s}_eeg.pt"
    if not os.path.exists(ck):
        return None
    model = AADModel(["eeg"], n_speakers=4).to(DEV)
    model.load_state_dict(torch.load(ck, map_location=DEV)); model.eval()
    recs = D.load_subjects([s])[s]
    te = D.materialize_classif(recs, "speaker4", ["eeg"])
    eeg, audio, y = te["eeg"], te["audio"], te["labels"]

    real = _acc(model, eeg, audio, y)
    rng = np.random.default_rng(0)
    nulls = [_acc(model, eeg[rng.permutation(len(y))], audio, y) for _ in range(20)]
    null = float(np.mean(nulls))
    lag = {}
    for L in LAGS:
        e2, a2 = _shift(eeg, audio, L)
        lag[L] = _acc(model, e2, a2, y)
    causal = float(np.mean([lag[L] for L in LAGS if L > 0]))
    anti = float(np.mean([lag[L] for L in LAGS if L < 0]))
    return dict(subject=s, real=real, null=null, causal_margin=causal - anti,
                lag={str(k): v for k, v in lag.items()}, n=len(y))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 17)))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in a.subjects:
        r = run_subject(s)
        if r is None:
            print(f"s{s}: no checkpoint", flush=True); continue
        rows.append(r)
        json.dump(r, open(f"{OUT}/s{s}.json", "w"), default=float)
        print(f"[adjudicate|s{s}] real={r['real']:.3f} null={r['null']:.3f} "
              f"causal_margin={r['causal_margin']:+.3f} "
              f"lags={{{', '.join(f'{k}:{v:.2f}' for k, v in r['lag'].items())}}} n={r['n']}", flush=True)
    if len(rows) >= 2:
        R = np.array([r["real"] for r in rows]); N = np.array([r["null"] for r in rows])
        C = np.array([r["causal_margin"] for r in rows])
        t, p = stats.ttest_rel(R, N); p1 = p / 2 if t > 0 else 1 - p / 2
        tc, pc = stats.ttest_1samp(C, 0.0); pc1 = pc / 2 if tc > 0 else 1 - pc / 2
        print(f"\n=== AGGREGATE ({len(rows)} subj) ===", flush=True)
        print(f"real          = {R.mean():.3f} [{R.mean()-R.std()/len(R)**.5:.3f},{R.mean()+R.std()/len(R)**.5:.3f}]", flush=True)
        print(f"EEG-shuffle null = {N.mean():.3f}", flush=True)
        print(f"real - null   = {(R-N).mean():+.3f}  paired t={t:.2f} p1={p1:.2e}", flush=True)
        print(f"causal_margin = {C.mean():+.3f}  t={tc:.2f} p1={pc1:.3f}", flush=True)
        json.dump([r for r in rows], open(f"{OUT}/all.json", "w"), default=float)


if __name__ == "__main__":
    main()
