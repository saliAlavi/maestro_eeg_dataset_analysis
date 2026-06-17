"""content_band: does restricting EEG to the canonical AAD envelope-tracking band help?

Cortical speech-envelope tracking lives in ~1-8 Hz (delta-theta). The cache EEG is broadband
(~1-32 Hz @ 64 Hz). This bandpasses the EEG AMPLITUDE to several bands before the improved
recon matcher (within-subject, the regime where content is strongest, baseline 0.396 broadband).
NOTE: this is the amplitude in-band, NOT delta/theta PHASE features (those hurt earlier).
"""
import os, importlib.util, numpy as np, torch
from scipy.signal import butter, filtfilt
os.environ["ENC"] = "improved"; os.environ["SSL"] = "0"; os.environ["FILM"] = "0"
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_v6 as V6
from content_attr import Recon, train, pred
from sklearn.model_selection import StratifiedKFold
FS = 64.0
BANDS = os.environ.get("BANDS", "none;1,8;1,4;2,8;0.5,12").split(";")


def bandpass(E, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, E, axis=-1).astype(np.float32)


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, C, Y, TK, SB = V6.load(); nb = C.shape[2]; n_subj = int(SB.max()) + 1
    print(f"n={len(Y)} bands={BANDS}  (within-subject improved recon; broadband baseline 0.396)", flush=True)
    for band in BANDS:
        Eb = E if band == "none" else V6.zs(bandpass(E, *[float(x) for x in band.split(",")]), 2)
        accs = []
        for s in np.unique(SB):
            m = SB == s; Es, Cs, Ys = Eb[m], C[m], Y[m]; pr = np.zeros(len(Ys), int)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Es, Ys):
                mdl = train(Recon(nb, n_subj), Es[tr], Cs[tr], Ys[tr]); pr[te] = pred(mdl, Es[te], Cs[te])
            accs.append((pr == Ys).mean())
        print(f"  band={band:9s} within-subj mean={np.mean(accs):.3f} +- {np.std(accs):.3f}", flush=True)


if __name__ == "__main__":
    main()
