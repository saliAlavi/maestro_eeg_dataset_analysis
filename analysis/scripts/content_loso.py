"""content_loso: LEAVE-ONE-SUBJECT-OUT for the content recon matcher.

Train on 15 subjects, test on the held-out subject whose EEG was NEVER seen. The strictest
generalization regime (cf. within-subject ~0.40, trial-disjoint cross-subject 0.355).
Audio is shared across subjects, but the RECON matcher (raw candidate, no learnable stimulus
transform) is immune to the target-content confound, so LOSO is clean here. EEG-SHUFFLE null
(SHUFFLE_EEG=1, consumed inside V6.load) must drop to chance 0.25.

ENC=improved|basic  -> full-res multi-scale encoder vs the old 8x-pooled encoder.
SSL=1               -> masked-time SSL pretrain on TRAINING subjects ONLY (held-out excluded).
FiLM is N/A in LOSO (unseen subject has no learned subject embedding) -> forced off.
"""
import os, numpy as np, torch
os.environ["FILM"] = "0"                       # FiLM inapplicable to unseen subjects
os.environ.setdefault("ENC", "improved")
os.environ.setdefault("SSL", "0")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_v6 as V6                         # reads SHUFFLE_EEG/FILM at import
from content_attr import Recon, train, pred     # recon matcher (ENC-aware, FiLM-free)

dev = V6.dev
SSL = os.environ.get("SSL", "0") == "1"
ENC = os.environ.get("ENC", "improved")


def main():
    torch.manual_seed(0); np.random.seed(0)
    E, C, Y, TK, SB = V6.load(); nb = C.shape[2]; n_subj = int(SB.max()) + 1
    print(f"ENC={ENC} SSL={SSL} SHUFFLE_EEG={V6.SHUF} n={len(Y)} n_subj={n_subj}  LEAVE-ONE-SUBJECT-OUT", flush=True)
    accs = []
    for s in np.unique(SB):
        tr = SB != s; te = SB == s
        ssl_state = V6.ssl_pretrain(E[tr], SB[tr], n_subj) if SSL else None   # train subjects only
        m = Recon(nb, n_subj)
        if ssl_state is not None: m.enc.load_state_dict(ssl_state)
        m = train(m, E[tr], C[tr], Y[tr], ep=25)
        a = float((pred(m, E[te], C[te]) == Y[te]).mean()); accs.append(a)
        print(f"  held-out S{int(s)+1:2d}: n_test={te.sum()} acc={a:.3f}", flush=True)
    a = np.array(accs)
    print(f"\n=== content LOSO  ENC={ENC} SSL={SSL}  (chance 0.25) ===")
    print(f"  mean={a.mean():.3f} +- {a.std():.3f}   {'<-- EEG-SHUFFLE NULL' if V6.SHUF else ''}")


if __name__ == "__main__":
    main()
