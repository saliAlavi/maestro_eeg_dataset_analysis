"""Turn the ladder result JSONs into the markdown tables for FIX_REPORT.md."""

import glob
import json
import os
import sys

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "fixed")

DESC = {
    "A0":  ("upstream encoder + upstream head + raw candidates", "4-way raw"),
    "A0b": ("same, upstream's own Adam lr=1e-4", "4-way raw"),
    "A1":  ("+ v2 encoder (RF 1 s, GroupNorm, no final ReLU, centred)", "4-way raw"),
    "A2":  ("+ correlation head (time-centred)", "4-way raw"),
    "A3":  ("+ anti-shortcut loss (CLIP / null hinges / VICReg / adversary)", "4-way raw"),
    "A4":  ("+ quantile-matched candidates", "4-way qmatch"),
    "A5":  ("+ same-source shifted negatives, quantile-matched", "3-way"),
    "A6":  ("same, binary match-mismatch", "2-way"),
    "A5r": ("shifted negatives WITHOUT quantile matching", "3-way"),
    "M-gaze": ("gaze -> spatial head (no audio input)", "4-way"),
    "M-imu": ("head IMU -> spatial head (no audio input)", "4-way"),
    "M-video": ("scene video -> spatial head (no audio input)", "4-way"),
    "M-eeg-spatial": ("EEG -> spatial head (no audio input)", "4-way"),
    "M-behav": ("gaze+IMU+video fused spatial", "4-way"),
    "M-full": ("EEG coupling + all-modality spatial fusion", "4-way qmatch"),
    "A4f": ("lag control: EEG restricted to +125..+1109 ms (neural direction)", "4-way qmatch"),
    "A4b": ("lag control: EEG restricted to -1109..-125 ms (acausal)", "4-way qmatch"),
    "A6f": ("lag control: EEG restricted to +125..+1109 ms (neural direction)", "2-way"),
    "A6b": ("lag control: EEG restricted to -1109..-125 ms (acausal)", "2-way"),
}
ORDER = ["A0", "A0b", "A1", "A2", "A3", "A4", "A5r", "A5", "A6",
         "A4f", "A4b", "A6f", "A6b",
         "M-gaze", "M-imu", "M-video", "M-eeg-spatial", "M-behav", "M-full"]


def load():
    out, probes = {}, {}
    for f in sorted(glob.glob(os.path.join(RES, "*_within.json"))):
        d = json.load(open(f))
        probes.update(d.get("probes", {}))
        for k, v in d.get("configs", {}).items():
            out[k] = v
    return out, probes


def fmt(v, n=4, signed=False):
    return f"{v:+.{n}f}" if signed else f"{v:.{n}f}"


def main():
    cfgs, probes = load()
    sel = sys.argv[1] if len(sys.argv) > 1 else "sel_margin"

    print("### Audio-only acceptance probe (is the shortcut gone?)\n")
    print("| candidate construction | audio-only 4/3/2-way probe | chance | leak |")
    print("|---|---|---|---|")
    for k in sorted(probes):
        K = int(k.split("_K")[1]); ch = 1.0 / K
        print(f"| `{k}` | {probes[k]:.4f} | {ch:.4f} | {probes[k]-ch:+.4f} |")

    print("\n### Ladder (within-subject protocol, 5 folds, mean +- sd over folds)\n")
    print("| id | what changed | task | acc | brain-shuffle null | margin | p | "
          "zeros-brain | flip | emb cos (centred) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        if name not in cfgs:
            continue
        m = cfgs[name]["mean"].get(sel)
        if not m:
            continue
        d, task = DESC.get(name, ("", ""))
        print(f"| **{name}** | {d} | {task} (chance {m['chance'][0]:.3f}) | "
              f"{fmt(m['acc'][0])} ± {m['acc'][1]:.3f} | {fmt(m['null_mean'][0])} | "
              f"**{fmt(m['margin'][0], signed=True)}** | {m['p_perm'][0]:.3f} | "
              f"{fmt(m['zeros_acc'][0])} | {m['flip_rate'][0]:.3f} | "
              f"{m['emb_cos_centered'][0]:.3f} |")

    print("\n### Stratified nulls (only for runs that recorded them)\n")
    print("| id | acc | global null | margin | same-position null | margin | "
          "same-trial null | margin |")
    print("|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        m = cfgs.get(name, {}).get("mean", {}).get(sel)
        if not m or "null_pos" not in m:
            continue
        row = [f"**{name}**", fmt(m["acc"][0]), fmt(m["null_mean"][0]),
               fmt(m["margin"][0], signed=True), fmt(m["null_pos"][0]),
               fmt(m["margin_pos"][0], signed=True)]
        if "null_trial" in m:
            row += [fmt(m["null_trial"][0]), f"**{fmt(m['margin_trial'][0], signed=True)}**"]
        else:
            row += ["—", "—"]
        print("| " + " | ".join(row) + " |")

    print("\n### Per-fold margin (selection = " + sel + ")\n")
    print("| id | " + " | ".join(f"fold {i}" for i in range(5)) + " |")
    print("|---" * 6 + "|")
    for name in ORDER:
        if name not in cfgs:
            continue
        row = []
        for i in range(5):
            f = cfgs[name]["folds"].get(str(i)) or cfgs[name]["folds"].get(i)
            row.append(fmt(f[sel]["margin"], 3, signed=True) if f else "—")
        print(f"| {name} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
