"""End-to-end smoke test for maestro-loader against a local build."""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import numpy as np
from maestro_loader import load_aad, get_dataloaders, make_split, assert_trial_disjoint

ROOT = "/fs/scratch/PAS2301/alialavi/maestro-build-test"


def banner(t):
    print(f"\n{'='*8} {t} {'='*8}")


banner("1. aligned-raw, all modalities, native rates")
ds = load_aad(local_path=ROOT, subjects=[1], trials="main", modalities="all",
              segment_length=5.0, overlap=0.5, preprocess=False)
print("n_segments:", len(ds))
s = ds[0]
for k in ("eeg", "gaze", "imu", "audio"):
    print(f"  {k:6s} shape={np.asarray(s[k]).shape} sfreq={s[k+'_sfreq']:.1f}")
print("  video_path:", Path(s["video_path"]).name, "frame_range:", s["video_frame_range"])
print("  labels: attended=%s hemi=%s inout=%s" % (
    s["attended_speaker"], s["hemisphere"], s["inout"]))
assert s["eeg"].shape[0] == 32
assert s["audio"].shape[0] == 6  # 6 speakers

banner("2. preprocess=True, common 64 Hz grid, mel audio")
ds2 = load_aad(local_path=ROOT, subjects=[1], modalities=["eeg", "gaze", "imu", "audio"],
               segment_length=5.0, overlap=0.0, preprocess=True)
s2 = ds2[0]
for k in ("eeg", "gaze", "imu", "audio"):
    print(f"  {k:6s} shape={np.asarray(s2[k]).shape} sfreq={s2[k+'_sfreq']:.1f}")
T = s2["eeg"].shape[-1]
assert T == 320, f"expected 320 samples @64Hz*5s, got {T}"
assert s2["audio"].shape == (6, 28, 320), s2["audio"].shape
# all streams share the time axis length -> perfectly aligned
assert s2["gaze"].shape[-1] == T and s2["imu"].shape[-1] == T
print("  -> all modalities length", T, "(perfectly aligned)")

banner("3. LOSO split (subjects 1,2) -- trial-disjoint")
tr, te = load_aad(local_path=ROOT, subjects=[1, 2], modalities=["eeg"],
                  segment_length=5.0, preprocess=True,
                  split={"setting": "loso", "fold": 1})  # fold1 -> test S02
print("  train segments:", len(tr), "test segments:", len(te))
tr_sub = {u[0] for u in tr.units}
te_sub = {u[0] for u in te.units}
print("  train subjects:", tr_sub, "test subjects:", te_sub)
assert te_sub == {"S02"} and tr_sub == {"S01"}
assert_trial_disjoint(tr.units, te.units)
print("  trial-disjoint OK")

banner("4. intra split (subject 1) -- trial-disjoint, 3 folds")
tr2, te2 = load_aad(local_path=ROOT, subjects=[1], modalities=["eeg"],
                    segment_length=5.0,
                    split={"setting": "intra", "fold": 0, "n_folds": 3})
print("  train units:", len(tr2.units), "test units:", len(te2.units))
print("  train trials:", sorted(u[1] for u in tr2.units))
print("  test  trials:", sorted(u[1] for u in te2.units))
assert_trial_disjoint(tr2.units, te2.units)
inter = {u[1] for u in tr2.units} & {u[1] for u in te2.units}
assert not inter, f"trial overlap! {inter}"
print("  trial-disjoint OK (no shared trial_ids)")

banner("5. get_dataloaders (torch) -- intra")
train_dl, test_dl = get_dataloaders(
    local_path=ROOT, subjects=[1], modalities=["eeg", "audio"],
    setting="intra", fold=0, n_folds=3, batch_size=4, preprocess=True)
batch = next(iter(train_dl))
print("  batch eeg:", tuple(batch["eeg"].shape), "audio:", tuple(batch["audio"].shape))
print("  attended labels:", batch["attended_speaker"][:4].tolist())

banner("6. video frame decode")
dsv = load_aad(local_path=ROOT, subjects=[1], modalities=["video"],
               segment_length=2.0, video_frames=True, video_max_frames=4)
sv = dsv[0]
print("  decoded frames:", np.asarray(sv["video"]).shape)
assert sv["video"].ndim == 4 and sv["video"].shape[-1] == 3

print("\nALL CHECKS PASSED ✅")
