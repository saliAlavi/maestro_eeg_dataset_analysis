# MERIT — Making Every Modality Earn Its Contribution

*What it is, why it exists, and which of our own mistakes produced each piece of it.*

## The one-sentence version

MERIT is a multimodal attended-talker decoder whose training objective and
model-selection criterion are built around the **permutation test**, and which
reports, next to accuracy, an **exactly additive decomposition of that accuracy
into per-stream credit** — so "the network uses EEG" is a measured quantity with
a p-value rather than an assumption.

## The problem it was built for

Every decoder in this project reports accuracy. Accuracy is not evidence about
*which input* produced it. Two failure modes, both of which we hit, are invisible
in an accuracy number and survive every standard held-out protocol:

1. **The model never reads the recording at all.** In the match–mismatch
   formulation the decoder sees the candidate audio directly. If any property of
   the audio marks the target, the task is solvable without physiology. On this
   corpus it is: target and masker recordings differ in the *distributional
   shape* of their envelope (crest factor 11.1 dB vs 20.5 dB), and every one of
   those statistics is affine-invariant, so it passes untouched through the
   per-candidate standardisation that is standard practice. A logistic
   regression on eight scale-free shape statistics names the attended talker
   56 % of the time on content-disjoint folds against 25 % chance — more than
   any decoder in this project extracts from EEG. See
   `analysis/n_gh_checks/FIXED_MODEL.md` §1.2, §5.1.
2. **The model reads exactly one strong stream and leaves the rest dead.** Add
   gaze or egocentric video to a decoder and accuracy rises, because in a
   free-field room listeners look at whom they attend. Nothing forces the EEG
   pathway to stay alive, and a late-fusion weight fitted on accuracy actively
   kills it: in `docs/draft/attended_speaker_decoding.tex` the vision branch
   takes the largest fusion weight (0.384) and the three EEG content branches
   between them take almost none.

## Mistakes that produced each component

| Component | The mistake it fixes |
|---|---|
| Time-centred **correlation coupling head** | The upstream scoring function — cosine of un-centred embeddings, time-averaged, then a linear read-out — *contains a working audio-only classifier as a special case*: make the physiological embedding constant in time and the score becomes a linear function of the time-averaged audio embedding. Gradient descent finds that solution. Measured: contribution `+0.0009`, decision-flip rate `0.008`, centred-embedding collapse `0.78`. Centring makes the constant-embedding solution score exactly zero on every candidate — unreachable, not merely discouraged. |
| **Certified candidate sets** (`qmatch`, `shifted_qm`) + audio-only probe | We first tried fixing the *encoder* (receptive field, GroupNorm, direction). Accuracy rose from 0.477 to 0.550 and the contribution stayed at exactly `+0.0000` — a better-conditioned encoder simply reads the shortcut better. The confound is in the data, and per-candidate rescaling cannot remove it; only matching the full marginal, or using same-talker negatives, can. |
| **Audio-only adversary** (gradient reversal) | The probe certifies against a *linear* reader. At short windows the trained model's own audio encoder finds residual structure the probe misses (null 0.325 vs probe 0.263 at 5 s). The adversary attacks what the probe cannot see. |
| **Audio-free orientation heads** | Our first multimodal attempt pushed gaze, IMU and video through the same envelope-matching head as EEG. They bear no temporal relationship to a speech envelope, so the problem is ill-posed and they collapsed. Each behavioural stream now gets the task it can actually do — classify the loudspeaker — and, taking no audio, its permutation null is at chance by construction. |
| **Per-stream contribution hinges** (tested; **off** in MERIT) | Modality dropout plus a per-stream auxiliary loss keeps the branches *trained*, but nothing makes the fused decision *depend* on them. The hinge writes the per-stream permutation test into the loss: permuting stream *m* across the batch must cost the correct slot a margin of score. Weights self-gate on inner-validation contribution, so a stream already contributing is not pushed further and the term cannot manufacture a contribution that is not in the data. |
| **Contribution-based checkpoint selection** | Selecting on validation accuracy selects the epoch that exploits the shortcut best: `+0.0929` contribution under accuracy selection against `+0.1689` under contribution selection, same run. Once candidates are confound-free the two criteria agree within 0.005, so it is a safeguard, not the source of the effect. |
| **Frozen V-JEPA 2 fovea stream** | Sixteen listeners cannot fit a video model; one trained here memorises the room. A frozen encoder over a gaze-centred crop is both stronger and cheaper than the 4-channel optical-flow summary we used first (0.662 vs 0.431 single-stream). The full-scene embedding is near-constant across trials — the camera barely moves — so it is noise in the branch and is used only as an alignment target. |
| **Shapley credit** | Per-stream contributions do not add up to the total: streams overlap (gaze and fovea are near-duplicates of the same orienting cue) and interact. The Shapley value over the `2^M` coalitions of intact streams is the unique decomposition that *is* additive, and with cached descriptors it costs less than one training epoch. |

## What the experiments changed (2026-09-21)

* **MERIT trains without the per-stream contribution hinge and without modality
  dropout** (`w_contrib: 0`, `modality_dropout: 0`; arm `H4_nohinge_nodrop` in
  `configs/runner/merit_hinge2.yaml`). In a clean 2x2 the hinge never raised the
  EEG's Shapley credit above the matching no-hinge arm, and with dropout it
  lowered it (0.223 -> 0.160). Dropout makes the validation contribution the gate
  reads noisy, so the hinge pushes whichever stream looks idle -- rarely the EEG.
  H4 has the most EEG credit on unseen listeners and ties for most within.
  The EEG-only null hinge on the coupling score (`w_null`) is separate and stays on.
* **The checkpoint rule is the lever that works**: selecting on
  `contribution + min_m phi_m` instead of accuracy raises EEG credit.
* **Per-trial orienting descriptors were never used.** `data.orient_feats=true`
  only builds them when `orient` is also in `data.static_mods`; the v2 runs put it
  only in the model's `static_mods`, and the model silently skips a declared
  stream that is absent from the batch. The runner now refuses to start in that
  case, and the v2 configs were edited to state what actually ran.
* On two public corpora (KU Leuven, DTU) the coupling branch transfers and survives
  same-talker candidates, but the lag-band control does not discriminate
  direction there, so those accuracies are not counted as neural decoding.

## Architecture

```
EEG (T×32) ─ dilated enc ─┬─ e (T×16) ── corr_t(e, â_k) ─────────► K coupling scores
                          └─ [mean;std] ─┐
gaze (T×6) ─ dilated enc ── [mean;std] ──┤
IMU  (T×6) ─ dilated enc ── [mean;std] ──┼─ per-stream speaker heads (NO audio in)
flow (T×4) ─ dilated enc ── [mean;std] ──┤   + gated sum + fusion head
fovea (1024, frozen V-JEPA 2) ─ proj ────┘        │
                                                  └─ gather by π ─► K slot scores
candidates a_1..a_K ─ shared audio enc ─► â_k ─► audio-only adversary (GRL)
```

`embed()` and `score()` are deliberately separate: a trained model can be
re-scored under any per-stream permutation without re-encoding, which is what
makes contribution and Shapley cheap enough to report on every run.

## What the numbers mean

* **accuracy** — how often the arg-max is right. Uninterpretable alone.
* **null** — accuracy with every stream permuted across test windows, each window
  keeping its own candidates and label. This is the audio-only floor *of this
  model*, not of a probe.
* **contribution** = accuracy − null. The part attributable to the recording.
* **contrib_*m*** — the same with only stream *m* permuted.
* **shapley_*m*** — per-stream credit; `Σ_m shapley_m = contribution` exactly.
* **lomo_*m*** — accuracy drop when stream *m* is zeroed instead of permuted.

A result is reported as `(task, split, window, candidate construction,
audio-only probe)`. An accuracy without its own null is not a result.

## Running it

```bash
python -m src.main data=merit model=merit runner=merit            # within-subject
python -m src.main data=merit model=merit runner=merit runner.protocols=[loso]
# orienting-free (same-talker negatives, exact 1/K floor):
python -m src.main data=merit model=merit runner=merit data.cand_mode=shifted_qm data.K=2
# lag-band control:
python -m src.main ... model.brain_dir=future model.lag_samples=8     # causal band
python -m src.main ... model.brain_dir=past   model.lag_samples=-8    # acausal band
```

## Known limits

* Distribution matching leaves `+0.010` on the four-way audio-only probe; only
  the same-talker construction is confound-free to three decimals.
* The orientation streams measure **overt orienting**, not covert attention. They
  are a real and useful property of the corpus but a different claim from
  stimulus tracking, and are always reported separately.
* Self-gating hinge weights are a heuristic. They are ablated against uniform
  weights (`model.w_contrib=0` and fixed-weight variants) precisely because a
  term that pushes a stream toward being used is the kind of thing that could
  manufacture an effect; the test-partition permutation null is the independent
  check, and it is never used for training or selection.
