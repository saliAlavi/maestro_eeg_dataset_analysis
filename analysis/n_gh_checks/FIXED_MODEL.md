# A Leakage-Controlled Decoder for Attended-Talker Identification from EEG and Behavioural Signals

## Abstract

We address the identification of which of several simultaneously speaking talkers a listener is
attending to, from concurrently recorded EEG, eye tracking, head motion and egocentric video.
The task is conventionally posed as a matching problem: a decoder is shown a segment of the
listener's recording together with `K` candidate acoustic signals and must select the attended
one. We show that this formulation, in the form in which it is normally implemented, admits an
exact solution that never consults the physiological recording, and that such a solution is what
gradient descent finds. Two mechanisms are responsible. First, the candidate set is confounded:
target and masker recordings differ in the *distributional shape* of their amplitude envelope,
and because that difference is invariant to affine rescaling it passes untouched through the
per-signal amplitude normalisation that is standard practice — a logistic regression on eight
scale-free shape statistics identifies the attended talker 56 % of the time on content-disjoint
folds against a 25 % chance level. We show that no per-candidate affine normalisation can remove
this, and that matching the full marginal distribution can. Second, the standard scoring
function, a time-averaged cosine similarity followed by a linear read-out, reduces to a linear
classifier on the audio alone whenever the physiological embedding is constant over time; the
model is therefore free to emit a constant embedding, and does.

We present a decoder built around three properties: candidate sets whose audio-only Bayes
accuracy is at chance; a scoring function for which a time-constant physiological embedding is
arithmetically forced to chance; and a training objective containing the permutation test as a
loss term, so that a decoder ignoring its physiological input cannot reach a low training loss.
On sixteen listeners with content-disjoint splits, the decoder attains 0.471 four-way accuracy
with an audio-only floor of 0.277, so that 0.194 of the accuracy is attributable to the
recording. The effect holds under leave-one-listener-out evaluation in 16 of 16 folds, grows
monotonically with decision-window length, survives a within-trial permutation, and is present
only when the model reads EEG *after* the audio it scores (contribution 0.228) and absent when
restricted to reading it before (0.018) — the temporal asymmetry of an evoked response, not of
a recording artifact. Assigning each behavioural modality a task it can perform raises four-way
accuracy to 0.596 with every modality's permutation null at chance.

---

## 1. Introduction

### 1.1 The task

In a multi-talker environment a listener attends selectively to one voice. **Auditory attention
decoding** (AAD) recovers that choice from a physiological recording. The dominant formulation
is a matching problem: given a window of the recording and `K` candidate acoustic signals of
which exactly one corresponds to the attended talker, select it. Accuracy against a `1/K` chance
level is the reported metric.

The candidate acoustic signal is typically the **speech envelope**, the slowly varying amplitude
contour of a talker's speech, because cortical activity tracks it: the response to a change in
envelope amplitude appears roughly 100–300 ms afterwards.

### 1.2 Two failure modes of the standard formulation

**Acoustic confounding of the candidate set.** The decoder observes the candidate audio
directly. If any property of the audio itself indicates which candidate is the target, the task
can be solved without the recording, and accuracy will nonetheless resemble successful attention
decoding. The usual safeguards do not detect this: holding out listeners does not, because the
cue is present for every listener; holding out stimulus content does not, because the cue is a
property of the target *role* rather than of any particular recording; and reporting a chance
level does not, because the cue lifts accuracy well above it.

It is tempting to assume that amplitude normalisation disposes of this, since the most visible
difference between target and masker is presentation level. It does not, and the reason is worth
stating precisely, because it determines what a correct remedy looks like.

*Level differences are already removed, completely.* The envelope pipeline of §2.2 is composed
entirely of linear operations, so `env(a·x) = a·env(x)`, and per-signal standardisation
`x ↦ (x − μ)/σ` is invariant to any affine map. Scaling a recording by ±15 dB and passing it
through the pipeline therefore yields a bit-identical standardised envelope; we verify this
directly (§5.1), the residual being 1.6 × 10⁻⁵, i.e. floating-point rounding. **A pure level
difference cannot survive standardisation, and consequently cannot be the confound.**

*What survives is everything affine-invariant.* Standardisation fixes exactly two numbers per
candidate, its mean and its variance. Every statistic invariant under `x ↦ ax + b` passes
through untouched: standardised moments such as skewness and kurtosis, quantile ratios,
sparsity measures computed on the standardised signal, and relative spectral band powers. If
target and masker recordings differ in any of these, no per-candidate rescaling will help.

*They do differ.* Measured on the standardised envelopes (§5.1), targets have lower kurtosis,
lower skew and lower temporal sparsity, and a wider inter-quantile range, than maskers. Level and
shape turn out to be two measurements of a *single* underlying difference rather than independent
cues: regressing each shape statistic on per-file level reduces the target/masker separation from
|*d*| = 0.44–0.83 to 0.03–0.18 (§5.1). The evidence indicates the difference is one of dynamics
rather than of gain — crest factor, which no gain can alter, differs by 9.4 dB — and is
consistent with target recordings having been normalised toward full scale with limiting while
maskers were not. This is an inference from the rendered material, not from the generation
procedure, and the design does not permit it to be tested causally, since every recording appears
in one role only.

The remedy therefore cannot be a better rescaling. It must equalise the candidates' full
marginal distributions, or remove the distinction between target and masker recordings
altogether (§2.4).

**A scoring function with a degenerate optimum.** The standard architecture for this task is a
pair of dilated temporal convolution stacks, one for the recording and one shared across
candidates, whose outputs are compared by normalising each at every time point, multiplying
element-wise, averaging over time, and passing the result through a linear layer to a scalar per
candidate. Let the physiological embedding be the same unit vector `ĉ` at every time step. The
time average then factors, and the score for candidate `k` becomes `w · (ĉ ⊙ mean_t â_k) + b` —
a linear classifier on the time-averaged audio embedding, with exactly the capacity needed to
read the shape signature above. The architecture therefore contains a functioning audio-only
decoder as a special case, reachable by emitting a constant embedding, and that optimum is
easier to find than the intended one.

The two mechanisms compound: the first makes the audio-only solution *sufficient*, the second
makes it *reachable*.

### 1.3 Detecting the failure

Neither mechanism is visible in an accuracy number. Both are visible under a **permutation of
the physiological input**: the recordings are permuted across test windows while each window
retains its own candidates and its own label. A decoder using the recording loses its accuracy;
a decoder relying on the candidates alone does not. This must be a *feature* permutation, not a
label permutation — permuting labels would also destroy the audio–label relationship and would
conceal precisely the shortcut being tested for.

### 1.4 Contributions

1. Two candidate constructions whose audio-only Bayes accuracy is at chance, one exactly so,
   together with a model-independent probe that certifies this before any model is trained.
2. A scoring function for which a time-constant physiological embedding is arithmetically forced
   to chance, so the degenerate optimum is removed rather than discouraged.
3. A training objective containing the permutation test as a loss term, and a checkpoint
   criterion based on it; we show that selecting on accuracy instead actively selects the
   shortcut where one exists.
4. A control battery — permutation nulls global, position-stratified and trial-stratified; a
   zeros ablation; a lag-band directionality test — sufficient to establish that the remaining
   effect is a causally directed neural response rather than a recording artifact.
5. An architecture assigning each modality a task it can perform, raising four-way accuracy from
   0.471 to 0.596 while every modality's permutation null remains at chance.

---

## 2. Task and data

### 2.1 Paradigm

Sixteen listeners each completed one hundred trials. In each trial four talkers spoke
simultaneously from four loudspeakers at fixed, known azimuths; the listener was instructed
before the trial which loudspeaker to attend and answered a comprehension question afterwards.
Trials last approximately 30 s. The label is the attended loudspeaker index
`y_spk ∈ {1,2,3,4}`, balanced across the dataset at 400 trials per class.

The four attendable talkers of a trial are distinct recordings. Across the hundred trials there
are 400 distinct voices with no reuse (mean 1.00 occurrences per voice), so no train/test audio
overlap exists to be exploited. The attended talker is presented approximately 15 dB above its
competitors, in every trial.

Alongside EEG, the listener's gaze, head motion and egocentric scene video were recorded.

### 2.2 Signals and preprocessing

All signals are brought to a common 64 Hz grid and standardised per channel over the trial.

| Signal | Channels | Processing |
|---|---|---|
| EEG | 32 | 60 Hz notch (Q = 30); zero-phase Butterworth band-pass 1–40 Hz, order 4; automatic bad-channel detection; mastoid reference where usable, otherwise average reference; bad-channel interpolation; per-channel standardisation; resampling to 64 Hz |
| Gaze | 6 | horizontal and vertical gaze in the scene image, three-dimensional gaze direction, pupil diameter; linear interpolation onto the 64 Hz grid; 10 Hz zero-phase low-pass; standardisation |
| Head motion | 6 | three accelerometer and three gyroscope axes; interpolation; resampling to 64 Hz; 20 Hz low-pass; standardisation |
| Scene video | 4 | frames reduced to 160 × 90 greyscale; dense optical flow between consecutive frames; per-frame summary [mean flow magnitude, standard deviation of flow magnitude, mean horizontal flow, mean vertical flow]; resampling to 64 Hz; standardisation |
| Speech envelope | 1 per talker | analytic-signal magnitude; 20 Hz zero-phase low-pass, order 4; resampling to 64 Hz; standardisation |

### 2.3 The decision problem

Signals are divided into windows of `W` seconds with hop `H`; unless stated otherwise
`W = 10 s`, `H = 5 s`, giving `T = 640` samples per window, five windows per trial and 8 000
windows in total (7 990 when all four modalities are required, two trials lacking gaze and head
motion).

One example comprises physiological signals `x^m ∈ R^{T×C_m}` for
`m ∈ {eeg, gaze, imu, video}` with `C_eeg = 32`, `C_gaze = C_imu = 6`, `C_video = 4`;
`K` candidate envelopes `a_1 … a_K ∈ R^{T×1}` of which exactly one is correct; the index
`y ∈ {1..K}` of that candidate; the attended loudspeaker index `y_spk`; the permutation `π`
recording which loudspeaker occupies which candidate slot; and the listener identity `s`.

Candidates are shuffled into slots independently for every window, so slot index carries no
information. At evaluation the shuffle is seeded by the window's global index and is therefore
identical across all configurations compared. The decoder emits `K` scores; accuracy is the rate
at which the largest falls on `y`.

### 2.4 Candidate constructions

How the candidates are assembled determines what the task measures. We use four constructions.

**Competing talkers.** The `K = 4` candidates are the four co-present talkers, standardised.
This is the natural formulation and it is confounded, for the reason given in §1.2:
standardisation fixes each candidate's mean and variance and nothing else, so the
affine-invariant shape differences between target and masker recordings pass through intact.

**Distribution-matched competing talkers.** The same four talkers, with the marginal
distribution of every candidate forced to be identical. Let `r_k(t)` be the rank of `a_k(t)`
within candidate `k`, and `m = (1/K) Σ_k sort(a_k)` the average order statistics. Each candidate
is replaced by `a'_k(t) = m[r_k(t)]`, then standardised. Every candidate then has the same
multiset of values — hence identical kurtosis, skew, sparsity, dynamic range and silence
fraction — while retaining its own temporal ordering, which is the property a neural response
tracks.

**Same-talker temporal negatives.** The correct candidate is the attended talker's envelope for
this window; the `K − 1` incorrect candidates are the *same talker's* envelope drawn from other
windows of the same trial whose time spans do not overlap it (`|Δ position| ≥ ⌈W/H⌉`). The
candidates are exchangeable by construction and the audio-only Bayes accuracy is exactly `1/K`.
This is the match–mismatch formulation standard in the stimulus-reconstruction literature.

Two implementation details proved to matter. Negatives must be drawn *uniformly at random* among
the admissible windows: selecting the temporally furthest window biases negatives toward trial
edges, whose onset and offset statistics are distinctive, and this alone raised the audio-only
probe to 0.60 on a chance-0.50 task. And a negative must never overlap the positive in time;
since `H < W` an overlapping negative would be partly correct, so we reduce `K` rather than
permit one. A 30 s trial at `W = 10, H = 5` yields five windows, supporting at most two
non-overlapping negatives, hence `K = 3` and `K = 2`.

**Same-talker temporal negatives, distribution-matched.** Both operations applied; the only
construction whose audio-only probe is at chance to three decimal places.

---

## 3. Method

### 3.1 Architecture overview

The decoder has two branches answering two different questions.

```
                    ┌──────────────────────────────────────────┐
   EEG  (T×32) ────►│ encoder ──► e_eeg (T×16) ──┐             │
                    └────────────────────────────┼─────────────┘
                                                 │
                        ┌────────────────────────▼──────────────┐
                        │ COUPLING BRANCH                       │
   candidate a_1 ──► encoder ──► â_1 (T×16) ──►  correlate with │──► K scores
   candidate a_K ──► encoder ──► â_K (T×16) ──►  e_eeg over time│
                        └───────────────────────────────────────┘
                                                                    (+)
                    ┌──────────────────────────────────────────┐     │
   gaze  (T×6)  ───►│ encoder ──► e_gaze  ──┐                  │     │
   head  (T×6)  ───►│ encoder ──► e_imu   ──┼──► ORIENTATION   │─────┘
   video (T×4)  ───►│ encoder ──► e_video ──┘    BRANCH        │──► K scores
   EEG          ───►│ (shared with above) ──┘   (no audio in)  │
                    └──────────────────────────────────────────┘
```

The **coupling branch** asks whether the recording's time course matches a candidate's time
course. Only EEG enters it, because only EEG bears a temporal relationship to a speech envelope.
The **orientation branch** asks which loudspeaker the listener was oriented toward. It receives
no audio, which makes it immune to acoustic confounding by construction.

### 3.2 Components

Both branches begin with the same encoder; we define its parts.

**One-dimensional convolution over time** with kernel size 3 computes each output sample from
three neighbouring input samples using weights shared across time positions. Stacking such
layers widens the span of input influencing one output — the **receptive field** — but only
linearly.

**Dilation** inserts gaps between the sampled positions: a convolution with dilation `d` and
kernel 3 reads positions `t − d`, `t`, `t + d` instead of three adjacent samples. Each layer with
dilation `d` widens the receptive field by `d · (k − 1)`, so for `L` layers

```
RF = 1 + (k − 1) · Σ_i d_i
```

Doubling the dilation each layer — `1, 2, 4, 8, 16` — gives `RF = 1 + 2(2⁵ − 1) = 63` samples from
only five layers: the span grows geometrically while the cost grows linearly. At 64 Hz that is
0.98 s, matching the timescale over which the cortical response to a speech envelope unfolds
(0–400 ms).

Matching the receptive field to that timescale is a deliberate constraint rather than a default.
A receptive field much larger than the analysis window is not merely wasteful: the deeper layers
then convolve mostly zero padding, and since the padding is identical in every window, their
output stops depending on the input at all. §7.4 quantifies how badly this can go — with
dilations tripling rather than doubling over seven layers, the receptive field reaches 34 s
against a 10 s window, and 85 % of what the deepest layer sees is padding.

**Padding and direction.** Padding symmetrically and keeping the full output gives a *centred*
receptive field, in which the output at `t` depends on inputs either side of `t`. Padding on one
side and discarding the corresponding tail gives a *directional* one, `[t − RF, t]` or
`[t, t + RF]`. We use the centred form, because the response to audio at `t` occurs at
`t + 100…300 ms` and a past-only encoder cannot see it. The directional forms are used in §5.6
to confine the model to a chosen lag band.

**Normalisation.** Group normalisation follows each convolution, rescaling each group of four
channels to zero mean and unit variance across the group and the time axis. Without it a stack
of rectified layers can drift into a regime where its output is nearly constant regardless of
input.

**Activation.** A rectifier (`ReLU`, which replaces negatives by zero) follows every layer except
the last. Omitting it on the last layer is not cosmetic. A rectified output has no negative
entries, so the dot product of any two embeddings is a sum of non-negative terms and their cosine
similarity is confined to `[0, 1]` — two embeddings can never be anti-correlated. Worse, if the
16 coordinates have mean `μ > 0` and standard deviation `σ`, the expected cosine between two
independent embeddings is roughly `1/(1 + (σ/μ)²)`, which approaches 1 whenever the coordinates
vary little relative to their mean. Since the coupling branch's score *is* a correlation, a final
rectifier would compress the very quantity being measured into a narrow band near 1.

### 3.3 The modality encoder

```
input x ∈ R^{T×C}, transposed to (C, T)

[EEG only]  1×1 convolution, C → 8
            (a learned linear recombination of the 32 electrodes into 8 virtual sensors;
             the standard spatial-filtering step in EEG decoding)

for i = 0 … 4:
    Conv1d(· → 16, kernel 3, dilation 2^i, padding 2^i)
    GroupNorm(4 groups, 16 channels)
    if i < 4:   ReLU,  Dropout(0.1)

transpose back to (T, 16),  then  Linear(16 → 16)
```

The output `e ∈ R^{T×16}` is an **embedding**: sixteen learned descriptors of the signal at each
time point. The final linear map projects every modality into a shared sixteen-dimensional
space so branches can be combined. The audio encoder is identical with `C = 1` and no
spatial-filtering step, with weights shared across the `K` candidates.

### 3.4 The coupling branch

This branch scores how well the EEG's moment-to-moment fluctuation matches a candidate's. Both
are `T × 16` tables — 16 descriptors at each of `T` time points — and we compare them one
descriptor at a time, using the ordinary correlation coefficient over the time axis.

For descriptor `d`, subtract each signal's own average over time, divide by its length, and take
the dot product:

```
ẽ      = (e − mean_t e) / ‖e − mean_t e‖₂                 (per descriptor)
ã_k    = (â_k − mean_t â_k) / ‖â_k − mean_t â_k‖₂
c_d(k) = Σ_t ẽ_{t,d} · ã_{k,t,d}                           ∈ [−1, 1]
```

That gives 16 correlations per candidate. A learned weight vector `w` (no bias term) combines
them into one number, scaled by a learned temperature `τ = e^θ` — initialised at 1/0.07, which
sets how decisively score differences translate into probabilities — and finally we subtract the
mean across candidates:

```
s_k     = τ · Σ_d w_d · c_d(k)
score_k = s_k − (1/K) Σ_j s_j
```

**Why this exact form.** The design requirement is that a decoder ignoring the EEG must be unable
to win, and the subtraction of `mean_t e` is what enforces it.

Consider what happens when an encoder gives up and emits the same 16 numbers at every time point
— the state it drifts into once it has learned to ignore its input. Its average over time is then
that same constant, so `e − mean_t e` is **exactly zero**. Zero correlates with nothing, so
`c_d(k) = 0` for every descriptor and every candidate, all `K` scores tie, and accuracy is pinned
at `1/K`.

This is the crucial property, and it is worth stating plainly: the useless solution is not
penalised, discouraged, or made unattractive. It is arithmetically unreachable. Nothing the
optimiser does can turn a time-constant embedding into a winning score. §7.3 shows the
conventional formulation, which lacks the centring, admits exactly that solution and converges to
it.

Two further properties follow from the same algebra. The correlation is unchanged if a candidate
is rescaled, `corr(e, αa + β) = corr(e, a)` for `α > 0`, so a candidate cannot be favoured merely
by being louder. And candidate-centring removes anything added equally to all candidates, which
is also why `w` carries no bias — a bias shifts every score by the same amount and can never
change which candidate wins.

### 3.5 The orientation branch

Gaze, head motion and scene motion indicate where a listener is oriented, which predicts the
attended loudspeaker directly. They bear no temporal relationship to a speech envelope, so
asking them to match one is not a well-posed problem. Each therefore receives a classifier over
loudspeaker index taking **no audio input**:

```
φ(e)   = [ mean_t e ; std_t e ] ∈ R^{32}
logits = Linear(32 → 32) → ReLU → Dropout(0.2) → Linear(32 → 4)
```

Mean-and-standard-deviation pooling summarises both average orientation and its variability.
Because audio never enters, no acoustic property can influence this branch and its permutation
null is at chance by construction: anything above chance is attributable to the recording. EEG
also receives such a branch, since attention produces lateralised cortical activity.

### 3.6 Fusion

Per-modality embeddings are concatenated along the feature axis and passed through a further
classifier of the same form, `Linear(2·16·M → 32) → ReLU → Dropout(0.2) → Linear(32 → 4)`.
During training each modality is independently withheld with probability 0.3, never all of them,
withheld modalities being zero-filled so the fusion layer always executes. This **modality
dropout**, with a per-modality auxiliary loss (§3.8), prevents the strongest branch from
absorbing the gradient and leaving the others untrained. Orientation scores are mapped into
candidate slots through `π` and added to the coupling scores:

```
score = score_coupling + gather(score_orientation, π)
```

### 3.7 Audio-only adversary

An auxiliary classifier attempts to identify the correct candidate from the audio embeddings
alone:

```
ψ(â_k) = [ mean_t â_k ; std_t â_k ] ∈ R^{32}  →  GRL(λ)  →  Linear(32→32) → ReLU → Linear(32→1)
```

`GRL` is a **gradient reversal layer**: the identity forward, multiplying the gradient by `−λ`
backward. The adversary trains normally to detect a shortcut while the audio encoder is trained
to defeat it, so any residual acoustic cue the adversary can read is removed from the audio
embedding.

### 3.8 Training objective

The permutation test used to *detect* a decoder that ignores its physiological input is written
into the loss, so that such a decoder cannot reach a low training loss. Let `a⁺` be the correct
candidate's embedding, `ρ` a cyclic shift of the batch by one, and `s(·,·)` the coupling score.

| Term | Definition | Weight | Purpose |
|---|---|---|---|
| Task | cross-entropy on the `K` scores, label smoothing 0.1 | 1.0 | the decision itself |
| Orientation auxiliary | mean over branches of cross-entropy on loudspeaker index | 0.3 | trains every branch individually |
| Contrastive | `S_ij = s(e_i, a⁺_j)` over the batch, restricted to same-listener pairs; `½[CE(S, I) + CE(Sᵀ, I)]` | 1.0 | a collapsed encoder makes `S` rank-one, pinning this term at chance |
| Permutation hinges | `softplus(s(e_{ρ(i)}, a⁺_i) − s(e_i, a⁺_i) + 0.5)`, and the same with a zero embedding | 0.5 | the permutation test and the zeros ablation, as losses |
| Anti-collapse | `mean relu(0.5 − std_t ẽ) + 0.04 · Σ_{d≠d'} C²_{dd'}/D`, `C` the covariance of time-centred embeddings | 0.1 | penalises the temporal variance that collapses |
| Adversarial | cross-entropy of the audio-only head through the reversal layer | 0.3 | removes residual acoustic cues |

The **contrastive term** is an InfoNCE objective along the physiological axis: for each window it
asks which recording belongs with which stimulus, the other windows in the batch serving as
negatives. Negatives are restricted to the *same listener*, because raw preprocessed EEG
identifies the listener with 0.90 accuracy in a sixteen-way test against 0.0625 chance — a
cross-listener batch is solved by listener identity alone and teaches nothing about attention.
To make the restriction cheap, every batch is drawn from a single listener.

### 3.9 Optimisation and model selection

AdamW; learning rate 1e-3; weight decay 1e-4; gradient-norm clipping at 1.0; batch size 32 from
one listener; learning rate halved after five epochs without improvement; at most 50 epochs;
early stopping after twelve. The retained checkpoint maximises

```
validation accuracy − mean validation accuracy under a permuted physiological input
```

rather than validation accuracy. Where a shortcut exists the accuracy criterion selects the
epoch exploiting it best; §5.2 quantifies this.

### 3.10 Model sizes

| Configuration | Parameters |
|---|---|
| Coupling branch only | 8 698 |
| One orientation branch | 4 964 – 5 420 |
| Orientation fusion, three behavioural modalities | 18 320 |
| Complete model, four modalities | 29 230 |

---

## 4. Evaluation methodology

### 4.1 Splits

**Content-disjoint, 5 folds.** Trials are partitioned by stimulus content, so no stimulus heard
during training reappears at test. All sixteen listeners appear on both sides.

**Leave-one-listener-out, 16 folds.** One listener held out per fold, with a global 20 % content
holdout layered on top, so the held-out listener is novel in identity *and* content.

An inner validation set is carved from the training partition for checkpoint selection —
content-disjoint in the first protocol, listener-disjoint in the second. The test partition is
evaluated exactly once per fold.

### 4.2 The permutation null

Physiological inputs are permuted across test windows while each window retains its own
candidates and label; twenty permutations. We report the **contribution**,
`accuracy − mean accuracy under permutation`, and the permutation p-value
`(#{null ≥ real} + 1)/(n + 1)`, whose floor at twenty permutations is 0.048.

### 4.3 Supporting controls

**Zeros ablation.** Zeros passed through the encoder in place of the recording.

**Decision-flip rate.** The fraction of windows whose selected candidate changes under
permutation; approximately zero for an inert decoder.

**Collapse measure.** Mean pairwise cosine similarity between the *time-centred,
unit-normalised* embeddings of different windows — the quantity the coupling score consumes.
A value near 1 means every window produces the same temporal pattern.

**Stratified permutations.** Permuting only within a stratum makes that stratum's information
useless. *Same position in trial* rules out a slow drift shared by recording and envelope.
*Same trial* is stricter: all windows of a trial share a listener, an attended talker and a
stimulus set, so a decoder merely recognising which trial it was viewing would survive a global
permutation but not a within-trial one.

**Lag-band control.** A neural response to audio at `t` occurs at `t + 100…300 ms`, never
before; an electrical artifact from stimulus playback would appear at zero lag and be symmetric
in time. Restricting the encoder's direction (§3.2) and shifting the EEG by eight samples
confines the model to `+125…+1109 ms` — the only band a neural response can occupy — or to
`−1109…−125 ms`, which no stimulus-evoked response can occupy.

**Audio-only acceptance probe.** Model-independent, run on the candidate arrays before any
training: a logistic classifier on eight scale-free shape statistics (kurtosis, skew, Gini
sparsity, dynamic range, silence fraction, three modulation-band ratios), content-disjoint
folds, per-window arg-max. A construction free of acoustic confounding must score `1/K`.

### 4.4 Reference linear decoder

A ridge-regularised backward model, the standard non-deep decoder for this task. The envelope is
reconstructed from lagged EEG,

```
ŝ(t) = Σ_{c,τ} g(c,τ) · r(c, t + τ),      τ ∈ [0, 390 ms]
```

with the regularisation weight chosen on inner validation, and candidates ranked by the Pearson
correlation between `ŝ` and each candidate. It is structurally immune to the acoustic shortcut
for the same reason the coupling branch is.

---

## 5. Results

Sixteen listeners; `W = 10 s`, `H = 5 s`; content-disjoint protocol with five folds unless
stated; one-time test evaluation per fold.

### 5.1 The confound, and certifying the candidate sets

**Level normalisation is not the issue.** Passing one recording through the envelope pipeline at
several gains and comparing the resulting standardised envelopes:

| Applied gain | max abs. difference vs unscaled |
|---|---|
| +3 dB | 1.5 × 10⁻⁵ |
| +15 dB | 1.6 × 10⁻⁵ |
| −15 dB | 1.6 × 10⁻⁵ |

The outputs are identical to floating-point precision, confirming §1.2: a pure level difference
is already removed exactly, so it cannot be what a decoder exploits.

**What differs is distributional shape.** Measured on the standardised envelopes of all 400
attendable recordings, separating targets from maskers:

| Statistic (standardised envelope) | AUC, target vs masker | Direction | Corr. with level |
|---|---|---|---|
| kurtosis | 0.190 | target lower | −0.27 |
| skew | 0.219 | target lower | −0.46 |
| Gini sparsity | 0.297 | target lower | −0.43 |
| inter-quantile range (p95 − p5) | 0.619 | target higher | +0.22 |
| 8–20 Hz relative power | 0.401 | target lower | −0.19 |

An AUC of 0.5 indicates no separation. Maskers are spikier — more near-silence punctuated by
peaks, giving heavy tails but a narrow bulk in units of their own standard deviation; targets are
fuller and more continuous. All five statistics are affine-invariant and therefore survive
standardisation intact.

**Level and shape are one factor.** Regressing each shape statistic on per-file level, across all
400 attendable recordings, and re-testing the target/masker separation:

| Shape statistic | Cohen *d*, raw | after removing level | after removing level, pause and dynamics |
|---|---|---|---|
| kurtosis | −0.44 | +0.12 | +0.10 |
| skew | −0.83 | +0.18 | +0.15 |
| p95 − p5 | +0.48 | +0.03 | −0.00 |

The separation collapses, so level and envelope shape are two readouts of a single underlying
difference between target and masker recordings, not independent cues. Standardisation removes
the level readout and leaves the shape readout intact.

**The difference is one of dynamics, not gain.** Crest factor — peak divided by RMS, which no
gain can alter — is 11.09 dB for targets and 20.47 dB for maskers (*d* = −3.18), and across all
400 recordings level and crest correlate at r = −0.908. Implied mean peak levels are −0.004 dBFS
for targets and −4.83 dBFS for maskers. A target crest near 11 dB is well below natural
conversational speech (15–20 dB), whereas the maskers' 20.5 dB is typical of unprocessed speech.
Pause statistics differ negligibly (|*d*| ≤ 0.18), so this is not a difference in speaking style,
and the maskers' noise floor is 15 dB *lower* relative to their median, ruling out a noise floor
filling their silent intervals. The pattern is consistent with target recordings having been
normalised toward full scale with limiting or compression while maskers were not. Because all 400
recordings are unique and none appears in both roles, this cannot be established causally — the
role is perfectly confounded with file identity — and we therefore report it as the mechanism the
evidence favours rather than as a demonstrated cause. It does not affect the remedy, which is
mechanism-agnostic.

**Certification of the candidate constructions.** A logistic classifier on the eight scale-free
statistics of §4.3, content-disjoint folds, per-window arg-max:

| Candidate construction | K | Audio-only probe | Chance | Excess |
|---|---|---|---|---|
| Competing talkers | 4 | 0.5597 | 0.2500 | **+0.3097** |
| Distribution-matched competing talkers | 4 | 0.2600 | 0.2500 | +0.0100 |
| Same-talker temporal negatives | 3 | 0.3804 | 0.3333 | +0.0470 |
| Same-talker temporal negatives, distribution-matched | 3 | 0.3571 | 0.3333 | +0.0238 |
| Same-talker temporal negatives, distribution-matched | 2 | **0.5002** | **0.5000** | **+0.0002** |

The natural construction concedes 0.31 of accuracy to a trivial linear probe — more than any
decoder in this study extracts from the EEG. Distribution matching, which equalises all of the
statistics above by construction rather than only the first two moments, removes almost all of
it; the binary same-talker construction, in which the candidates are the same recording,
removes it to three decimals.

### 5.2 Component ablation

Each row changes exactly one thing from the row above. *Contribution* is accuracy minus the
permutation null; *flip* is the decision-flip rate; *collapse* is the time-centred embedding
similarity.

| Configuration | Candidates | Accuracy | Null | Contribution | p | Zeros | Flip | Collapse |
|---|---|---|---|---|---|---|---|---|
| **Standard architecture** | competing talkers | 0.4769 ± 0.032 | 0.4760 | **+0.0009** | 0.743 | 0.4772 | 0.008 | 0.780 |
| **Standard architecture, alternative optimiser** | competing talkers | 0.4800 ± 0.054 | 0.4800 | **−0.0000** | 1.000 | 0.4800 | 0.000 | 0.888 |
| **+ redesigned encoder** | competing talkers | 0.5501 ± 0.092 | 0.5501 | **+0.0000** | 0.943 | 0.5501 | 0.000 | 0.778 |
| **+ correlation scoring** | competing talkers | 0.5415 ± 0.122 | 0.5304 | **+0.0111** | 0.210 | 0.5446 | 0.144 | 0.780 |
| **+ permutation-aware objective** | competing talkers | 0.5869 ± 0.035 | 0.4180 | **+0.1689** | 0.048 | 0.4806 | 0.519 | 0.502 |
| **Proposed model, four-way** | distribution-matched | 0.4709 ± 0.017 | **0.2772** | **+0.1937** | 0.048 | 0.3071 | 0.699 | 0.291 |
| Proposed model, three-way, no distribution matching | same-talker | 0.5584 ± 0.019 | 0.3322 | +0.2262 | 0.048 | 0.3340 | 0.662 | 0.173 |
| **Proposed model, three-way** | same-talker, matched | 0.5495 ± 0.024 | **0.3329** | **+0.2166** | 0.048 | 0.3476 | 0.662 | 0.217 |
| **Proposed model, binary** | same-talker, matched | 0.6974 ± 0.014 | **0.4996** | **+0.1978** | 0.048 | 0.5041 | 0.496 | 0.207 |

Chance is 0.250 for four-way rows, 0.333 for three-way, 0.500 for binary. Per fold, the
contribution is positive in 5 of 5 folds for every row from *permutation-aware objective*
downward (minimum +0.156) and in 0–2 of 5 folds above it.

Four observations.

1. **The standard architecture is inert.** Contribution ≈ 0, flip rate ≈ 0, zeros identical to
   real, collapse 0.78–0.89: the physiological input is not used at all, exactly as §1.2
   predicts.
2. **Fixing the encoder alone achieves nothing.** Accuracy *rises* to 0.550 while the
   contribution stays at exactly +0.0000 — a better-conditioned encoder reads the acoustic
   shortcut better. Receptive-field, normalisation and directionality are real design defects
   but are not what causes the failure.
3. **Correlation scoring is the enabling change.** It is the only difference from the row above
   and produces the first non-zero contribution. Its own effect is small because the confound is
   still in the data, but the permutation-aware objective cannot be expressed against the
   standard scoring function: a constant embedding there still yields candidate-dependent
   scores, so the hinges have a solution satisfying them while remaining audio-driven.
4. **The proposed four-way model reproduces the standard architecture's accuracy and
   reinterprets it.** 0.4709 against 0.4769, statistically indistinguishable, but the audio-only
   floor falls from 0.4760 to 0.2772, essentially chance: 0.194 attributable to the recording
   rather than 0.001.

**Selection criterion.** Both criteria were computed on every fold:

| Configuration | Selected on contribution | Selected on accuracy |
|---|---|---|
| + redesigned encoder | +0.0000 | +0.0000 |
| + correlation scoring | +0.0111 | +0.0002 |
| + permutation-aware objective | +0.1689 | +0.0929 |
| Proposed model, four-way | +0.1937 | +0.1910 |
| Proposed model, binary | +0.1978 | +0.1920 |

Where a shortcut exists, selecting on accuracy selects it. Once candidates are confound-free the
criteria agree within 0.005, so the criterion is a safeguard rather than a source of the effect.

### 5.3 Generalisation to unseen listeners

Leave-one-listener-out, sixteen folds, each with a held-out listener on held-out content.

| Configuration | Accuracy | Null | Contribution | Folds positive | Wilcoxon p |
|---|---|---|---|---|---|
| Standard architecture | 0.5025 ± 0.084 | 0.5039 | **−0.0014** | **0 / 16** | 0.068 |
| + correlation scoring | 0.4662 ± 0.089 | 0.4447 | +0.0215 | 11 / 16 | 1.8e−2 |
| **Proposed model, four-way** | 0.4919 ± 0.073 | 0.2834 | **+0.2085** | **16 / 16** | 4.4e−4 |
| **Proposed model, binary** | 0.6844 ± 0.072 | 0.4973 | **+0.1870** | **16 / 16** | 3.1e−5 |
| Orientation fusion | 0.4665 ± 0.152 | 0.2549 | +0.2116 | 14 / 16 | 4.3e−4 |
| **Complete multimodal model** | 0.5789 ± 0.117 | 0.2737 | **+0.3053** | **16 / 16** | — |

The effect transfers to unseen listeners undiminished: +0.209 against +0.194 within-listener.
The standard architecture is negative in every one of the sixteen folds.

### 5.4 Decision-window length

Non-overlapping windows unless noted; the 10 s / 5 s row is the headline configuration.

| Window | Hop | Four-way accuracy | Null | Contribution | Binary accuracy | Null | Contribution |
|---|---|---|---|---|---|---|---|
| 1 s | 1 s | 0.2937 ± 0.023 | 0.2931 | +0.0006 | 0.4986 ± 0.003 | 0.4981 | +0.0005 |
| 2 s | 2 s | 0.2989 ± 0.039 | 0.2906 | +0.0083 | 0.5247 ± 0.022 | 0.4976 | +0.0271 |
| 4 s | 4 s | 0.3762 ± 0.013 | 0.3155 | +0.0606 | 0.6115 ± 0.017 | 0.4980 | +0.1135 |
| 8 s | 8 s | 0.4385 ± 0.017 | 0.2881 | +0.1504 | 0.6681 ± 0.016 | 0.5001 | +0.1680 |
| 10 s | 5 s | 0.4871 ± 0.028 | 0.2901 | +0.1970 | 0.7002 ± 0.018 | 0.4995 | +0.2007 |
| 10 s | 10 s | 0.4856 ± 0.041 | 0.3023 | +0.1833 | 0.6552 ± 0.076 | 0.5018 | +0.1534 |
| 16 s | 16 s | 0.3519 ± 0.104 | 0.2598 | +0.0921 | — | — | — |
| 30 s | 30 s | 0.5969 ± 0.021 | 0.2654 | +0.3314 | — | — | — |

The contribution rises monotonically from chance at 1 s to a large effect by 8–10 s, the
canonical accuracy-versus-window-length relationship for this task. The binary construction is
undefined beyond 10 s, since a 30 s trial contains only one non-overlapping window at 16 s and
30 s and thus no admissible same-talker negative. The 16 s and 30 s points rest on one window
per trial (≈ 320 test, ≈ 1 000 training windows) with fold deviations up to 0.104 and should not
be over-interpreted. The two 10 s rows differ only in hop and agree to within 0.014.

### 5.5 Window specificity

| Configuration | Protocol | Accuracy | Global null → contribution | Same-position → contribution | Same-trial → contribution |
|---|---|---|---|---|---|
| Four-way | content-disjoint | 0.4871 | 0.2901 → +0.1970 | 0.2981 → +0.1890 | 0.3292 → **+0.1579** |
| Three-way | content-disjoint | 0.5566 | 0.3324 → +0.2243 | 0.3472 → +0.2094 | 0.3372 → **+0.2194** |
| Binary | content-disjoint | 0.7002 | 0.4995 → +0.2007 | 0.5157 → +0.1845 | 0.5026 → **+0.1977** |
| Four-way | LOSO | 0.4919 | 0.2813 → +0.2106 | 0.2827 → +0.2092 | 0.3226 → **+0.1693** |
| Binary | LOSO | 0.6875 | 0.4989 → +0.1886 | 0.5020 → +0.1855 | 0.4962 → **+0.1913** |

For the same-talker constructions the within-trial null is indistinguishable from the global
null, so the model matches the specific temporal segment rather than recognising the trial. For
the four-way construction the contribution falls by about a fifth, which is expected: with four
co-present talkers the attended talker is the same in every window of a trial, so trial-level
recognition legitimately carries part of the signal. Four fifths remains window-specific.

### 5.6 Neural directionality

| Configuration | EEG visible at | Accuracy | Null | Contribution | Collapse |
|---|---|---|---|---|---|
| **Binary, causal band** | **+125 … +1109 ms** | 0.7282 ± 0.057 | 0.5000 | **+0.2282** | 0.267 |
| **Binary, acausal band** | **−1109 … −125 ms** | 0.5246 ± 0.009 | 0.5067 | **+0.0179** | 0.595 |
| **Four-way, causal band** | +125 … +1109 ms | 0.4952 ± 0.019 | 0.2697 | **+0.2256** | 0.356 |
| **Four-way, acausal band** | −1109 … −125 ms | 0.3765 ± 0.081 | 0.3713 | **+0.0052** | 0.831 |

The coupling is thirteen times larger in the causal direction for the binary task and forty-three
times larger for the four-way task, and neither acausal configuration is significantly above its
own null (p = 0.181, 0.314). Zero-lag stimulus bleed and any time-symmetric shared drift are
excluded. Confining the model to the correct lag band also improves the result: 0.7282 against a
null of exactly 0.5000 is the strongest figure in the study.

### 5.7 Modality contributions

Orientation branches take no audio input, so their permutation null is at chance by construction.

| Model | Inputs | Accuracy (chance 0.250) | Null | Contribution |
|---|---|---|---|---|
| Orientation, EEG | EEG | 0.3678 ± 0.043 | 0.2515 | +0.1163 |
| Orientation, gaze | gaze | 0.3698 ± 0.015 | 0.2503 | +0.1195 |
| Orientation, head motion | head IMU | 0.3732 ± 0.038 | 0.2510 | +0.1222 |
| Orientation, scene video | video | 0.4307 ± 0.037 | 0.2499 | +0.1808 |
| **Orientation fusion** | gaze + head + video | **0.5001 ± 0.044** | 0.2501 | **+0.2500** |
| Coupling only | EEG + audio | 0.4709 ± 0.017 | 0.2772 | +0.1937 |
| **Complete multimodal model** | all four + audio | **0.5961 ± 0.048** | 0.2780 | **+0.3181** |

Fusion is genuinely additive: 0.500 for the three behavioural streams against 0.431 for the best
single one, and 0.596 once the coupling branch is added, against 0.471 for that branch alone.
The orientation branches measure **overt orienting**, not covert attention — a legitimate and
useful property of the dataset, but a different claim from stimulus tracking, and reported
separately. The orienting-free result is the binary coupling model.

### 5.8 Linear reference decoder

| Task | Protocol | Accuracy | Null | Contribution | Reconstruction r | Proposed model |
|---|---|---|---|---|---|---|
| Distribution-matched, four-way | content-disjoint | 0.2544 | 0.2470 | +0.0074 | 0.0036 | +0.1937 |
| Same-talker, binary | content-disjoint | 0.5155 | 0.4982 | +0.0173 | 0.0042 | +0.1978 |
| Distribution-matched, four-way | LOSO | 0.2419 | 0.2464 | −0.0046 | 0.0026 | +0.2085 |
| Same-talker, binary | LOSO | 0.5012 | 0.4920 | +0.0092 | 0.0032 | +0.1870 |

Restricting the EEG to 1–8 Hz, the conventional band for envelope tracking, does not help
(reconstruction r = 0.0035). The linear decoder is at chance — but so is its envelope
reconstruction, at r ≈ 0.004 against the 0.05–0.20 typically reported. This is a property of the
baseline as configured; see §6.

---

## 6. Discussion and limitations

**The linear reference is uninformative, and this is unresolved.** A reconstruction correlation
of 0.004 is one to two orders of magnitude below published values for this method, so the
comparison establishes only that the proposed decoder beats *this* baseline, not that it beats a
working one. The most likely explanation is alignment: a fixed 0–390 ms linear lag window is
intolerant of timing jitter that a learned ±0.5 s encoder can absorb. Until this is resolved, no
claim of the form *"the deep model outperforms linear decoding"* is supported.

**The four-way construction retains a small confound.** Distribution matching leaves +0.010 on
the audio-only probe; only the binary same-talker construction is confound-free to three
decimals.

**Part of the four-way contribution is trial-level.** About a fifth does not survive a
within-trial permutation (§5.5). The four-way figure should carry that qualification; the
same-talker figures should not.

**Orientation results are not covert attention.** See §5.7.

**Long-window estimates are unstable.** At 16 s and 30 s there is one window per trial,
approximately one thousand training windows, and fold deviations up to 0.104.

**No hyper-parameter search was performed.** Embedding width, loss weights, lag range and layer
count were each set once from first principles and never tuned; the reported figures are
untuned rather than best-case.

**Selection uses three validation permutations** against twenty at test. A noisier selection
signal can cost contribution but cannot inflate it.

**Coverage gaps.** The window sweep is content-disjoint only and covers the coupling
configurations; leave-one-listener-out and the multimodal configurations were evaluated at 10 s
only. Windows of 5, 15 and 20 s were not run.

**The confound's mechanism is inferred, not demonstrated.** The evidence in §5.1 favours a
dynamics-processing difference between target and masker recordings, but every recording appears
in exactly one role, so role and file identity are perfectly confounded and no causal test is
possible from the rendered material. Confirming it would require the stimulus-generation
procedure. This does not affect any result: the audio-only probe measures the cue's readability
whatever its origin, and both remedies are mechanism-agnostic. It does, however, bear on stimulus
design for future collection — a corpus in which each recording serves as target in some trials
and masker in others would remove the confound at source and make it testable.

---
---

# INTERNAL — not part of the manuscript

## 7. What was wrong with the old model, and what we changed

This section is for us. It walks through each problem, shows the old code and the new code side
by side, and explains why the change works. It assumes no background beyond "a convolution slides
a filter over a signal" — everything else is built up as we go.

Old code: `scripts/{model_classification,dataloader,train_aad}.py`.
New code: `analysis/n_gh_checks/fixed/`.

---

### 7.0 Vocabulary and notation

A few terms used throughout. Skip if they're familiar.

**Window.** We chop each 30-second trial into 10-second pieces. At 64 samples per second, one
window is `T = 640` numbers per channel.

**Embedding.** The encoder converts a window into a `640 × 16` table of numbers: at each of the
640 time points, 16 numbers describing what the signal is doing there. We call one row of that
table `b_t` (for the EEG) or `a_{k,t}` (for candidate `k`'s audio), and one column a *dimension*.
Think of the 16 dimensions as 16 learned detectors running in parallel.

**Candidate.** One of the `K` audio signals the model has to choose between. Exactly one is
correct.

**Score.** A single number per candidate. Highest score wins.

**The contribution.** Our headline metric. Take the trained model, shuffle the EEG recordings
across test windows so each window gets someone else's brain data — but keep each window's own
audio and its own correct answer. Then

```
contribution = accuracy with the real EEG  −  accuracy with shuffled EEG
```

If the model is genuinely using the EEG, shuffling it should hurt. If the contribution is zero,
the EEG was decorative. The old model's contribution was **+0.0009**.

**Notation for the walkthroughs.** `mean_t(x)` means "average over the 640 time points". `⟨u, v⟩`
means the dot product `Σ_d u_d v_d`. `u ⊙ v` means multiply element by element.

---

### 7.1 What the old model did

For one window: 640 time points of EEG across 32 electrodes, plus four candidate envelopes of
640 points each.

| Step | What happens | Where |
|---|---|---|
| 1 | The EEG goes through 7 stacked dilated convolution layers → a `640 × 16` embedding | `model_classification.py:34-61` |
| 2 | Each audio candidate goes through the same kind of 7-layer stack → its own `640 × 16` embedding | `:128-131` |
| 3 | EEG and candidate embeddings are compared, giving 16 similarity numbers | `:135-136` |
| 4 | Those 16 numbers go through a linear layer to one score per candidate; softmax | `:133, 170-176` |
| 5 | Train with cross-entropy on the 4-way choice. Keep the epoch with the best validation accuracy | `train_aad.py:66-69, 90, 115-116` |

Gaze, head motion and video were plugged into step 1 in place of EEG and otherwise treated
identically (`model_classification.py:149-168`).

There are five things wrong here. Two of them are fatal on their own.

---

### 7.2 Problem 1 — the four audio candidates are not interchangeable

#### The setup

In every trial the talker the listener was told to attend was played about 15 dB louder than the
other three. That is a deliberate design choice — it makes the target intelligible — but it means
the correct answer is marked in the audio itself. If a model can spot that mark, it can score
well without ever looking at the brain.

The old code appears to handle this:

```python
# dataloader.py, extract_envelope
if target_rms is not None:
    current_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    audio = audio * (target_rms / current_rms)      # 246  make all talkers equally loud
env = np.abs(hilbert(audio)).astype(np.float32)     # 248  extract the loudness contour
...
return _zscore(env)[:, np.newaxis]                  # 252  subtract mean, divide by std
```

#### Why line 246 does nothing (and why that's fine)

Line 246 scales every talker to the same loudness. Line 252 then subtracts the mean and divides
by the standard deviation. **The second operation undoes the first**, because every step between
them is linear.

Here is the argument in full. Write `E(·)` for the envelope extraction on line 248 and `Z(·)` for
the standardisation on line 252. Suppose we make a recording `α` times louder:

1. The Hilbert transform is linear, so `H(αx) = α·H(x)`, and since `α > 0`, `|H(αx)| = α·|H(x)|`.
2. Low-pass filtering and resampling are also linear, so they pass the `α` straight through.
3. Therefore `E(αx) = α·E(x)` — **scaling the input just scales the envelope**.
4. Now standardise. If `e` has mean `μ` and standard deviation `σ`, then `αe` has mean `αμ` and
   standard deviation `ασ`. So

   ```
   Z(αe) = (αe − αμ) / (ασ) = α(e − μ) / (ασ) = (e − μ)/σ = Z(e)
   ```

   The `α` cancels top and bottom.
5. Putting 3 and 4 together: `Z(E(αx)) = Z(E(x))`. **Identical.**

We checked this on a real recording. Scaling by +3 dB, +15 dB and −15 dB gives standardised
envelopes that differ by at most `1.6 × 10⁻⁵` — that is float32 rounding error, i.e. zero.

So line 246 is dead code. It is worth deleting because it *looks* like loudness is being handled,
but deleting it changes no result: line 252 was already doing the job, and doing it more
thoroughly.

#### The actual problem: loudness was never the cue

Here is the part that took us a while to see. Since a pure loudness difference is provably erased
by line 252, and a simple probe can *still* pick the attended talker 56 % of the time from the
standardised envelopes (chance is 25 %), **the cue cannot be loudness**. It has to be something
else about the target recordings that survives standardisation.

What survives? Standardisation pins down exactly two numbers per candidate: its mean and its
standard deviation. Everything else about the *shape* of the distribution is untouched. Formally,
any statistic `S` that doesn't change when you rescale — `S(αe + β) = S(e)` — passes straight
through, because standardisation *is* a rescale (`α = 1/σ`, `β = −μ/σ`).

That class is large:

| Survives standardisation | Why |
|---|---|
| skewness, kurtosis (how lopsided / how heavy-tailed) | defined as ratios to `σ³`, `σ⁴` — the scale cancels |
| p95 − p5, measured in units of `σ` | numerator and denominator both scale |
| fraction of samples below `−0.5σ` | threshold moves with the data |
| relative band powers, `P(4–8 Hz)/P(total)` | ratio of two things that scale together |

And these really do differ between targets and maskers:

| Statistic | AUC, target vs masker | Direction |
|---|---|---|
| kurtosis | 0.190 | targets lower |
| skew | 0.219 | targets lower |
| Gini sparsity | 0.297 | targets lower |
| p95 − p5 | 0.619 | targets higher |
| 8–20 Hz relative power | 0.401 | targets lower |

(AUC 0.5 = no difference. 0.19 and 0.62 are both far from 0.5.)

In words: **maskers are spikier.** Long quiet stretches punctuated by peaks, which gives heavy
tails (high kurtosis) but a narrow middle. Targets are fuller and more continuous. Section 7.2.1
digs into why.

The conclusion that matters: **no amount of clever rescaling can fix this.** The remedy has to
either equalise the whole distribution, or get rid of the target/masker distinction entirely. We
did both.

#### Fix A — make all four candidates have identical value distributions

`fixed/candidates_v2.py:42-58`:

```python
def quantile_match(A, chunk=1000):
    """A: (N,K,T) -> (N,K,T). Forces the K candidates of each window onto a
    common marginal (their average order statistics) while preserving each
    candidate's own temporal ordering."""
    N, K, T = A.shape
    out = np.empty_like(A, dtype=np.float32)
    ar = np.arange(T)
    for s in range(0, N, chunk):
        a = A[s:s + chunk]
        n = a.shape[0]
        order = np.argsort(a, axis=2, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(ar, (n, K, T)), axis=2)
        ref = np.sort(a, axis=2).mean(axis=1, keepdims=True)          # (n,1,T)
        matched = np.take_along_axis(np.broadcast_to(ref, (n, K, T)), ranks, axis=2)
        out[s:s + chunk] = _zscore_last(matched).astype(np.float32)
    return out
```

**What this does, on a toy example.** Say `T = 5` and two candidates:

```
a₁ = [3, 1, 9, 4, 2]        a₂ = [8, 7, 1, 2, 6]
```

Sort each and average position by position — that's `ref`, the shared "vocabulary" of values:

```
sort(a₁) = [1, 2, 3, 4, 9]
sort(a₂) = [1, 2, 6, 7, 8]
ref      = [1, 2, 4.5, 5.5, 8.5]     ← the average of the two sorted lists
```

Now rank each candidate's samples (0 = smallest) and replace each sample with the `ref` value at
that rank:

```
a₁ ranks = [2, 0, 4, 3, 1]   →   a₁' = [4.5, 1, 8.5, 5.5, 2]
a₂ ranks = [4, 3, 0, 1, 2]   →   a₂' = [8.5, 5.5, 1, 2, 4.5]
```

Look at what came out. `a₁'` and `a₂'` contain **exactly the same five numbers** —
`{1, 2, 4.5, 5.5, 8.5}` — just in different orders.

**Why that kills the cue.** Every statistic in the table above is computed from the *set* of
values, ignoring their order. Kurtosis, skew, Gini, dynamic range, silence fraction — you could
shuffle the samples in time and get the same answer. Since all `K` candidates now hold the
identical set of values, all of those statistics are identical too. Not approximately: exactly,
by construction.

What still differs is the **order**, which is precisely what a neural response tracks. We removed
the cue and kept the signal.

**Why a small leak remains (+0.010).** Three of the eight probe features are frequency-band
ratios, and those *do* depend on the order — that's the whole point of a frequency. So they
survive. Measured: 0.2600 against 0.2500 chance. This isn't a bug; any operation that also
flattened the spectra would destroy the signal we're trying to measure.

#### Fix B — make the wrong answers be the same talker

`fixed/candidates_v2.py:61-95`:

```python
def build_shifted_imposters(trial_ids, window_sec, hop_sec, n_neg=3, seed=0):
    """For every window, pick `n_neg` imposter windows from the same trial whose
    time spans do not overlap it.  Same voice, same recording, different segment."""
    gap = max(1, int(np.ceil(window_sec / hop_sec)))
    out = np.full((len(trial_ids), n_neg), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for t in np.unique(trial_ids):
        idx = np.where(trial_ids == t)[0]            # contiguous, temporal order
        n = len(idx)
        for p in range(n):
            valid = [q for q in range(n) if abs(q - p) >= gap]   # non-overlapping only
            rng.shuffle(valid)                                    # uniform draw
            ...
            out[idx[p]] = idx[np.asarray(pick)]
    return out, n_fallback
```

Instead of "which of these four talkers?", the question becomes "which of these `K` segments of
*the same talker* lines up with the EEG?" The candidates are now slices of one recording, so
there is no target-versus-masker difference left to detect — not a small one, none at all.

**Why the audio-only accuracy is exactly `1/K`.** Which segment is "correct" is defined purely by
whether it lines up in time with the EEG window. A model that can't see the EEG has no way to
tell — the candidates are interchangeable as far as it is concerned, so its best possible guess is
uniform. Measured: **0.5002** on a binary version where chance is 0.5000.

**Two details that turned out to matter.**

*Never let a negative overlap the positive.* Our hop is 5 s and our window is 10 s, so
consecutive windows share half their audio. A negative drawn from an overlapping window would be
half-correct, which both weakens the task and muddies the interpretation. The `abs(q - p) >= gap`
condition enforces disjointness. Since a 30-second trial only yields five windows, this caps us at
two usable negatives — hence `K = 3` and `K = 2` rather than 4.

*Draw negatives uniformly, not at maximum distance.* Our first version picked the temporally
furthest segment, reasoning that further apart is safer. That was wrong: it systematically picked
trial-start and trial-end segments, which have their own onset and offset character, and the probe
climbed to **0.60** on a chance-0.50 task. Shuffling the admissible set fixed it. Worth
remembering — "maximally different" is not the same as "unbiased".

#### Result

| Candidate construction | Probe | Chance |
|---|---|---|
| Old (four competing talkers) | 0.5597 | 0.2500 |
| Fix A (distribution-matched) | 0.2600 | 0.2500 |
| Fix B (same talker, binary) | 0.5002 | 0.5000 |

---

#### 7.2.1 Why do targets and maskers differ in shape? (`fixed/mechanism_trace.py`, job 6915522)

Descriptors over all 400 attendable recordings. Cohen's *d* is target minus masker in units of
pooled standard deviation; |d| > 0.8 is conventionally "large".

| Descriptor | Target | Masker | *d* |
|---|---|---|---|
| loudness (RMS) | −11.09 dB | −25.30 dB | +4.48 |
| **crest factor (peak ÷ RMS)** | **11.09 dB** | **20.47 dB** | **−3.18** |
| noise floor, relative to median | −97.3 dB | −112.7 dB | +0.17 |
| fraction silent | 0.107 | 0.123 | −0.17 |
| mean pause length | 0.072 s | 0.076 s | −0.06 |
| pauses per second | 1.309 | 1.461 | −0.18 |
| kurtosis | 0.128 | 3.611 | −0.44 |
| skew | 0.848 | 1.449 | −0.83 |

**The two groups are not related by a volume knob.** Crest factor is peak divided by RMS. Turn a
recording up and *both* go up by the same factor, so crest doesn't move — it is immune to volume
changes in exactly the way standardisation is. Yet targets and maskers differ by 9.4 dB. Whatever
made the targets louder also changed the shape of the waveform.

Supporting numbers: across all 400 files, loudness and crest correlate at **r = −0.908** (a pure
volume difference would give 0). The implied average peak level is **−0.004 dBFS for targets** —
that is, sitting exactly at digital full scale — versus −4.83 dBFS for maskers. Natural
conversational speech has a crest factor of 15–20 dB; the maskers' 20.5 dB is normal, the targets'
11 dB is what you get after compression or limiting.

**Two alternative explanations, both ruled out:**

- *Noise floor.* The idea: maskers were turned down, so their noise floor is relatively higher and
  fills in their quiet gaps. This predicts maskers should be *smoother* (lower kurtosis). They're
  28× spikier. And their noise floor is 15 dB *lower* relative to their own median, not higher.
- *Speaking style.* The idea: targets were simply spoken more continuously. All three pause
  descriptors are tiny (|d| ≤ 0.18) next to crest's 3.18. Targets and maskers pause about equally.

**Loudness and shape are one thing, not two.** If we statistically remove loudness from each shape
statistic and re-test the target/masker gap:

| Shape statistic | *d* as measured | after removing loudness | after removing loudness + pauses + dynamics |
|---|---|---|---|
| kurtosis | −0.44 | +0.12 | +0.10 |
| skew | −0.83 | +0.18 | +0.15 |
| p95 − p5 | +0.48 | +0.03 | −0.00 |

The gap essentially vanishes. So loudness and shape are two symptoms of a single underlying
difference, not two independent cues.

**How this squares with §7.2.** No contradiction, and the distinction is worth being clear about:

- *Within one recording*, standardisation removes volume exactly. Still true.
- *Across the 400 recordings*, loudness happens to travel with shape, because one processing step
  produced both.

Standardisation removes the loudness symptom and leaves the shape symptom. That's exactly why the
fix has to operate on the distribution.

**What we can't determine.** All 400 recordings are unique and none is used as both a target and a
masker. So "these recordings were processed because they were targets" and "already-processed
recordings were chosen as targets" are indistinguishable from the rendered audio. Settling it needs
the stimulus-generation code. It doesn't change any result — the probe measures readability
regardless of origin, and both fixes work regardless of cause — but for future data collection,
using each recording as target in some trials and masker in others would remove the problem at
source.

**Bug note.** The first run returned NaN for two descriptors and crashed the multivariate fit.
Cause: zero-phase filtering of `|Hilbert(x)|` can dip slightly below zero, so the 1st percentile of
the envelope was negative and its logarithm undefined; the NaN column then broke the least-squares
solve. Fixed by clipping to a positive floor and standardising the design matrix. Numbers above
are from the corrected run.

---

### 7.3 Problem 2 — the scoring formula lets the model ignore the brain

This is the one that actually broke the model.

#### The old code

```python
# model_classification.py
def _cosine_sim(self, a, b):                                              # 135
    return (F.normalize(a, dim=2) * F.normalize(b, dim=2)).mean(dim=1)    # 136

logits = []                                                               # 170
for aud in audio:                                                         # 171
    aud_enc = self.audio_encoder(aud)                                     # 172
    sim     = self._cosine_sim(brain_enc, aud_enc)                        # 173
    logits.append(self.sim_proj(sim))                                     # 174
return F.softmax(torch.cat(logits, dim=1), dim=1)                         # 176
```

Reading line 136 slowly: at each time point `t`, take the EEG's 16-number vector `b_t` and scale it
to length 1 (`F.normalize`); do the same to the candidate's `a_{k,t}`; multiply them element by
element; then average over the 640 time points. Out comes a 16-number similarity vector. Line 174
dots that with a learned weight vector `w` and adds a bias `β`.

In symbols, writing `b̂_t = b_t/‖b_t‖`:

```
sim_d(k) = (1/T) Σ_t  b̂_{t,d} · â_{k,t,d}
logit_k  = ⟨w, sim(k)⟩ + β
```

#### What goes wrong

Ask what happens if the EEG encoder gives up and outputs **the same 16 numbers at every time
point** — call that constant vector `b`, and its normalised version `b̂`. This is the state an
encoder drifts into when it has learned to ignore its input.

Substitute `b̂_t = b̂` (same at every `t`) into the formula:

```
sim_d(k) = (1/T) Σ_t  b̂_d · â_{k,t,d}        ← b̂_d doesn't depend on t, so pull it out
         = b̂_d · (1/T) Σ_t â_{k,t,d}
         = b̂_d · mean_t(â_k)_d
```

so `sim(k) = b̂ ⊙ mean_t(â_k)`, and therefore

```
logit_k = ⟨w, b̂ ⊙ mean_t(â_k)⟩ + β = ⟨w ⊙ b̂, mean_t(â_k)⟩ + β
```

**Look at what that is.** It's a weight vector `u = w ⊙ b̂` dotted with the time-averaged audio
embedding, plus a bias. That is a linear classifier operating on the audio alone. The scores still
differ between candidates — they have to, because `mean_t(â_k)` differs — so the model can rank the
candidates correctly *with an EEG embedding that carries no information whatsoever*.

And `u` has 16 free parameters, which is comfortably enough to read the shape statistics from
Problem 1.

So the failure was never an optimisation accident. **The architecture contains a working
audio-only classifier as a special case**, and that case is easier for gradient descent to reach
than the real solution. It found it. Measured: contribution +0.0009, decisions unchanged when we
shuffled the EEG (flip rate 0.008), identical accuracy when we fed literal zeros.

#### The fix

`fixed/model_v2.py:179-212`:

```python
class CouplingHead(nn.Module):
    """Time-centred per-dimension correlation score.

        score(b, a) = tau * w . corr_t(b, a)      (w bias-free)

    Structural guarantee: if the brain embedding is constant over time its
    centred version is exactly 0, so every candidate scores 0, the logits tie
    and the model sits at chance.  There is no constant-brain solution to find.
    """

    def __init__(self, D, learn_w=True, init_tau=0.07):
        super().__init__()
        self.w = nn.Linear(D, 1, bias=False)                    # note: no bias
        self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))

    @staticmethod
    def corr(b, a, eps=1e-6):
        """Per-dimension Pearson correlation over time. b,a: (B,T,D) -> (B,D)."""
        b = b - b.mean(1, keepdim=True)            # <-- THE FIX: remove each signal's
        a = a - a.mean(1, keepdim=True)            #     own average over time
        b = b / (b.norm(dim=1, keepdim=True) + eps)
        a = a / (a.norm(dim=1, keepdim=True) + eps)
        return (b * a).sum(1)

    def score_from_corr(self, c):
        return self.w(c).squeeze(-1) * self.log_tau.exp()

    def forward(self, brain, aud_encs):
        s = torch.stack([self.score_from_corr(self.corr(brain, a))
                         for a in aud_encs], dim=1)    # (B,K)
        return s - s.mean(1, keepdim=True)             # candidate-centring
```

The whole fix is the two `- mean(1)` lines. Instead of normalising each time point separately, we
subtract each signal's own average **over time**, then normalise the whole 640-point trace. That
turns the comparison into an ordinary correlation coefficient.

#### Why the model can no longer cheat

Run the same substitution. If `b_t = b` at every time point, then the average over time is also
`b`, so:

```
b − mean_t(b) = b − b = 0
```

Not small — **exactly zero**, every dimension, every time point. And zero correlates with nothing:

```
corr(0, a_k) = 0     for every candidate k
score_k      = τ · ⟨w, 0⟩ = 0
```

Every candidate scores 0, so they all tie, the softmax is uniform, and accuracy is pinned at `1/K`.
The escape hatch is not discouraged or penalised — it is arithmetically unreachable.

Verified numerically: with a constant EEG embedding, the old head produces scores that vary across
candidates by 0.011, while the new head produces `2.7 × 10⁻⁸` — zero to floating-point precision.

There's a stronger way to state this. The score depends on the EEG embedding **only through its
fluctuating part**. Add any constant vector to every time point and nothing changes. So the DC
component of the embedding is not merely unhelpful; it is invisible to the score.

#### Two smaller properties that come for free

**Loudness immunity.** For any `α > 0` and any `β`:

```
(αa + β) − mean_t(αa + β) = α(a − mean_t a)
```

The `β` cancels in the subtraction, and dividing by the norm — which is now `α‖a − mean_t a‖` —
cancels the `α`. So `corr(b, αa + β) = corr(b, a)`. Verified: difference `2.2 × 10⁻⁸`.

This is a second, independent line of defence: even with unfixed candidates, this head can't read a
pure volume cue. It *can* still read a shape cue, which is why Problem 1 has to be fixed at the
data level too — the two fixes cover different things.

**Candidate-centring.** The last line subtracts the mean score across candidates. If something adds
the same amount `γ` to every candidate's score, it cancels:

```
(s_k + γ) − (1/K)Σ_j(s_j + γ) = s_k − (1/K)Σ_j s_j
```

Dropping the bias term from `w` serves the same purpose — a bias is by definition the same for
every candidate, so it can only shift all scores together and can never change which one wins.

#### Result

Contribution +0.0000 → **+0.0111**. Small on its own, because the candidates were still confounded
— but this change is what makes Problem 4's fix expressible at all. (With the old head, a constant
EEG embedding still produces candidate-dependent scores, so a loss term demanding "real EEG must
beat shuffled EEG" has a solution that satisfies it while remaining audio-driven. With the new
head, no such solution exists.)

---

### 7.4 Problem 3 — the encoder was badly built

Three independent flaws, all in these lines:

```python
# model_classification.py
dilation = kernel_size ** i               # 38  gives 1, 3, 9, 27, 81, 243, 729
padding  = dilation * (kernel_size - 1)   # 39
self.dil_convs.append(...)                # 40  — no normalisation layer anywhere in this file
self.acts.append(nn.ReLU())               # 46  ReLU on every layer, including the last
x = x[:, :, :-(conv.padding[0])]          # 59  past-only ("causal") receptive field
```

#### (a) The filter reaches 34 seconds into a 10-second window

The **receptive field** is how far back a single output sample can see. Each layer with dilation
`d` and kernel size `k` extends it by `d·(k−1)`, so for `L` layers:

```
RF = 1 + (k − 1) · Σᵢ dᵢ
```

With `dᵢ = kⁱ` this collapses to something neat: `RF = k^L`. Here `k = 3`, `L = 7`, so

```
RF = 3⁷ = 2187 samples = 34.2 seconds at 64 Hz
```

The input window is 640 samples — **10 seconds**. The filter is over three times longer than the
thing it's looking at, so most of what it processes is the zero padding used to make the shapes
work out.

How much? With past-only padding, output position `t` sees inputs `[t − 2186, t]`, of which only
`t + 1` are real. Averaging across the window:

```
(1/T) Σ_{t=0}^{639} (t+1)/2187 = (640 + 1)/(2 × 2187) = 0.147
```

**About 85 % of what the deepest layer convolves is zeros.** Zeros are the same in every window, so
the output is dominated by a fixed pattern that has nothing to do with the input — which is a
direct route to producing the same embedding no matter what you feed it.

The fix uses `dᵢ = 2ⁱ` over 5 layers: `RF = 1 + 2(2⁵ − 1) = 63` samples = **0.98 s**. That matches
the timescale of the brain response we're looking for (0–400 ms), and with centred padding 90 % of
output positions see no padding at all.

#### (b) A ReLU on the last layer forces everything to look alike

`ReLU` sets negatives to zero, so after line 46 every embedding has non-negative entries. Two
consequences:

*It can never be negative.* For non-negative `u, v`, the dot product `⟨u,v⟩ = Σ u_d v_d` is a sum
of non-negative terms, so cosine similarity is stuck in `[0, 1]`. Two embeddings can never be
anti-correlated.

*It's usually near 1.* If the 16 coordinates have mean `μ > 0` and variance `σ²`, then for two
independent embeddings `E⟨u,v⟩ = 16μ²` and `E‖u‖² = 16(μ² + σ²)`, so

```
cos(u, v) ≈ μ² / (μ² + σ²) = 1 / (1 + (σ/μ)²)
```

When the coordinates don't vary much relative to their mean, this goes to 1. And the old encoder is
exactly in that regime — because of (a), most of its input is constant padding, so `σ/μ` is small.

This matters because the old scoring function *is* a cosine. The architecture squeezed the very
quantity it was measuring into a narrow band near 1. **Measured: 0.9995 to 1.0000 between different
windows** — the encoder was emitting practically the same array regardless of input.

The fix drops the final activation, which lets the coordinates centre near zero, giving `cos ≈ 0`
with small fluctuations. Toy check: 0.48 with the ReLU versus 0.20 without.

#### (c) The encoder looks backwards, but the signal is forwards

The brain's response to a sound arrives 100–300 ms *after* the sound. So to say anything about the
audio at time `t`, you want EEG from `t + 100 ms` to `t + 300 ms` — the *future* relative to `t`.

Line 59 makes the encoder strictly causal: at position `t` it sees only EEG up to `t`. That EEG
reflects audio from before `t − 100 ms`. To connect it to audio *at* `t`, the model has to bridge
the gap using the envelope's own predictability. It's working against the physics.

The fix uses symmetric padding, so position `t` sees roughly ±0.49 s — comfortably containing the
whole response.

#### The fixed encoder

`fixed/model_v2.py:109-160`:

```python
for i in range(layers):                                   # layers = 5, not 7
    d = 2 ** i                                            # 1,2,4,8,16 — not 1,3,9,27,81,243,729
    full = d * (kernel_size - 1)
    pad = full if direction in ("past", "future") else full // 2      # centred by default
    self.convs.append(nn.Conv1d(ch, dilation_filters, kernel_size,
                                dilation=d, padding=pad))
    self.norms.append(nn.GroupNorm(min(4, dilation_filters), dilation_filters))   # NEW
    self.trims.append(pad if direction in ("past", "future") else 0)
    ch = dilation_filters
self.receptive_field = 1 + sum(2 ** i * (kernel_size - 1) for i in range(layers))  # = 63

def forward(self, x):
    ...
    for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
        x = conv(x)
        if self.trims[i] > 0:
            x = (x[:, :, :-self.trims[i]] if self.direction == "past"
                 else x[:, :, self.trims[i]:])
        x = norm(x)
        if i < self.n_layers - 1:            # NO activation on the last layer
            x = self.drop(F.relu(x))
    return x.transpose(1, 2)
```

`GroupNorm` rescales each group of 4 channels to zero mean and unit variance, which stops a stack
of rectified layers from drifting into a near-constant regime.

#### Result — and an honest note

Embedding similarity dropped from 0.78–0.89 to 0.29. **But on its own this change bought nothing:**
accuracy went *up* to 0.5501 while the contribution stayed at exactly +0.0000.

A better-conditioned encoder just reads the audio shortcut more efficiently. These three arguments
show the old encoder *couldn't* produce a useful embedding; they say nothing about whether a good
one will be *used*. That's Problem 4's job. Necessary, not sufficient.

---

### 7.5 Problem 4 — nothing in the training objective rewarded using the brain

#### The old code

```python
# train_aad.py
def _label_smoothing_ce(probs, targets, smoothing=0.1):                  # 66
    smooth_tgt = targets * (1 - smoothing) + smoothing / n_cls           # 68
    return -(smooth_tgt * torch.log(probs + 1e-8)).sum(dim=1).mean()     # 69
loss = _label_smoothing_ce(probs, labels, smoothing)                     # 90

if vl_acc > best_acc:                                                    # 115
    best_acc = vl_acc; torch.save(model.state_dict(), ckpt_path)         # 116
```

Line 90 is the entire objective. It is a function of the four scores and nothing else. It has no
opinion about *how* those scores were produced — and by §7.3 they can be produced without the EEG.
So the degenerate solution costs nothing. The loss surface is flat in exactly the direction we care
about.

Then line 115 keeps whichever epoch had the best validation accuracy, which is whichever epoch
exploited the shortcut best. The bad solution is not just permitted, it's actively selected.

#### The fix, term by term

`fixed/losses_v2.py`. Four new terms, added to the ordinary task loss.

**(a) Contrastive term — makes collapse the *worst* option, not the easiest**

```python
def clip_brain_axis(head, brain, aud_pos, subj):
    B = brain.shape[0]
    S = pairwise_scores(head, brain, aud_pos)        # S[i,j] = score(EEG_i, audio_j)
    same = subj[:, None] == subj[None, :]
    S = S.masked_fill(~same, float("-inf"))          # only compare within one listener
    tgt = torch.arange(B, device=brain.device)
    l1 = F.cross_entropy(S[usable], tgt[usable])     # each EEG must pick its own audio
    l2 = F.cross_entropy(S.t()[usable], tgt[usable]) # and each audio its own EEG
    return 0.5 * (l1 + l2)
```

Within a batch of `B` windows, build the full `B × B` table of scores between every EEG and every
audio. The correct answer for row `i` is column `i`. This asks: *which brain recording goes with
which stimulus?*

Now watch what a collapsed encoder does to it. If every window produces the same embedding `b`,
then `S[i,j] = score(b, audio_j)`, which **doesn't depend on `i` at all**. Every row of the table
is identical — call that shared row `c`.

For the first term, the loss works out to `logsumexp(c) − mean(c)`. By Jensen's inequality, the
average of `e^{c_j}` is at least `e^{mean(c)}`, which rearranges to

```
logsumexp(c) ≥ log B + mean(c)      hence      CE(S, I) ≥ log B
```

and `log B` is exactly the loss you get from guessing at random. For the second term it's even more
direct: each row of `Sᵀ` is the constant vector `(c_i, c_i, …, c_i)`, whose softmax is perfectly
uniform, so that half equals `log B` on the nose.

**So collapsing doesn't minimise this term — it pins it at chance, the worst achievable value.**
This is the term that supplies the gradient pressure the task loss never had. Verified at `B = 32`:
CE(S,I) = 4.023 and CE(Sᵀ,I) = 3.4657, against `log 32` = 3.4657.

*Why only within one listener.* Raw EEG identifies **which of the 16 people** it came from with
0.90 accuracy (chance 0.0625). Across listeners, the score table is separable by identity alone, so
the model could drive this loss down by learning *who* rather than *what* — a second degenerate
solution, and a subtle one. The `masked_fill` line blocks it, and we draw each batch from a single
listener so the masking doesn't waste the batch.

**(b) Permutation hinges — the evaluation test, written as a loss**

```python
def null_hinges(head, brain, aud_pos, margin=0.5):
    """real-brain score must beat shuffled-brain and zeros-brain scores."""
    B = brain.shape[0]
    roll = torch.roll(torch.arange(B, device=brain.device), 1)
    s_real = head.score_from_corr(head.corr(brain, aud_pos))
    s_shuf = head.score_from_corr(head.corr(brain[roll], aud_pos))
    s_zero = head.score_from_corr(head.corr(torch.zeros_like(brain), aud_pos))
    return (F.softplus(s_shuf - s_real + margin).mean()
            + F.softplus(s_zero - s_real + margin).mean())
```

Three scores per window: with its own EEG, with the next window's EEG (`roll`), and with zeros. The
loss demands the real one win by at least `margin`.

At the collapsed solution, `s_real = s_shuf` (the score doesn't care whose EEG it got) and
`s_zero = 0 = s_real`. Both terms sit at `softplus(0.5) ≈ 0.974` — a strictly positive penalty with
a non-zero derivative, so the very first gradient step pushes the real-EEG score upward.

The quantity these hinges maximise, `s_real − s_shuf`, is **exactly the contribution we report in
§5**. The training objective and the evaluation metric are the same functional. That is deliberate:
we optimise the thing we claim.

**(c) Anti-collapse term — penalise flatness directly**

```python
def anti_collapse(brain, gamma=0.5, l_var=1.0, l_cov=0.04):
    zc = brain - brain.mean(1, keepdim=True)                     # (B,T,D)
    var = F.relu(gamma - zc.std(1)).mean()                       # temporal std hinge
    z = zc.reshape(-1, brain.shape[-1])
    z = z - z.mean(0)
    cov = (z.t() @ z) / max(1, z.shape[0] - 1)
    off = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    return l_var * var + l_cov * off
```

The first line measures how much each embedding dimension varies **over time** — the exact quantity
that goes to zero when the encoder collapses — and penalises it whenever it drops below `γ = 0.5`.
The second penalises correlation *between* dimensions, blocking the cheap escape of satisfying the
variance requirement with 16 copies of the same signal.

**(d) Selection criterion**

`fixed/train_v2.py:305-320`:

```python
best = {"margin": (-9e9, None), "acc": (-9e9, None)}
for ep in range(1, args.epochs + 1):
    tl = run_epoch(model, tr_loader, opt, lcfg, True)
    ev = Evaluator(model, vl_loader, DEVICE)
    b  = ev.battery(n_shuffle=args.val_shuffles, seed=7)   # computes acc AND shuffled-acc
    sch.step(b["margin"])
    if b["margin"] > best["margin"][0]:                    # <-- was: if b["acc"] > ...
        best["margin"] = (b["margin"], {...state dict...})
```

Split validation accuracy into two parts: `A = N + M`, where `N` is the accuracy you'd get with
shuffled EEG (the audio-only part) and `M` is the contribution.

Picking the epoch with the best `A` maximises the sum. When a shortcut exists, `N` is both larger
and more variable across epochs than `M`, so the winner is decided by noise in `N` — the part we
don't want. Picking the best `M` targets the contribution directly.

Measured: with a shortcut present the two criteria differ by a factor of two (contribution 0.1689
vs 0.0929). Once the candidates are clean, `N ≈ 1/K` is constant, so `A` and `M` differ only by a
constant, the two criteria pick the same epoch, and the measured results agree to within 0.005.
**The criterion is a safeguard that does nothing when the data are clean — it cannot manufacture an
effect.**

#### Result

Contribution +0.0111 → **+0.1689**, with nothing else changed. The largest single change in the
study.

---

### 7.6 Problem 5 — all four modalities were given the same, wrong task

#### The old code

```python
# model_classification.py
self.fusion = nn.Sequential(nn.Linear(n_enc*D_common, D_common), nn.ReLU())  # 123
embeddings.append(self.eeg_proj(self.eeg_encoder(eeg)))        # 151
embeddings.append(self.video_proj(self.video_encoder(video)))  # 153
embeddings.append(self.gaze_proj(self.gaze_encoder(gaze)))     # 155
embeddings.append(self.imu_proj(self.imu_encoder(imu)))        # 157
brain_enc = self.fusion(torch.cat(embeddings, dim=2))          # 166
```

Everything is merged at line 166 and then sent through the audio-matching head at line 173.

The audio-matching head asks: *does this signal's moment-to-moment wiggle line up with the speech
envelope's moment-to-moment wiggle?* For EEG that's a real question with a real answer. For gaze
direction, head acceleration and scene optical flow, there is no such relationship to find — the
true answer is zero. Those branches weren't failing at a hard task; they were being asked a
question with no answer.

Line 123 adds a second problem: no modality dropout and no per-modality loss, so once one branch
gets good the others stop receiving useful gradient.

#### The fix — give each modality a question it can answer

`fixed/model_v2.py:215-228`:

```python
class SpatialHead(nn.Module):
    """Predicts the attended SPEAKER INDEX (a fixed loudspeaker azimuth)
    straight from a modality embedding.  Takes no audio input, so it is
    structurally incapable of using the acoustic shortcut — whatever it scores
    is genuinely orienting/lateralisation information."""

    def __init__(self, D, n_spk=N_SPEAKERS, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_spk))

    def forward(self, emb):                            # (B,T,D) -> (B,n_spk)
        return self.net(torch.cat([emb.mean(1), emb.std(1)], dim=-1))
```

Where the listener is *oriented* predicts which loudspeaker they were attending, and gaze, head
motion and scene video all measure orientation. So each gets a classifier over loudspeaker index —
"which of the four speakers was this person facing?" — summarising its embedding by mean and
standard deviation over time.

**Crucially, this head never sees the audio.** Not "we hope it won't use it": there is no audio
argument to `forward`.

#### Why its shuffle test is exactly chance — for free

This branch outputs four numbers `ℓ`, one per loudspeaker. Slot `k` in the shuffled candidate order
holds loudspeaker `π_k`, so `score_k = ℓ_{π_k}`, and the model picks
`argmax_k ℓ_{π_k}`, which is the slot holding the loudspeaker with the highest `ℓ`.

Now shuffle: window `i`'s audio and label get window `j`'s recording. Working through the
permutation, the prediction is correct exactly when `argmax ℓ(x_j) = ` window `i`'s true
loudspeaker. Since `j` is unrelated to `i`, this is just "does an unrelated window's guess happen to
match?" With the four loudspeakers balanced at 25 % each:

```
P(correct) = Σ_s P(guess = s) × P(true = s) = Σ_s q_s × (1/4) = 1/4
```

because the `q_s` sum to 1 whatever they are.

**This holds no matter how biased the classifier is.** A branch that always answers "loudspeaker 2"
still nulls at exactly 25 %. Verified: 0.2501 for a maximally biased classifier, 0.2495 for a random
one. Measured in the real runs: 0.2497–0.2515 across all four modalities.

So for these branches, anything above 25 % is real, by construction rather than by argument.

#### Keeping every branch alive

```python
keep = [m for m in embs if torch.rand(()) > self.modality_dropout]   # :343  p = 0.3
```

Without this, once one branch classifies correctly the gradient reaching the others is
`∂L/∂score × ∂score/∂e_m × …`, and the first factor has already gone to zero. Dropping the strong
branch 30 % of the time forces the others to be independently useful on those steps. The per-branch
auxiliary loss (`fixed/losses_v2.py:85-90`) makes this unconditional by giving each encoder a
gradient path that doesn't pass through the fusion layer at all.

#### Result

Every modality went from a contribution of 0.000 to between +0.116 and +0.181 on its own, +0.250
for the three behavioural streams fused, and +0.318 with the EEG coupling branch added.

---

### 7.7 Problem 6 — there was no shuffle test at all

`grep -rn "shuffle\|permut" scripts/` finds nothing relevant. Without it, the reported number can't
be interpreted: you don't know what fraction is audio-only, and in this case that fraction was all
of it.

#### Why we shuffle the brain and not the labels

There are two obvious permutation tests and only one of them works here.

*Shuffling labels* breaks the link between the audio and the answer **and** between the brain and
the answer. A model that only uses audio would also collapse to chance, so the test comes back
"significant" for a model with no brain content at all. It hides exactly what we're hunting.

*Shuffling the brain* leaves each window's audio and answer paired, breaking only the brain-to-answer
link. An audio-only model scores unchanged; a brain-using model drops. That's the test we want.

In one line: we're testing "is the score independent of the brain, given the audio and the answer?",
not "are audio and brain jointly independent of the answer?"

#### The lag-band control — telling a brain response from an electrical artifact

One worry remains. Stimulus audio can leak electrically into an EEG recording. That leak would be
genuinely window-specific — it really is *this* window's audio in *this* window's EEG — so it would
survive the shuffle test and look like success.

But it's distinguishable, because of timing. A brain response to a sound arrives 100–300 ms
*after* it. An electrical leak arrives instantly, and shows up symmetrically whether you look
forwards or backwards in time.

So we restrict the encoder to look only one way, and shift the EEG:

```python
# fixed/model_v2.py:319-326
if self.lag_samples and m == self.couple_mod:
    # x'[t] = x[t + lag]: positive lag reads the brain AFTER the stimulus sample it is
    # matched against (the neural direction); negative lag reads it BEFORE, which no
    # neural response can explain and which therefore isolates shared artifacts.
    L = self.lag_samples
    x = (F.pad(x, (0, 0, 0, L))[:, L:] if L > 0
         else F.pad(x, (0, 0, -L, 0))[:, :L])
```

Combined with the one-sided receptive field from §7.4, this gives two disjoint windows:

| Setting | EEG visible at | Can a brain response live here? |
|---|---|---|
| `direction="future"`, `Λ = +8` | +125 … +1109 ms | **yes** |
| `direction="past"`, `Λ = −8` | −1109 … −125 ms | **no** — it would precede its own cause |

(8 samples = 125 ms; the receptive field of 63 samples = 984 ms.)

An electrical leak sits at zero lag and contributes to both equally. A brain response contributes
only to the first. **The difference between the two is therefore attributable to neural activity.**

Measured: **+0.2282 versus +0.0179** for the binary task, **+0.2256 versus +0.0052** for the
four-way — ratios of 13 and 43, with neither backwards-looking version significantly above its own
chance level (p = 0.181 and 0.314). Artifacts and time-symmetric drift are excluded. And restricting
to the correct direction *improves* the result: 0.7282 against a null of exactly 0.5000 is the best
number in the study.

*One caveat.* The backwards band isn't perfectly empty in principle: speech envelopes are
autocorrelated over about a second, so `s(t)` is partly predictable from `s(t + Δ)` and a response
could leak backwards through that route. The measured +0.0179 is consistent with a small leak of
this kind. It doesn't affect the argument, which rests on the *difference* between the two
directions — leakage can only shrink that difference, never create it.

---

### 7.8 Checking the arguments numerically

Every claim above is checked in `fixed/verify_propositions.py`. Run it directly:

```
$ python verify_propositions.py

P1  max |Δ| over gains ±15 dB          = 1.60e-05   (predicted: 0, float eps)
P5  old head, spread across candidates = 0.0110     (predicted: > 0, a classifier exists)
P6  new head, max |score|              = 2.69e-08   (predicted: 0, chance is forced)
P7  |corr(b,3a+7) − corr(b,a)|max      = 2.24e-08   (predicted: 0)
P9  RF old = 2187 (34.2 s), RF new = 63 (0.98 s); mean real fraction = 0.147
P10 mean pairwise cosine: rectified = 0.4811, centred = 0.2042
P11 collapsed InfoNCE: CE(S,I) = 4.0233, CE(Sᵀ,I) = 3.4657, log B = 3.4657
P14 permuted acc, unbiased             = 0.2495     (predicted: 0.2500)
P14 permuted acc, always speaker 2     = 0.2501     (predicted: 0.2500)
```

Reading it: **P5 vs P6** is the heart of §7.3 — the same constant-EEG input gives the old head a
working classifier (spread 0.011) and the new head nothing (2.7e−8). **P11** confirms the collapsed
contrastive loss can't beat `log B`, with the symmetric half landing exactly on it as the algebra
requires. **P14** confirms that even a maximally biased audio-free branch nulls at 25 %.

Two honest notes. The three random-draw checks (P5, P10, P11) move slightly between runs; the
predicted *relations* hold every time. And P10's toy check gives 0.48 versus 0.20 — the right
direction, but far short of the 0.9995 we measured in the real encoder. The gap is itself
informative: the real encoder also suffers from §7.4(a), so 85 % of its receptive field is constant
padding, which drives `σ/μ` down and pushes `1/(1 + (σ/μ)²)` toward 1. **The flaws compound rather
than merely coexist.**

---

### 7.9 Summary

| Problem | Where it was | What we changed | Effect on its own |
|---|---|---|---|
| 1 Candidates are distinguishable from audio alone | `dataloader.py:244-252` | `candidates_v2.py:42-95`, `data_v2.py:59,70` | +0.169 → +0.194, and moves the null to chance |
| 2 Scoring formula permits ignoring the brain | `model_classification.py:135-136,170-176` | `model_v2.py:179-212` | +0.000 → +0.011; unlocks fix 4 |
| 3 Encoder badly conditioned | `model_classification.py:34-61` | `model_v2.py:109-160` | **+0.000 — nothing** |
| 4 Objective never rewards using the brain | `train_aad.py:66-69,90,115-116` | `losses_v2.py:42-112`, `train_v2.py:52,312-313` | +0.011 → **+0.169** |
| 5 Wrong task for gaze / head / video | `model_classification.py:123,149-168` | `model_v2.py:215-228,343,365,379` | 0.000 → +0.250 fused |
| 6 No shuffle test | absent | `train_v2.py:185-247`, `model_v2.py:319-326` | makes everything else interpretable |

Ranked by measured effect: **objective ≫ candidate construction > scoring formula ≫ encoder**, with
the encoder contributing nothing measurable on its own.

The headline lesson is that this was not a modelling failure. The network was fine at what it was
asked to do. Nothing in the objective ever asked it to use the brain, the scoring formula offered
an exact alternative that didn't require one, and the stimulus material made that alternative good
enough. It took the deal.

| | Old | New |
|---|---|---|
| Four-way accuracy | 0.4769 | 0.4709 |
| Accuracy with the EEG shuffled | 0.4760 | 0.2772 (chance 0.2500) |
| **Contribution of the recording** | **+0.0009** | **+0.1937** |
| Decisions changed by shuffling | 0.8 % | 69.9 % |
| Embedding collapse | 0.780 | 0.291 |
| Audio-only leak in the candidates | +0.3097 | +0.0100 |
| Parameters | 10 425 | 8 698 |

### 7.10 Configuration names

| Name in the manuscript | JSON key in `results/fixed/` |
|---|---|
| Standard architecture | `A0` |
| Standard architecture, alternative optimiser | `A0b` |
| + redesigned encoder | `A1` |
| + correlation scoring | `A2` |
| + permutation-aware objective | `A3` |
| Proposed model, four-way | `A4` |
| Proposed model, three-way, no distribution matching | `A5r` |
| Proposed model, three-way | `A5` |
| Proposed model, binary | `A6` |
| Four-way / binary, causal band | `A4f` / `A6f` |
| Four-way / binary, acausal band | `A4b` / `A6b` |
| Orientation: gaze / head / video / EEG | `M-gaze` / `M-imu` / `M-video` / `M-eeg-spatial` |
| Orientation fusion | `M-behav` |
| Complete multimodal model | `M-full` |

### 7.11 Reproduction

```bash
cd analysis/n_gh_checks/fixed
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

$PY train_v2.py --configs A0,A0b,A1,A2,A3,A4,A5,A6 --split_setting within
$PY train_v2.py --configs A4,A6,M-behav,M-full     --split_setting loso
$PY train_v2.py --configs A6f,A6b,A4f,A4b          --split_setting within
$PY train_v2.py --configs A4,A6 --window_sec 8 --hop_sec 8
$PY ridge_baseline.py --split_setting within --band 1,8
$PY make_report.py
```

Batch wrappers `slurm/{fix_ladder,fix_debug,fix_long,ridge_band}.sbatch`; results
`results/fixed/*.json`; jobs 6900038, 6900360–4, 6901199 (ablation); 6904642–7 (LOSO);
6904648–56, 6906613–5 (sweep, linear reference, reruns); 6906608, 6906610–2 (stratified
permutations, band-limited reference); 6907532 (lag-band control).
