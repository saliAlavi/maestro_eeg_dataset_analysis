# maestro — MAESTRO-Net, the headline multimodal AAD decoder

## What it is
A stimulus-aware match-mismatch decoder that fuses **all** recorded modalities:
EEG (query) · per-speaker envelopes (keys) · gaze / IMU / video (gated
overt-orienting context, fused by a small transformer over a CLS token). One
trained model is evaluable on any modality subset and reports a
**leave-one-modality-out** table.

## Mechanisms
- **Match-mismatch** over the six candidate envelopes (speakers 5/6 = hard
  negatives) → stimulus-aware, comparable to the AAD challenge literature.
- **Modality dropout** during training → robust to any subset → LOMO is read off
  one model, not retrained per mask.
- **Subject FiLM** (zero-initialised) → seen subjects get a learned modulation;
  an unseen LOSO subject defaults to identity, no special-casing.
- **Adversarial gaze head** (gradient reversal) → the EEG query is penalised for
  being gaze-predictable. The accuracy under this pressure is our defensible
  "EEG-beyond-overt-orienting" number.
- **Aux losses**: InfoNCE EEG↔attended-envelope alignment + EEG→envelope
  reconstruction, both stronger training signals than bare cross-entropy.

## Motivation / intuition & what mistakes led here
Our diagnostics exposed the field's central confound on *our* data: gaze alone
(~0.77) beats EEG, and motion-residualised EEG is at chance. A model that simply
maximises accuracy would quietly become a gaze decoder. MAESTRO-Net is built so
that (a) the orienting modalities are *explicit and ablatable* rather than
hidden leakage, and (b) the adversarial head forces an honest read of the
EEG-only contribution. Earlier flat 4-class and EEG-only nets could neither fuse
the modalities principledly nor quantify each one's marginal value — MAESTRO-Net
does both from a single trained model.

## Key hyperparameters (`configs/model/maestro.yaml`)
- `d_model`, `n_heads`, `n_ctx_layers`, `dropout`
- `modality_dropout`, `w_match`, `w_info`, `w_recon`, `w_adv_gaze`,
  `info_temp`, `adv_lambda`
- Video is currently fed as a zero token (`present_video=0`); enable real video
  features in the data config when the egocentric pipeline is wired.
