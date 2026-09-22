"""Objective terms for MERIT.

The permutation test is what *detects* a decoder that ignores an input.  Every
term below exists to make that test part of the objective, so a decoder that
ignores an input cannot reach a low training loss in the first place.

  ``contribution_hinges``  -- the new term.  The permutation test written per
      stream: permuting stream ``m`` across the batch must cost the correct
      slot a margin of score.  A model that rides one strong stream and leaves
      the others dead pays for every dead stream.  Weights are self-gating --
      a stream already contributing on inner validation is not pushed further,
      so the term cannot manufacture a contribution that is not there.
  ``clip_brain_axis``      -- InfoNCE along the physiological axis.  A collapsed
      encoder makes the similarity matrix rank-one, which pins this term at
      chance: it cannot be minimised by collapsing.  Negatives are within-listener,
      because raw EEG identifies the listener at 0.90 in a sixteen-way test and a
      cross-listener batch is solved by identity alone.
  ``brain_null_hinges``    -- the global permutation test and the zeros ablation,
      as losses, on the coupling score.
  ``anti_collapse``        -- VICReg-style hinge on the per-dimension *temporal*
      standard deviation (the exact quantity that goes to zero when an encoder
      gives up) plus an off-diagonal covariance penalty.
  adversary cross-entropy  -- an audio-only head through a gradient-reversal
      layer, so a readable acoustic shortcut is unlearned from the audio encoder.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _slot_margin(logits, label):
    """Score of the correct slot relative to the mean over slots."""
    return logits.gather(1, label[:, None]).squeeze(1) - logits.mean(1)


def contribution_hinges(model, z, brain, aud_encs, perm, label,
                        weights=None, margin=0.3, include_zeros=True):
    """Per-stream permutation loss.

    For each stream ``m`` the descriptors of *other windows* are substituted for
    its own (a within-batch roll -- the training-time image of the test-time
    permutation null) and the correct slot must lose at least ``margin`` of
    score.  Optionally the same is required against a zeroed stream, which
    separates "uses ``m``" from "uses the batch statistics of ``m``".

    Returns ``(loss, {stream: float})`` where the dict reports the *realised*
    per-stream score drop, which the runner logs and feeds back into ``weights``.
    """
    B = label.shape[0]
    if B < 2 or not z:
        zero = (brain.sum() * 0.0 if brain is not None
                else next(iter(z.values())).sum() * 0.0)
        return zero, {}
    roll = torch.roll(torch.arange(B, device=label.device), 1)
    real_logits, _, _ = model.score(z, brain, aud_encs, perm)
    s_real = _slot_margin(real_logits, label)

    total, drops = 0.0, {}
    for m in list(z):
        w = 1.0 if weights is None else float(weights.get(m, 1.0))
        z_p = dict(z)
        z_p[m] = z[m][roll]
        b_p = brain[roll] if (m == model.couple_mod and brain is not None) else brain
        lg, _, _ = model.score(z_p, b_p, aud_encs, perm)
        s_perm = _slot_margin(lg, label)
        drops[m] = float((s_real - s_perm).mean().detach())
        if w <= 0:
            continue
        term = F.softplus(s_perm - s_real + margin).mean()
        if include_zeros:
            z_0 = dict(z)
            z_0[m] = torch.zeros_like(z[m])
            b_0 = (torch.zeros_like(brain)
                   if (m == model.couple_mod and brain is not None) else brain)
            lg0, _, _ = model.score(z_0, b_0, aud_encs, perm)
            term = term + F.softplus(_slot_margin(lg0, label) - s_real + margin).mean()
        total = total + w * term
    if isinstance(total, float):
        total = real_logits.sum() * 0.0
    return total, drops


def _pairwise_scores(head, brain, aud):
    b = brain - brain.mean(1, keepdim=True)
    a = aud - aud.mean(1, keepdim=True)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-6)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-6)
    return head.score_from_corr(torch.einsum("itd,jtd->ijd", b, a))


def clip_brain_axis(head, brain, aud_pos, subj):
    B = brain.shape[0]
    S = _pairwise_scores(head, brain, aud_pos)
    same = subj[:, None] == subj[None, :]
    S = S.masked_fill(~same, float("-inf"))
    tgt = torch.arange(B, device=brain.device)
    usable = same.sum(1) >= 2
    if usable.sum() == 0:
        return brain.sum() * 0.0
    return 0.5 * (F.cross_entropy(S[usable], tgt[usable])
                  + F.cross_entropy(S.t()[usable], tgt[usable]))


def brain_null_hinges(head, brain, aud_pos, margin=0.5):
    B = brain.shape[0]
    roll = torch.roll(torch.arange(B, device=brain.device), 1)
    s_real = head.score_from_corr(head.corr(brain, aud_pos))
    s_shuf = head.score_from_corr(head.corr(brain[roll], aud_pos))
    s_zero = head.score_from_corr(head.corr(torch.zeros_like(brain), aud_pos))
    return (F.softplus(s_shuf - s_real + margin).mean()
            + F.softplus(s_zero - s_real + margin).mean())


def anti_collapse(brain, gamma=0.5, l_var=1.0, l_cov=0.04):
    zc = brain - brain.mean(1, keepdim=True)
    var = F.relu(gamma - zc.std(1)).mean()
    z = zc.reshape(-1, brain.shape[-1])
    z = z - z.mean(0)
    cov = (z.t() @ z) / max(1, z.shape[0] - 1)
    D = z.shape[1]
    off = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    return l_var * var + l_cov * off


def total_loss(model, out, label, spk_label, subj, perm, cfg, hinge_weights=None):
    """Returns ``(loss, components, per-stream score drops)``."""
    parts, drops = {}, {}
    logits = out["logits"]
    loss = F.cross_entropy(logits, label, label_smoothing=cfg.get("smoothing", 0.1))
    parts["ce"] = float(loss.detach())

    # per-stream auxiliary CE: stops the strongest branch absorbing the gradient
    per = out.get("spk_logits") or {}
    if per and spk_label is not None and cfg.get("w_aux", 0) > 0:
        aux = sum(F.cross_entropy(v, spk_label) for v in per.values()) / len(per)
        loss = loss + cfg["w_aux"] * aux
        parts["aux"] = float(aux.detach())

    if cfg.get("w_contrib", 0) > 0:
        l, drops = contribution_hinges(
            model, out["z"], out["brain"], out["aud_encs"], perm, label,
            weights=hinge_weights, margin=cfg.get("contrib_margin", 0.3),
            include_zeros=cfg.get("contrib_zeros", True))
        loss = loss + cfg["w_contrib"] * l
        parts["contrib"] = float(l.detach()) if torch.is_tensor(l) else 0.0

    brain, aud_encs = out.get("brain"), out.get("aud_encs")
    if brain is not None and aud_encs is not None and cfg.get("anti_shortcut", True):
        head = model.head
        aud_pos = torch.stack(aud_encs, 1)[torch.arange(len(label)), label]
        if cfg.get("w_clip", 0) > 0:
            l = clip_brain_axis(head, brain, aud_pos, subj)
            loss = loss + cfg["w_clip"] * l
            parts["clip"] = float(l.detach())
        if cfg.get("w_null", 0) > 0:
            l = brain_null_hinges(head, brain, aud_pos, cfg.get("margin", 0.5))
            loss = loss + cfg["w_null"] * l
            parts["null"] = float(l.detach())
        if cfg.get("w_vic", 0) > 0:
            l = anti_collapse(brain)
            loss = loss + cfg["w_vic"] * l
            parts["vic"] = float(l.detach())
        if "adv_logits" in out and cfg.get("w_adv", 0) > 0:
            l = F.cross_entropy(out["adv_logits"], label)
            loss = loss + cfg["w_adv"] * l
            parts["adv"] = float(l.detach())
    return loss, parts, drops


__all__ = ["total_loss", "contribution_hinges", "clip_brain_axis",
           "brain_null_hinges", "anti_collapse"]
