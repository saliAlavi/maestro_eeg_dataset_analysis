"""Losses that make the brain-shuffle null part of the objective.

The upstream loss is a plain 4-way CE on the softmax output.  Nothing in it
ever rewards using the brain, and the checkpoint is then selected on inner-val
accuracy — which the audio shortcut already maximises — so the collapsed
solution is what gets saved.  The terms below attack that directly:

  clip_brain_axis  — CLIP-style InfoNCE along the BRAIN axis: for each window
                     its own attended-audio embedding is the positive and other
                     windows' are negatives, and symmetrically.  A collapsed
                     brain encoder makes the similarity matrix rank-1, which
                     pins this term at chance — it cannot be minimised by
                     collapsing.  Negatives are restricted to the same subject.

  null_hinges      — the shuffle test written as a loss: the real brain must
                     score the correct candidate above (a) another window's
                     brain and (b) a zeros brain, by a margin.

  anti_collapse    — VICReg-style: hinge on the per-dimension TEMPORAL std of
                     the brain embedding (the exact quantity that went to ~0
                     upstream) plus an off-diagonal covariance penalty.

  adversary CE     — an audio-only head trained to name the attended candidate
                     from the audio embeddings alone; the audio encoder gets the
                     reversed gradient, so a readable shortcut is unlearned.
"""

import torch
import torch.nn.functional as F


def pairwise_scores(head, brain, aud):
    """All-pairs coupling score. brain (B,T,D), aud (B,T,D) -> (B,B)."""
    b = brain - brain.mean(1, keepdim=True)
    a = aud - aud.mean(1, keepdim=True)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-6)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-6)
    C = torch.einsum("itd,jtd->ijd", b, a)                       # (B,B,D)
    return head.score_from_corr(C)


def clip_brain_axis(head, brain, aud_pos, subj):
    B = brain.shape[0]
    S = pairwise_scores(head, brain, aud_pos)
    same = subj[:, None] == subj[None, :]
    S = S.masked_fill(~same, float("-inf"))
    tgt = torch.arange(B, device=brain.device)
    usable = same.sum(1) >= 2
    if usable.sum() == 0:
        return brain.sum() * 0.0
    l1 = F.cross_entropy(S[usable], tgt[usable])
    l2 = F.cross_entropy(S.t()[usable], tgt[usable])
    return 0.5 * (l1 + l2)


def null_hinges(head, brain, aud_pos, margin=0.5):
    """real-brain score must beat shuffled-brain and zeros-brain scores."""
    B = brain.shape[0]
    roll = torch.roll(torch.arange(B, device=brain.device), 1)
    s_real = head.score_from_corr(head.corr(brain, aud_pos))
    s_shuf = head.score_from_corr(head.corr(brain[roll], aud_pos))
    s_zero = head.score_from_corr(head.corr(torch.zeros_like(brain), aud_pos))
    return (F.softplus(s_shuf - s_real + margin).mean()
            + F.softplus(s_zero - s_real + margin).mean())


def anti_collapse(brain, gamma=0.5, l_var=1.0, l_cov=0.04):
    zc = brain - brain.mean(1, keepdim=True)                     # (B,T,D)
    var = F.relu(gamma - zc.std(1)).mean()                       # temporal std hinge
    z = zc.reshape(-1, brain.shape[-1])
    z = z - z.mean(0)
    cov = (z.t() @ z) / max(1, z.shape[0] - 1)
    D = z.shape[1]
    off = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    return l_var * var + l_cov * off


def total_loss(out, label, subj, cfg, spk_label=None):
    """Returns (loss, dict of components)."""
    parts = {}
    logits = out["logits"]
    loss = F.cross_entropy(logits, label, label_smoothing=cfg.get("smoothing", 0.1))
    parts["ce"] = float(loss.detach())

    # per-branch auxiliary spatial CE (stops one branch from starving the rest)
    if out["spk_logits"] and spk_label is not None and cfg.get("w_aux", 0) > 0:
        aux = sum(F.cross_entropy(v, spk_label)
                  for k, v in out["spk_logits"].items() if k != "fused")
        aux = aux / max(1, len([k for k in out["spk_logits"] if k != "fused"]))
        loss = loss + cfg["w_aux"] * aux
        parts["aux"] = float(aux.detach())

    brain, aud_encs = out.get("brain"), out.get("aud_encs")
    if brain is not None and aud_encs is not None and cfg.get("anti_shortcut"):
        head = cfg["head_module"]
        aud_pos = torch.stack(aud_encs, 1)[torch.arange(len(label)), label]
        if cfg.get("w_clip", 0) > 0:
            l = clip_brain_axis(head, brain, aud_pos, subj)
            loss = loss + cfg["w_clip"] * l
            parts["clip"] = float(l.detach())
        if cfg.get("w_null", 0) > 0:
            l = null_hinges(head, brain, aud_pos, cfg.get("margin", 0.5))
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
    return loss, parts
