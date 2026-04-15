"""Perturbation / norm helpers for adversarial attacks."""

from __future__ import annotations


def clip_linf(x, eps: float):
    """Clip tensor to the L_inf ball of radius eps around 0, element-wise."""
    import torch

    return torch.clamp(x, -eps, eps)


def clip_l2(x, eps: float):
    """Project tensor onto the L2 ball of radius eps (per-sample)."""
    import torch

    flat = x.view(x.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factor = (eps / norms).clamp(max=1.0)
    return (flat * factor).view_as(x)
