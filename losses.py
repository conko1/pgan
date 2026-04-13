from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1(pred, target, mask, min_pixels=32.0):
    # L1 chyba počítaná iba v oblasti definovanej maskou.
    # min_pixels zabraňuje príliš veľkým hodnotám pri malých maskách.
    pixels = mask.sum(dim=(1, 2, 3)).clamp_min(min_pixels)
    return ((pred - target).abs() * mask).sum(dim=(1, 2, 3)).div(pixels).mean()


def boundary_ring(mask, k=7):
    # Vytvorí prstenec okolo hranice masky pomocou dilatácie a erózie.
    # Používa sa na zvýraznenie kvality prechodu na okrajoch.
    pad = k // 2
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


def d_hinge(real_logits, fake_logits):
    # Hinge loss pre diskriminátor.
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge(fake_logits):
    # Hinge loss pre generátor.
    return -fake_logits.mean()


def feature_matching(fake_feats, real_feats):
    # Porovnáva intermediate feature mapy reálnych a generovaných obrázkov.
    # Pomáha generátoru produkovať stabilnejšie a realistickejšie detaily.
    total, n = 0.0, 0
    for ff_scale, rf_scale in zip(fake_feats, real_feats):
        for ff, rf in zip(ff_scale, rf_scale):
            total += F.l1_loss(ff, rf.detach())
            n += 1
    return total / max(1, n)


def r1_penalty(discriminator_single_scale, corrupted, real, mask):
    # R1 regularizácia penalizuje príliš ostré gradienty diskriminátora
    # vzhľadom na reálny vstup.
    real = real.requires_grad_(True)
    pred = discriminator_single_scale(corrupted, real, mask)
    grad = torch.autograd.grad(pred.sum(), real, create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()
