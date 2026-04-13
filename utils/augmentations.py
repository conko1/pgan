from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def ensure_min_size(*tensors: torch.Tensor, min_size: int):
    # Zabezpečí, aby všetky vstupy mali aspoň minimálny rozmer.
    # Masky sa škálujú nearest interpoláciou, obrázky bilineárne.
    h, w = tensors[0].shape[-2:]
    if h >= min_size and w >= min_size:
        return tensors

    scale = max(min_size / h, min_size / w)
    nh = int(math.ceil(h * scale))
    nw = int(math.ceil(w * scale))

    out = []
    for t in tensors:
        is_mask = bool(torch.all((t == 0) | (t == 1)))
        interp = InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
        t = TF.resize(t, [nh, nw], interpolation=interp, antialias=not is_mask)
        out.append((t > 0.5).float() if is_mask else t)
    return tuple(out)


def pad_chw(x: torch.Tensor, pad: int, mode: str = "reflect", value: float = 0.0) -> torch.Tensor:
    # Pad pre CHW tensor. Reflect padding vyžaduje batch dim, preto unsqueeze.
    if mode == "constant":
        return F.pad(x, (pad, pad, pad, pad), mode=mode, value=value)
    return F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode=mode).squeeze(0)


def center_crop_chw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # Center crop pre CHW tensor.
    _, H, W = x.shape
    top = max(0, (H - h) // 2)
    left = max(0, (W - w) // 2)
    return x[:, top : top + h, left : left + w]


def deterministic_lesion_crop(*tensors: torch.Tensor, mask: torch.Tensor, crop_size: int):
    # Deterministický crop okolo reálne existujúceho pozitívneho pixla masky.
    # Ak je maska prázdna, fallbackne na center crop.
    tensors = ensure_min_size(*tensors, mask, min_size=crop_size)
    *imgs, mask = tensors
    _, h, w = mask.shape

    ys, xs = torch.where(mask[0] > 0.5)

    if len(xs) == 0:
        x = max(0, (w - crop_size) // 2)
        y = max(0, (h - crop_size) // 2)
    else:
        # deterministicky vyber jeden existujúci foreground pixel
        order = torch.argsort(ys * w + xs)
        mid_idx = order[len(order) // 2]

        cx = xs[mid_idx].item()
        cy = ys[mid_idx].item()

        x = cx - crop_size // 2
        y = cy - crop_size // 2

        x = max(0, min(x, w - crop_size))
        y = max(0, min(y, h - crop_size))

    cropped = [img[:, y:y + crop_size, x:x + crop_size] for img in imgs]
    cropped_mask = mask[:, y:y + crop_size, x:x + crop_size]

    return (*cropped, cropped_mask)


def lesion_crop(*tensors: torch.Tensor, mask: torch.Tensor, crop_size: int):
    # Vyberie crop okolo náhodného pixla v maske. Ak je maska prázdna, vyberie center crop.
    tensors = ensure_min_size(*tensors, mask, min_size=crop_size)
    *imgs, mask = tensors
    _, h, w = mask.shape

    ys, xs = torch.where(mask[0] > 0.5)

    if len(xs) == 0:
        x = max(0, (w - crop_size) // 2)
        y = max(0, (h - crop_size) // 2)
    else:
        i = random.randint(0, len(xs) - 1)
        cx = xs[i].item()
        cy = ys[i].item()
        x = cx - crop_size // 2 + random.randint(-8, 8)
        y = cy - crop_size // 2 + random.randint(-8, 8)
        x = max(0, min(x, w - crop_size))
        y = max(0, min(y, h - crop_size))

    cropped = [img[:, y : y + crop_size, x : x + crop_size] for img in imgs]
    return (*cropped, mask[:, y : y + crop_size, x : x + crop_size])


def _shared_rotate_translate(*imgs: torch.Tensor, mask: torch.Tensor):
    # Rovnaká rotácia + translácia pre obrázky aj masku.
    angle = random.uniform(-12.0, 12.0)
    max_shift = 8
    pad = 32
    h, w = imgs[0].shape[-2:]

    out_imgs = []
    for img in imgs:
        img = pad_chw(img, pad, mode="reflect")
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        out_imgs.append(center_crop_chw(img, h + 2 * max_shift, w + 2 * max_shift))

    mask = pad_chw(mask, pad, mode="constant", value=0.0)
    mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0.0)
    mask = center_crop_chw(mask, h + 2 * max_shift, w + 2 * max_shift)

    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    left = max_shift - tx
    top = max_shift - ty

    out_imgs = [img[:, top : top + h, left : left + w] for img in out_imgs]
    mask = mask[:, top : top + h, left : left + w]
    return (*out_imgs, (mask > 0).float())


def _shared_intensity_noise(*imgs: torch.Tensor):
    # Spoločné intenzitné zmeny a šum (jemné augmentácie).
    if random.random() < 0.2:
        gamma = random.uniform(0.97, 1.03)
        gain = random.uniform(0.98, 1.03)
        imgs = tuple(TF.adjust_gamma(img, gamma=gamma, gain=gain) for img in imgs)

    if random.random() < 0.2:
        contrast = random.uniform(0.95, 1.08)
        imgs = tuple(TF.adjust_contrast(img, contrast) for img in imgs)

    if random.random() < 0.05:
        noise = torch.randn_like(imgs[0]) * 0.003
        imgs = tuple((img + noise).clamp(0, 1) for img in imgs)

    return imgs


def augment_sample(real: torch.Tensor, mask: torch.Tensor, corrupted: torch.Tensor | None = None):
    # Augmentácie pre sample. Keď corrupted neexistuje, augmentujeme iba real + mask.
    items = [real] if corrupted is None else [real, corrupted]

    if random.random() < 0.5:
        items = [TF.hflip(x) for x in items]
        mask = TF.hflip(mask)

    if corrupted is None:
        real, mask = _shared_rotate_translate(items[0], mask=mask)
        (real,) = _shared_intensity_noise(real)
        return real, mask

    real, corrupted, mask = _shared_rotate_translate(items[0], items[1], mask=mask)
    real, corrupted = _shared_intensity_noise(real, corrupted)
    return real, corrupted, mask
