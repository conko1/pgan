from __future__ import annotations

from pathlib import Path

import torch

from utils.image_io import save_triplet_grid, to_image_range

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
except ImportError as e:
    raise ImportError(
        "Tento projekt vyžaduje torchmetrics. Nainštaluj ho napr. `pip install torchmetrics`."
    ) from e


def _to_rgb_01(x_model_range: torch.Tensor) -> torch.Tensor:
    x = to_image_range(x_model_range)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.clamp(0.0, 1.0)


def evaluate_metrics(
    G,
    loader,
    device,
    kid_subsets=10,
    kid_subset_size=24,
):
    # Vyhodnotí model na loaderi pomocou LPIPS, KID a MS-SSIM.
    G.eval()

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    kid_metric = KernelInceptionDistance(
        subsets=kid_subsets,
        subset_size=kid_subset_size,
        normalize=True,
        reset_real_features=True,
    ).to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    lpips_sum = 0.0
    ms_ssim_sum = 0.0
    n_batches = 0
    n_images = 0

    with torch.no_grad():
        for batch in loader:
            corrupted = batch["corrupted_crop"].to(device)
            mask = batch["target_mask"].to(device)
            real = batch["real_crop"].to(device)

            fake, _ = G.compose(corrupted, mask)

            fake_01_rgb = _to_rgb_01(fake)
            real_01_rgb = _to_rgb_01(real)
            fake_01_gray = to_image_range(fake).clamp(0.0, 1.0)
            real_01_gray = to_image_range(real).clamp(0.0, 1.0)

            lpips_value = lpips_metric(fake_01_rgb, real_01_rgb)
            ms_ssim_value = ms_ssim_metric(fake_01_gray, real_01_gray)

            lpips_sum += lpips_value.item() * corrupted.size(0)
            ms_ssim_sum += ms_ssim_value.item() * corrupted.size(0)
            n_images += corrupted.size(0)
            n_batches += 1

            kid_metric.update(real_01_rgb, real=True)
            kid_metric.update(fake_01_rgb, real=False)

    kid_mean, kid_std = kid_metric.compute()
    metrics = {
        "lpips": lpips_sum / max(1, n_images),
        "ms_ssim": ms_ssim_sum / max(1, n_images),
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "num_images": n_images,
        "num_batches": n_batches,
    }

    lpips_metric.reset()
    kid_metric.reset()
    ms_ssim_metric.reset()
    return metrics


def save_eval_visuals(G, loader, device, output_dir, epoch, max_samples=8):
    # Uloží fixné ukážky z val/test množiny.
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    G.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            corrupted = batch["corrupted_crop"].to(device)
            mask = batch["target_mask"].to(device)
            real = batch["real_crop"].to(device)
            names = batch["image_name"]

            fake, _ = G.compose(corrupted, mask)
            for i in range(corrupted.size(0)):
                save_triplet_grid(
                    corrupted[i],
                    mask[i],
                    fake[i],
                    real[i],
                    epoch_dir / f"{saved:02d}_{names[i]}.png",
                )
                saved += 1
                if saved >= max_samples:
                    return


def evaluate_and_save_samples(
    G,
    loader,
    device,
    output_dir,
    epoch,
    split_name="val",
    max_samples=8,
    kid_subsets=10,
    kid_subset_size=24,
):
    metrics = evaluate_metrics(
        G,
        loader,
        device=device,
        kid_subsets=kid_subsets,
        kid_subset_size=kid_subset_size,
    )
    save_eval_visuals(
        G,
        loader,
        device=device,
        output_dir=Path(output_dir) / split_name,
        epoch=epoch,
        max_samples=max_samples,
    )
    return metrics
