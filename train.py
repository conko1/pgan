from __future__ import annotations

from pathlib import Path

import torch

from data.loader import build_loader
from losses import (
    boundary_ring,
    d_hinge,
    feature_matching,
    g_hinge,
    masked_l1,
    r1_penalty,
)
from metrics import evaluate_and_save_samples
from models.discriminator import MultiScaleDiscriminator
from models.generator import ResidualUNetGenerator

def train(
    train_healthy_dir,
    train_mask_dir,
    train_corrupted_dir=None,
    val_healthy_dir=None,
    val_mask_dir=None,
    val_corrupted_dir=None,
    test_healthy_dir=None,
    test_mask_dir=None,
    test_corrupted_dir=None,
    epochs=20,
    batch_size=8,
    eval_batch_size=None,
    crop_size=256,
    save_dir="runs/pgan_run",
    best_metric_name="lpips",
    device=None,
    num_workers=0,
    lr_g=1e-4,
    lr_d=1e-4,
    lambda_hole=20.0,
    lambda_bg=5.0,
    lambda_boundary=15.0,
    lambda_fm=5.0,
    r1_gamma=5.0,
    r1_every=8,
    sample_count=8,
):
    # Hlavná tréningová slučka pre GAN inpainting model.
    # Trénuje na train split-e, vyhodnocuje na val split-e a na konci voliteľne aj na test split-e.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_batch_size = eval_batch_size or batch_size

    save_dir = Path(save_dir)
    ckpt_dir = save_dir / "checkpoints"
    vis_dir = save_dir / "visuals"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(
        train_healthy_dir,
        train_mask_dir,
        train_corrupted_dir,
        crop_size,
        batch_size,
        num_workers,
        augment=True,
        shuffle=True,
        deterministic_eval=False,
    )

    val_loader = None
    if val_healthy_dir and val_mask_dir:
        val_loader = build_loader(
            val_healthy_dir,
            val_mask_dir,
            val_corrupted_dir,
            crop_size,
            eval_batch_size,
            num_workers,
            augment=False,
            shuffle=False,
            deterministic_eval=True,
        )

    test_loader = None
    if test_healthy_dir and test_mask_dir:
        test_loader = build_loader(
            test_healthy_dir,
            test_mask_dir,
            test_corrupted_dir,
            crop_size,
            eval_batch_size,
            num_workers,
            augment=False,
            shuffle=False,
            deterministic_eval=True,
        )

    G = ResidualUNetGenerator().to(device)
    D = MultiScaleDiscriminator().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.9))

    amp_enabled = device.startswith("cuda")
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if best_metric_name not in {"lpips", "kid_mean", "ms_ssim"}:
        raise ValueError("best_metric_name must be one of: 'lpips', 'kid_mean', 'ms_ssim'")

    best_metric = float("inf") if best_metric_name in {"lpips", "kid_mean"} else -float("inf")
    history = []
    step_global = 0

    for epoch in range(epochs):
        G.train()
        D.train()

        for step, batch in enumerate(train_loader):
            corrupted = batch["corrupted_crop"].to(device)
            mask = batch["target_mask"].to(device)
            real = batch["real_crop"].to(device)

            # ---- D ----
            # Diskriminátor sa učí rozlišovať real a fake obrázky.
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                fake, _ = G.compose(corrupted, mask)
                pred_real = D(corrupted, real, mask)
                pred_fake = D(corrupted, fake.detach(), mask)
                loss_d = sum(d_hinge(pr, pf) for pr, pf in zip(pred_real, pred_fake)) / len(pred_real)

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            loss_r1 = torch.tensor(0.0, device=device)
            if step_global % r1_every == 0:
                # Periodická R1 regularizácia pre stabilizáciu diskriminátora.
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=False):
                    loss_r1 = 0.5 * r1_gamma * r1_penalty(D.d1, corrupted.float(), real.float(), mask.float())
                loss_r1.backward()
                opt_d.step()

            # ---- G ----
            # Generátor sa učí vyplniť maskovanú oblasť realisticky a konzistentne.
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                fake, _ = G.compose(corrupted, mask)
                pred_fake, fake_feats = D(corrupted, fake, mask, return_features=True)
                _, real_feats = D(corrupted, real, mask, return_features=True)

                loss_adv = sum(g_hinge(pf) for pf in pred_fake) / len(pred_fake)
                bg = 1.0 - mask
                boundary = boundary_ring(mask, k=5)
                loss_hole = masked_l1(fake, real, mask, min_pixels=32.0)
                loss_bg = masked_l1(fake, real, bg, min_pixels=256.0)
                loss_boundary = masked_l1(fake, real, boundary, min_pixels=32.0)
                loss_fm = feature_matching(fake_feats, real_feats)

                # Finálna strata kombinuje GAN loss a viacero rekonštrukčných zložiek.
                loss_g = (
                    loss_adv
                    + lambda_hole * loss_hole
                    + lambda_bg * loss_bg
                    + lambda_boundary * loss_boundary
                    + lambda_fm * loss_fm
                )

            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            if step % 20 == 0:
                # Priebežný výpis metrík počas tréningu.
                print(
                    f"epoch {epoch + 1}/{epochs} step {step:04d} | "
                    f"D={loss_d.item():.4f} R1={loss_r1.item():.4f} G={loss_g.item():.4f} "
                    f"adv={loss_adv.item():.4f} hole={loss_hole.item():.4f} "
                    f"bg={loss_bg.item():.4f} boundary={loss_boundary.item():.4f} fm={loss_fm.item():.4f}"
                )

            step_global += 1

        # checkpoint po každej epoche
        last_path = ckpt_dir / "last_generator.pt"
        torch.save(G.state_dict(), last_path)

        epoch_record = {"epoch": epoch + 1}

        if val_loader is not None:
            do_eval = (epoch < 5) or (epoch > epochs - 5) or (epoch % 5 == 0)
            if do_eval:
                val_metrics = evaluate_and_save_samples(
                    G,
                    val_loader,
                    device=device,
                    output_dir=vis_dir,
                    epoch=epoch + 1,
                    split_name="val",
                    max_samples=sample_count,
                    kid_subsets=10,
                    kid_subset_size=min(24, len(val_loader.dataset)),
                )
                epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
                print(
                    f"[VAL] epoch {epoch + 1:03d} | "
                    f"LPIPS={val_metrics['lpips']:.4f} "
                    f"KID={val_metrics['kid_mean']:.4f}±{val_metrics['kid_std']:.4f} "
                    f"MS-SSIM={val_metrics['ms_ssim']:.4f}"
                )

                current_metric = val_metrics[best_metric_name]
                is_better = current_metric < best_metric if best_metric_name in {"lpips", "kid_mean"} else current_metric > best_metric
                if is_better:
                    best_metric = current_metric
                    best_path = ckpt_dir / f"best_by_{best_metric_name}.pt"
                    torch.save(G.state_dict(), best_path)
                    print(f"Saved new best model to {best_path} ({best_metric_name}={current_metric:.6f})")

        history.append(epoch_record)

    # po tréningu vždy ulož aj finálny model s názvom final_generator.pt
    final_path = ckpt_dir / "final_generator.pt"
    torch.save(G.state_dict(), final_path)
    print(f"Saved final generator to {final_path}")

    # Ak je k dispozícii test set, vyhodnoť best checkpoint (ak existuje) alebo final checkpoint.
    test_metrics = None
    if test_loader is not None:
        best_path = ckpt_dir / f"best_by_{best_metric_name}.pt"
        eval_path = best_path if best_path.exists() else final_path
        G.load_state_dict(torch.load(eval_path, map_location=device))
        G.eval()
        test_metrics = evaluate_and_save_samples(
            G,
            test_loader,
            device=device,
            output_dir=vis_dir,
            epoch=epochs,
            split_name="test",
            max_samples=sample_count,
            kid_subsets=10,
            kid_subset_size=min(24, len(test_loader.dataset)),
        )
        print(
            f"[TEST] using {eval_path.name} | "
            f"LPIPS={test_metrics['lpips']:.4f} "
            f"KID={test_metrics['kid_mean']:.4f}±{test_metrics['kid_std']:.4f} "
            f"MS-SSIM={test_metrics['ms_ssim']:.4f}"
        )

    return {
        "generator": G,
        "discriminator": D,
        "history": history,
        "test_metrics": test_metrics,
        "save_dir": str(save_dir),
    }
