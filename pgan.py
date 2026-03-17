import os
import random
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ----------------------------
# utils
# ----------------------------

def load_grayscale(path):
    return Image.open(path).convert("L")


def pil_to_tensor(img):
    return TF.to_tensor(img)


def tensor_to_mask(x, thr=0.5):
    return (x > thr).float()


def save_tensor_image(tensor, path):
    tensor = tensor.detach().cpu().clamp(0, 1)
    img = TF.to_pil_image(tensor)
    img.save(path)


def random_rotate(img_t, mask_t):
    angle = random.uniform(-15, 15)
    img_t = TF.rotate(
        img_t,
        angle,
        interpolation=TF.InterpolationMode.BILINEAR
    )

    mask_t = TF.rotate(
        mask_t,
        angle,
        interpolation=TF.InterpolationMode.NEAREST
    )
    return img_t, mask_t


def find_mask_bbox(mask):
    ys, xs = torch.where(mask[0] > 0.5)
    if len(xs) == 0:
        return None
    y1, y2 = ys.min().item(), ys.max().item()
    x1, x2 = xs.min().item(), xs.max().item()
    return x1, y1, x2, y2


def crop_with_pad(img, x, y, size):
    c, h, w = img.shape
    x2 = x + size
    y2 = y + size

    pad_l = max(0, -x)
    pad_t = max(0, -y)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    if pad_l or pad_t or pad_r or pad_b:
        img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)
        x += pad_l
        y += pad_t

    return img[:, y:y+size, x:x+size]


def get_breast_mask(img, thr=0.05):
    return (img > thr).float()


def crop_is_inside_breast(breast_crop, min_breast_fraction=0.95):
    return breast_crop.mean().item() >= min_breast_fraction


def random_healthy_crop(img, mask, crop_size, tries=100, min_breast_fraction=0.95):
    _, h, w = img.shape

    breast_mask = get_breast_mask(img)

    for _ in range(tries):
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)

        crop_img = img[:, y:y+crop_size, x:x+crop_size]
        crop_mask = mask[:, y:y+crop_size, x:x+crop_size]
        crop_breast = breast_mask[:, y:y+crop_size, x:x+crop_size]

        no_lesion = crop_mask.sum() == 0
        enough_breast = crop_is_inside_breast(
            crop_breast,
            min_breast_fraction=min_breast_fraction
        )

        if no_lesion and enough_breast:
            return crop_img, crop_mask

    best = None
    best_score = float("inf")

    for _ in range(tries):
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)

        crop_img = img[:, y:y+crop_size, x:x+crop_size]
        crop_mask = mask[:, y:y+crop_size, x:x+crop_size]
        crop_breast = breast_mask[:, y:y+crop_size, x:x+crop_size]

        lesion_score = crop_mask.sum().item()
        breast_penalty = 1.0 - crop_breast.mean().item()

        score = lesion_score + 1000.0 * breast_penalty

        if score < best_score:
            best_score = score
            best = (crop_img, crop_mask)

    return best


def lesion_center_crop(img, mask, crop_size):
    bbox = find_mask_bbox(mask)
    _, h, w = img.shape

    if bbox is None:
        x = max(0, (w - crop_size) // 2)
        y = max(0, (h - crop_size) // 2)
        return img[:, y:y+crop_size, x:x+crop_size], mask[:, y:y+crop_size, x:x+crop_size]

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x = cx - crop_size // 2
    y = cy - crop_size // 2
    return crop_with_pad(img, x, y, crop_size), crop_with_pad(mask, x, y, crop_size)


# ----------------------------
# dataset
# ----------------------------

class MammogramGanDataset(Dataset):
    def __init__(self, image_dir, mask_dir, crop_size=256, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.crop_size = crop_size
        self.augment = augment

        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
        if not self.image_paths:
            raise ValueError("No .jpg files found in image_dir")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + "_mask.png")

        img = pil_to_tensor(load_grayscale(str(img_path)))
        mask = pil_to_tensor(load_grayscale(str(mask_path)))
        mask = tensor_to_mask(mask)

        if self.augment:
            img, mask = random_rotate(img, mask)

        real_crop, real_mask = lesion_center_crop(img, mask, self.crop_size)
        healthy_crop, _ = random_healthy_crop(img, mask, self.crop_size)

        target_mask = real_mask.clone()

        return {
            "healthy_crop": healthy_crop,
            "target_mask": target_mask,
            "real_crop": real_crop,
            "real_mask": real_mask,
            "image_name": img_path.stem,
        }


# ----------------------------
# generator
# ----------------------------

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
        self.d1 = Down(in_ch, 64, norm=False)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up(512, 256)
        self.u2 = Up(512, 128)
        self.u3 = Up(256, 64)
        self.u4 = nn.ConvTranspose2d(128, out_ch, 4, 2, 1)

    def forward(self, healthy, target_mask):
        x = torch.cat([healthy, target_mask], dim=1)

        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        u1 = self.u1(d4)
        u1 = torch.cat([u1, d3], dim=1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d2], dim=1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d1], dim=1)

        out = self.u4(u3)
        out = torch.sigmoid(out)
        return out


# ----------------------------
# discriminator
# ----------------------------

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        return self.net(x)


# ----------------------------
# training
# ----------------------------

def train_minimal(
    image_dir,
    mask_dir,
    epochs=5,
    batch_size=4,
    crop_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="generator.pt",
):
    ds = MammogramGanDataset(image_dir, mask_dir, crop_size=crop_size, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    for epoch in range(epochs):
        for step, batch in enumerate(dl):
            healthy = batch["healthy_crop"].to(device)
            target_mask = batch["target_mask"].to(device)
            real = batch["real_crop"].to(device)
            real_mask = batch["real_mask"].to(device)

            fake = G(healthy, target_mask).detach()

            pred_real = D(real, real_mask)
            pred_fake = D(fake, target_mask)

            loss_d_real = bce(pred_real, torch.ones_like(pred_real))
            loss_d_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            fake = G(healthy, target_mask)
            pred_fake = D(fake, target_mask)

            loss_g_adv = bce(pred_fake, torch.ones_like(pred_fake))

            bg = 1.0 - target_mask
            loss_bg = l1(fake * bg, healthy * bg)

            loss_lesion = l1(fake * target_mask, real * target_mask)

            loss_g = loss_g_adv + 50.0 * loss_bg + 20.0 * loss_lesion

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            if step % 20 == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} step {step:04d} | "
                    f"loss_d={loss_d.item():.4f} "
                    f"loss_g={loss_g.item():.4f} "
                    f"adv={loss_g_adv.item():.4f} "
                    f"bg={loss_bg.item():.4f} "
                    f"lesion={loss_lesion.item():.4f}"
                )

    torch.save(G.state_dict(), save_path)
    print(f"saved generator to {save_path}")

    return G, D


# ----------------------------
# generation
# ----------------------------

def generate_from_directory(
    image_dir,
    mask_dir,
    model_path="generator.pt",
    output_dir="generated",
    crop_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    image_paths = sorted(list(Path(image_dir).glob("*.jpg")))
    if not image_paths:
        raise ValueError("No .jpg files found in image_dir")

    with torch.no_grad():
        for img_path in image_paths:
            mask_path = Path(mask_dir) / (img_path.stem + "_mask.png")

            img = pil_to_tensor(load_grayscale(str(img_path)))
            mask = pil_to_tensor(load_grayscale(str(mask_path)))
            mask = tensor_to_mask(mask)

            healthy_crop, _ = random_healthy_crop(img, mask, crop_size)
            _, real_mask = lesion_center_crop(img, mask, crop_size)

            healthy_batch = healthy_crop.unsqueeze(0).to(device)
            target_mask_batch = real_mask.unsqueeze(0).to(device)

            fake = G(healthy_batch, target_mask_batch)[0].cpu()

            overlay = healthy_crop * (1.0 - real_mask) + fake * real_mask

            base = img_path.stem
            save_tensor_image(healthy_crop, output_dir / f"{base}_healthy.jpg")
            save_tensor_image(real_mask, output_dir / f"{base}_target_mask.jpg")
            save_tensor_image(fake, output_dir / f"{base}_fake.jpg")
            save_tensor_image(overlay, output_dir / f"{base}_overlay.jpg")

            print(f"generated for {img_path.name}")


if __name__ == "__main__":
    G, D = train_minimal(
        image_dir="images_vindr",
        mask_dir="masks_vindr",
        epochs=15,
        batch_size=8,
        crop_size=256,
        save_path="generator.pt",
    )

    generate_from_directory(
        image_dir="images_vindr",
        mask_dir="masks_vindr",
        model_path="generator.pt",
        output_dir="generated_7",
        crop_size=512,
    )