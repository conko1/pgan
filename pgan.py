import math
import random
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# Pomocné funkcie na načítanie, prevod rozsahu hodnôt a základné spracovanie obrázkov a masiek.

def load_gray(path: str) -> torch.Tensor:
    # Načíta obrázok v grayscale a prevedie ho na tensor v rozsahu <0, 1>.
    return TF.to_tensor(Image.open(path).convert("L"))


def load_binary_mask(path: str) -> torch.Tensor:
    # Načíta masku a binarizuje ju, aby mala len 0/1 hodnoty.
    return (TF.to_tensor(Image.open(path).convert("L")) >= 1.0 / 255.0).float()


def to_model_range(x: torch.Tensor) -> torch.Tensor:
    # Prevedie dáta z rozsahu <0, 1> do rozsahu <-1, 1> pre model.
    return x * 2.0 - 1.0


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    # Prevedie výstup modelu z rozsahu <-1, 1> späť do <0, 1>.
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def save_image(x: torch.Tensor, path):
    # Uloží tensor ako obrázok; ak je v modelovom rozsahu, najprv ho prevedie.
    x = x.detach().cpu()
    if x.min() < 0:
        x = to_image_range(x)
    TF.to_pil_image(x.clamp(0, 1)).save(path)


def save_mask(mask: torch.Tensor, path):
    # Uloží binárnu masku ako obrázok.
    TF.to_pil_image((mask.detach().cpu() > 0.5).float()).save(path)


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
    # Pridá padding okolo CHW tensoru; pre masky sa dá použiť aj konštantná hodnota.
    if mode == "constant":
        return F.pad(x, (pad, pad, pad, pad), mode=mode, value=value)
    return F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode=mode).squeeze(0)


def center_crop_chw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # Vyreže stredový crop požadovanej veľkosti.
    _, H, W = x.shape
    top = max(0, (H - h) // 2)
    left = max(0, (W - w) // 2)
    return x[:, top:top + h, left:left + w]


def lesion_crop(*tensors: torch.Tensor, mask: torch.Tensor, crop_size: int):
    # Vytvorí crop zameraný na léziu podľa masky.
    # Ak maska neobsahuje nič, použije sa stredový crop.
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

    cropped = [img[:, y:y + crop_size, x:x + crop_size] for img in imgs]
    return (*cropped, mask[:, y:y + crop_size, x:x + crop_size])

# Spoločné augmentácie pre obrázky a masky, aby zostali navzájom zarovnané.

def _shared_rotate_translate(*imgs: torch.Tensor, mask: torch.Tensor):
    # Aplikuje rovnakú rotáciu a posun na obrázky aj masku.
    # Pri maske sa používa nearest interpolácia, aby ostala binárna.
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

    out_imgs = [img[:, top:top + h, left:left + w] for img in out_imgs]
    mask = mask[:, top:top + h, left:left + w]
    return (*out_imgs, (mask > 0).float())


def _shared_intensity_noise(*imgs: torch.Tensor):
    # Mení intenzitu, kontrast a občas pridáva jemný šum.
    # Úpravy sa aplikujú rovnako na všetky vstupné obrázky.
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
    # Zabalí augmentácie do jednej funkcie pre sample.
    # Podporuje režim len s real+mask aj režim s real+corrupted+mask.
    items = [real] if corrupted is None else [real, corrupted]

    if random.random() < 0.5:
        items = [TF.hflip(x) for x in items]
        mask = TF.hflip(mask)

    if corrupted is None:
        real, mask = _shared_rotate_translate(items[0], mask=mask)
        real, = _shared_intensity_noise(real)
        return real, mask

    real, corrupted, mask = _shared_rotate_translate(items[0], items[1], mask=mask)
    real, corrupted = _shared_intensity_noise(real, corrupted)
    return real, corrupted, mask


# Dataset pripravuje cropy, masky a corrupted vstupy pre trénovanie inpainting modelu.

class MammogramInpaintDataset(Dataset):
    def __init__(self, healthy_dir, mask_dir, corrupted_dir=None, crop_size=256, augment=True):
        # Inicializácia ciest, parametrov datasetu a kontrola konzistencie súborov.
        self.healthy_dir = Path(healthy_dir)
        self.mask_dir = Path(mask_dir)
        self.corrupted_dir = Path(corrupted_dir) if corrupted_dir is not None else None
        self.crop_size = crop_size
        self.augment = augment

        self.healthy_paths = sorted(self.healthy_dir.glob("*.jpg"))
        if not self.healthy_paths:
            raise ValueError("No .jpg files found in healthy_dir")

        missing_masks = []
        missing_corrupted = []
        for p in self.healthy_paths:
            mp = self.mask_dir / f"{p.stem}_mask.png"
            if not mp.exists():
                missing_masks.append(mp.name)
            if self.corrupted_dir is not None:
                cp = self.corrupted_dir / p.name
                if not cp.exists():
                    missing_corrupted.append(cp.name)

        if missing_masks:
            raise ValueError(f"Missing masks, first few: {missing_masks[:5]}")
        if missing_corrupted:
            raise ValueError(f"Missing corrupted images, first few: {missing_corrupted[:5]}")

    def __len__(self):
        # Vráti počet vzoriek v datasete.
        return len(self.healthy_paths)

    def __getitem__(self, idx):
        # Načíta sample, pripraví crop okolo masky a vráti vstupy pre model.
        img_path = self.healthy_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}_mask.png"

        real = load_gray(str(img_path))
        mask = load_binary_mask(str(mask_path))
        corrupted = load_gray(str(self.corrupted_dir / img_path.name)) if self.corrupted_dir is not None else None

        if corrupted is None:
            real, mask = ensure_min_size(real, mask, min_size=self.crop_size)
            if self.augment:
                real, mask = augment_sample(real, mask)
            real, mask = lesion_crop(real, mask=mask, crop_size=self.crop_size)
        else:
            real, corrupted, mask = ensure_min_size(real, corrupted, mask, min_size=self.crop_size)
            if self.augment:
                real, corrupted, mask = augment_sample(real, mask, corrupted)
            real, corrupted, mask = lesion_crop(real, corrupted, mask=mask, crop_size=self.crop_size)

        target_mask = mask.clone()

        # Ak neexistuje corrupted vstup alebo je maska prázdna,
        # synteticky sa "vymaže" oblasť masky z real obrázka.
        if corrupted is None or target_mask.sum().item() == 0:
            corrupted = real * (1.0 - target_mask)

        return {
            "corrupted_crop": to_model_range(corrupted),
            "target_mask": target_mask,
            "real_crop": to_model_range(real),
            "image_name": img_path.stem,
        }


# Základné stavebné bloky generátora a diskriminátora.

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, act="relu"):
        # Konvolučný blok: reflection padding + conv + instance norm + aktivácia.
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        ]
        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act != "none":
            raise ValueError(act)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # Prechod vstupu cez celý blok.
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        # Reziduálny blok stabilizuje učenie a pomáha zachovať informáciu.
        super().__init__()
        self.c1 = ConvNormAct(ch, ch, act="relu")
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.c2 = ConvNormAct(ch, ch, act="none")

    def forward(self, x):
        # Skip connection: vstup sa pripočíta k transformovanému výstupu.
        return x + self.c2(self.do(self.c1(x)))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        # Downsampling blok znižuje rozlíšenie a zvyšuje počet kanálov.
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # Aplikuje downsampling blok.
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        # Upsampling blok obnovuje rozlíšenie v decoder časti siete.
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        # Aplikuje upsampling blok.
        return self.block(x)


def crop_to(src, ref):
    # Zarovná feature mapu na veľkosť referenčnej mapy pomocou stredového orezu.
    _, _, h, w = src.shape
    _, _, rh, rw = ref.shape
    if h == rh and w == rw:
        return src
    top = max((h - rh) // 2, 0)
    left = max((w - rw) // 2, 0)
    return src[:, :, top:top + rh, left:left + rw]


# U-Net generátor s reziduálnym bottleneckom pre inpainting.

class ResidualUNetGenerator(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=64, n_res=4):
        # Encoder-decoder architektúra so skip connections.
        # Vstupom je poškodený obrázok + maska.
        super().__init__()
        self.d1 = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d2 = Down(base, base * 2)
        self.d3 = Down(base * 2, base * 4)
        self.d4 = Down(base * 4, base * 8)
        self.mid = nn.Sequential(*[ResBlock(base * 8, dropout=0.1) for _ in range(n_res)])
        self.u1 = Up(base * 8, base * 4, dropout=0.1)
        self.u2 = Up(base * 8, base * 2)
        self.u3 = Up(base * 4, base)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, corrupted, mask):
        # Predikuje obsah chýbajúcej oblasti na základe vstupu a masky.
        x = torch.cat([corrupted, mask], dim=1)
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        b = self.mid(d4)

        u1 = torch.cat([crop_to(self.u1(b), d3), d3], dim=1)
        u2 = torch.cat([crop_to(self.u2(u1), d2), d2], dim=1)
        u3 = torch.cat([crop_to(self.u3(u2), d1), d1], dim=1)
        return self.out(u3)

    def compose(self, corrupted, mask):
        # Zloží finálny obrázok: mimo masky ostáva vstup,
        # vo vnútri masky sa použije predikcia generátora.
        pred_hole = self.forward(corrupted, mask)
        fake = (corrupted * (1.0 - mask) + pred_hole * mask).clamp(-1.0, 1.0)
        return fake, pred_hole


# PatchGAN diskriminátor v dvoch mierkach.

class SNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        # Konvolučný blok so spectral normalization pre stabilnejší tréning GAN.
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # Prechod vstupu cez diskriminačný blok.
        return self.block(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        # Hodnotí realizmus lokálnych patchov namiesto celého obrázka naraz.
        super().__init__()
        self.b1 = SNBlock(in_ch, base, stride=2)
        self.b2 = SNBlock(base, base * 2, stride=2)
        self.b3 = SNBlock(base * 2, base * 4, stride=2)
        self.b4 = SNBlock(base * 4, base * 8, stride=1)
        self.out = spectral_norm(nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, corrupted, img, mask, return_features=False):
        # Diskriminátor dostáva corrupted vstup, výsledný obrázok a masku.
        # Voliteľne vracia aj intermediate features pre feature matching loss.
        x = torch.cat([corrupted, img, mask], dim=1)
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        logits = self.out(f4)
        if return_features:
            return logits, [f1, f2, f3, f4]
        return logits


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        # Dva PatchGAN diskriminátory pracujú na rôznych mierkach.
        super().__init__()
        self.d1 = PatchDiscriminator()
        self.d2 = PatchDiscriminator()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, corrupted, img, mask, return_features=False):
        # Prvá vetva pracuje v pôvodnom rozlíšení, druhá v zmenšenom.
        corrupted2, img2, mask2 = self.pool(corrupted), self.pool(img), self.pool(mask)

        if return_features:
            out1, feat1 = self.d1(corrupted, img, mask, return_features=True)
            out2, feat2 = self.d2(corrupted2, img2, mask2, return_features=True)
            return [out1, out2], [feat1, feat2]

        out1 = self.d1(corrupted, img, mask)
        out2 = self.d2(corrupted2, img2, mask2)
        return [out1, out2]


# Straty pre rekonštrukciu, adversarial učenie a stabilizáciu.

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


# Funkcie na vytvorenie DataLoadera, tréning modelu
# a generovanie výstupov.

def build_loader(
    healthy_dir,
    mask_dir,
    corrupted_dir=None,
    crop_size=256,
    batch_size=8,
    num_workers=0,
    augment=True,
):
    # Vytvorí dataset a DataLoader pre tréning alebo inferenciu.
    ds = MammogramInpaintDataset(
        healthy_dir=healthy_dir,
        mask_dir=mask_dir,
        corrupted_dir=corrupted_dir,
        crop_size=crop_size,
        augment=augment,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=augment,
    )


def train(
    healthy_dir,
    mask_dir,
    corrupted_dir=None,
    epochs=20,
    batch_size=8,
    crop_size=256,
    save_path="generator.pt",
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
):
    # Hlavná tréningová slučka pre GAN inpainting model.
    # Striedavo sa učí diskriminátor a generátor.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dl = build_loader(healthy_dir, mask_dir, corrupted_dir, crop_size, batch_size, num_workers, augment=True)

    G = ResidualUNetGenerator().to(device)
    D = MultiScaleDiscriminator().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.9))

    amp_enabled = device.startswith("cuda")
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    step_global = 0

    for epoch in range(epochs):
        G.train()
        D.train()

        for step, batch in enumerate(dl):
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

    # Po tréningu sa uloží naučený generátor.
    torch.save(G.state_dict(), save_path)
    print(f"Saved generator to {save_path}")
    return G, D


def generate(
    healthy_dir,
    mask_dir,
    corrupted_dir=None,
    model_path="generator.pt",
    output_dir="generated",
    crop_size=256,
    device=None,
):
    # Načíta natrénovaný generátor a vytvorí výstupné obrázky pre všetky vstupy.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = ResidualUNetGenerator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    healthy_paths = sorted(Path(healthy_dir).glob("*.jpg"))
    if not healthy_paths:
        raise ValueError("No .png files found in healthy_dir")

    with torch.no_grad():
        for img_path in healthy_paths:
            # Pripraví vstupný crop a masku pre generovanie.
            real = load_gray(str(img_path))
            mask = load_binary_mask(str(Path(mask_dir) / f"{img_path.stem}_mask.png"))
            corrupted = (
                load_gray(str(Path(corrupted_dir) / img_path.name))
                if corrupted_dir is not None
                else real.clone()
            )

            real, corrupted, mask = ensure_min_size(real, corrupted, mask, min_size=crop_size)
            real, corrupted, mask = lesion_crop(real, corrupted, mask=mask, crop_size=crop_size)

            corrupted_batch = to_model_range(corrupted).unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0).to(device)

            # Mimo masky ostáva vstup, vnútro masky doplní model.
            fake_batch, _ = G.compose(corrupted_batch, mask_batch)
            fake = to_image_range(fake_batch[0].cpu())

            # Uloží real crop, masku a vygenerovaný výsledok.
            base = img_path.stem
            save_image(real.cpu(), out_dir / f"{base}_real_crop.png")
            save_mask(mask, out_dir / f"{base}_target_mask.png")
            save_image(fake, out_dir / f"{base}_fake.png")


if __name__ == "__main__":
    # Príklad tréningu modelu.
    # train(
    #     healthy_dir="images_vindr",
    #     mask_dir="masks_vindr",
    #     corrupted_dir="images_vindr_inpainted_bigger_3",
    #     epochs=15,
    #     batch_size=8,
    #     crop_size=256,
    #     save_path="generator_classic-inpainted_bigger_3_out16.pt",
    # )

    # Príklad generovania syntetických výstupov z natrénovaného modelu.
    generate(
        healthy_dir="images_vindr_removed",
        mask_dir="images_vindr_removed_masks",
        model_path="latest.pt",
        output_dir="synthetic_dataset_vindr_removed",
        crop_size=1024,
    )