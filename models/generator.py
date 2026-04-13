from __future__ import annotations

import torch
import torch.nn as nn


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
