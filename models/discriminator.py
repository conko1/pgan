import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


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