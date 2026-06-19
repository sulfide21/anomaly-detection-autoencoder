"""
Minimal convolutional autoencoder over a DINOv2 feature map.

Design goal: reconstruct *normal* features well while staying UNABLE to copy
anomalies (the identity-shortcut problem). DINOv2 features are strong enough
that a powerful decoder would just copy its input verbatim, defects included.
Two cheap anti-shortcut mechanisms, both deliberate:

  1. a spatial + channel BOTTLENECK (16x16x768 -> 8x8x128, ~24x squeeze) so the
     model must summarize, not memorize — and it reconstructs each location from
     spatial context, not from itself;
  2. norm-scaled feature NOISE in training (a denoising objective) so it learns
     the normal-feature manifold instead of an identity map.

This is the feature-space analog of our VAE's bottleneck — the same lesson, one
abstraction level up.
"""
import torch
import torch.nn as nn


class FeatRecon(nn.Module):
    def __init__(self, in_dim=768, mid=384, bottleneck=128, noise_std=0.2):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),  # g -> g/2
            nn.Conv2d(256, bottleneck, 3, padding=1), nn.BatchNorm2d(bottleneck), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),  # g/2 -> g
            nn.Conv2d(256, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, in_dim, 3, padding=1),
        )

    def add_noise(self, f):
        if self.training and self.noise_std > 0:
            # noise scaled by each token's feature norm (UniAD-style jitter)
            norm = f.norm(dim=1, keepdim=True) / f.shape[1]
            f = f + torch.randn_like(f) * norm * self.noise_std
        return f

    def forward(self, f):
        return self.decoder(self.encoder(self.add_noise(f)))
