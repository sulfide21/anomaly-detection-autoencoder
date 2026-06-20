"""
Structural Similarity (SSIM) — used two ways in this project:

  1. As a TRAINING loss   : loss = 1 - mean(SSIM(recon, input))
  2. As an ANOMALY score  : a per-pixel (1 - SSIM) map lights up where the
                            reconstruction structurally disagrees with the input.

Why SSIM instead of plain pixel error (MSE)? MSE over-reacts to tiny edge
shifts and ignores defects that keep the same brightness. SSIM compares the
*local structure* (luminance + contrast + correlation) inside a sliding
Gaussian window, which is what fixed the carpet/texture case in Bergmann 2019.

Everything here is differentiable, so it can be used directly as a loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """A (channels, 1, k, k) separable Gaussian kernel, normalised to sum 1."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]                 # (k, k)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    """Windowed SSIM. forward() returns (mean_ssim, ssim_map)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 channels: int = 1, val_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        # data is in [0, 1] (ToTensor), so val_range = 1.0
        self.C1 = (0.01 * val_range) ** 2
        self.C2 = (0.03 * val_range) ** 2
        self.register_buffer("window", _gaussian_window(window_size, sigma, channels))

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        # depthwise conv: one Gaussian per channel
        return F.conv2d(x, self.window, padding=self.window_size // 2,
                        groups=self.channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu_x = self._filter(x)
        mu_y = self._filter(y)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sigma_x2 = self._filter(x * x) - mu_x2
        sigma_y2 = self._filter(y * y) - mu_y2
        sigma_xy = self._filter(x * y) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2))
        return ssim_map.mean(), ssim_map
