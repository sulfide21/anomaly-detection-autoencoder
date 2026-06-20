"""
Convolutional Variational Autoencoder (VAE).

Difference from a plain autoencoder: the encoder does not output a single code,
it outputs a DISTRIBUTION (a mean `mu` and a log-variance `logvar`). We sample
from it (the "reparameterisation trick") and decode. A KL term then pulls that
distribution toward a standard normal, which keeps the latent space smooth.

Heads-up for anomaly detection: at TEST time we decode `mu` directly (no random
sampling) so the reconstruction is deterministic and the anomaly score is stable.

img_size must be divisible by 32 (there are 5 stride-2 down/up-sampling stages).
"""
import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(self, in_channels: int = 1, img_size: int = 256,
                 latent_dim: int = 256, base: int = 32):
        super().__init__()
        assert img_size % 32 == 0, "img_size must be divisible by 32"
        self.in_channels = in_channels
        self.img_size = img_size
        self.feat = img_size // 32                       # spatial size at bottleneck
        chans = [in_channels, base, base * 2, base * 4, base * 8, base * 8]

        # ---- encoder: 5 x (conv stride 2) ----
        enc = []
        for i in range(5):
            enc += [
                nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1),
                nn.BatchNorm2d(chans[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.encoder = nn.Sequential(*enc)

        self._bottleneck_c = chans[-1]
        self.flat_dim = chans[-1] * self.feat * self.feat
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # ---- decoder: 5 x (transpose-conv stride 2) ----
        rch = list(reversed(chans))                      # base*8 .. in_channels
        dec = []
        for i in range(5):
            dec += [nn.ConvTranspose2d(rch[i], rch[i + 1], 4, stride=2, padding=1)]
            if i < 4:
                dec += [nn.BatchNorm2d(rch[i + 1]), nn.LeakyReLU(0.2, inplace=True)]
        dec += [nn.Sigmoid()]                            # output in [0, 1]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self._bottleneck_c, self.feat, self.feat)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x):
        """Deterministic reconstruction (decode the mean) — used at test time."""
        mu, _ = self.encode(x)
        return self.decode(mu)
