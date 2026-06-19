"""
Frozen DINOv2 feature extractor.

Runs an image through a frozen DINOv2 ViT and returns its patch tokens as a
spatial feature map (B x C x g x g), where g = img_size / 14. These pretrained
features already "understand" texture and structure — that's the whole point:
we reconstruct *these* instead of raw pixels, so the model never has to repaint
a carpet weave (the thing our pixel-VAE couldn't do).

ImageNet normalization is applied inside here, so the dataset can keep feeding
plain [0,1] RGB tensors.
"""
import torch
import torch.nn as nn

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DinoV2Features(nn.Module):
    def __init__(self, model_name="dinov2_vitb14", img_size=224):
        super().__init__()
        assert img_size % 14 == 0, "DINOv2 patch size is 14; img_size must be a multiple"
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.embed_dim = self.backbone.embed_dim          # 768 for ViT-B/14
        self.grid = img_size // 14
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        # x: B x 3 x H x W in [0,1]
        x = (x - self.mean) / self.std
        out = self.backbone.forward_features(x)
        tokens = out["x_norm_patchtokens"]                # B x N x C
        b, n, c = tokens.shape
        g = int(n ** 0.5)
        return tokens.permute(0, 2, 1).reshape(b, c, g, g)  # B x C x g x g
