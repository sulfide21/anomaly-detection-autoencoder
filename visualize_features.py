"""
Visualize what DINOv2 'features' look like.

A single feature vector is 768 numbers (not visual). To SEE the feature map, we
take the 16x16 grid of 768-dim vectors, reduce 768 -> 3 dims with PCA, and paint
those 3 numbers as RGB. Result: patches with similar MEANING get similar colors —
so the carpet background is one color and a defect pops out as a different color.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

from dino_features import DinoV2Features

ROOT = Path("mvtec_anomaly_detection/carpet/test")
IMAGES = [("normal carpet", ROOT / "good/000.png"),
          ("defect: hole", ROOT / "hole/000.png"),
          ("defect: color", ROOT / "color/000.png")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = DinoV2Features("dinov2_vitb14", 224).to(device).eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

imgs, feats = [], []
for _, path in IMAGES:
    x = tf(Image.open(path).convert("RGB"))
    imgs.append(x.permute(1, 2, 0).numpy())
    f = backbone(x.unsqueeze(0).to(device))[0]      # 768 x 16 x 16
    feats.append(f.permute(1, 2, 0).reshape(-1, f.shape[0]).cpu().numpy())  # 256 x 768

# one PCA fit over ALL patches so colors are comparable across images
allp = np.concatenate(feats)
pca = PCA(n_components=3).fit(allp)
proj = pca.transform(allp)
lo, hi = proj.min(0), proj.max(0)

fig, ax = plt.subplots(len(IMAGES), 2, figsize=(6, 3 * len(IMAGES)))
for i, (name, _) in enumerate(IMAGES):
    rgb = (pca.transform(feats[i]) - lo) / (hi - lo)     # 256 x 3 in [0,1]
    rgb = rgb.reshape(16, 16, 3)
    ax[i, 0].imshow(imgs[i]);            ax[i, 0].set_title(f"{name}\n(raw pixels)")
    ax[i, 1].imshow(rgb, interpolation="nearest")
    ax[i, 1].set_title("DINOv2 features\n(768→3 via PCA, as RGB)")
    for a in ax[i]:
        a.axis("off")
fig.tight_layout()
out = Path("feature_visualization.png")
fig.savefig(out, dpi=110)
print(f"saved {out}")
