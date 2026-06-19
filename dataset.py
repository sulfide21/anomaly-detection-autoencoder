"""
MVTec-AD loader.

The dataset rule: TRAIN on normal images only (`train/good/`), then TEST on a
mix of `good` + several defect folders. `ground_truth/` holds pixel masks for
the defects (none for `good`).

  MVTecDataset(root, category, split="train")  -> normal images only
  MVTecDataset(root, category, split="test")   -> (image, label, mask)
       label: 0 = good (normal), 1 = anomalous
       mask : (1, H, W) float in {0,1}; all zeros for good images
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", img_size=256, channels=1):
        self.root = Path(root) / category
        self.split = split
        self.channels = channels

        img_tf = [transforms.Resize((img_size, img_size))]
        if channels == 1:
            img_tf.append(transforms.Grayscale(num_output_channels=1))
        img_tf.append(transforms.ToTensor())            # -> [0,1], (C,H,W)
        self.img_tf = transforms.Compose(img_tf)

        # masks: nearest-neighbour resize so they stay binary
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.samples = []  # (img_path, label, mask_path_or_None)
        if split == "train":
            for p in sorted((self.root / "train" / "good").glob("*.png")):
                self.samples.append((p, 0, None))
        elif split == "test":
            for defect_dir in sorted((self.root / "test").iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect = defect_dir.name
                label = 0 if defect == "good" else 1
                for p in sorted(defect_dir.glob("*.png")):
                    mask = None
                    if label == 1:
                        m = self.root / "ground_truth" / defect / f"{p.stem}_mask.png"
                        mask = m if m.exists() else None
                    self.samples.append((p, label, mask))
        else:
            raise ValueError(f"unknown split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        img = self.img_tf(Image.open(img_path).convert("RGB"))
        if self.split == "train":
            return img
        if mask_path is None:
            mask = torch.zeros(1, img.shape[1], img.shape[2])
        else:
            mask = self.mask_tf(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
        return img, label, mask
