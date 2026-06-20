# Anomaly Detection on MVTec-AD: Pixel-VAE → UniAD → DINO-AE

Unsupervised industrial defect detection on [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad):
train only on normal images, detect & localize defects at test time. This project
walks one idea through three stages — **what should the autoencoder reconstruct?**

| method | image AUROC | pixel AUROC | reconstructs | model |
|---|---|---|---|---|
| Pixel-VAE (baseline) | 0.687 | 0.805 | raw pixels | conv VAE ×15 |
| UniAD (reproduced SOTA) | 0.942 | 0.960 | EfficientNet features | transformer + 3 tricks |
| **DINO-AE (ours)** | **0.972** | **0.966** | **DINOv2 features** | **tiny conv AE ×15** |

**Takeaway:** going from raw pixels → pretrained features → foundation features
mattered far more than model complexity. A tiny autoencoder on frozen DINOv2
features **beats** the purpose-built transformer SOTA.

## Code
| file | role |
|---|---|
| `dataset.py` | MVTec loader (shared) |
| `baseline_vae/` (model.py, ssim.py, main.py) | Pixel-VAE baseline: network, SSIM loss/score, train+eval |
| `dino_features.py`, `feat_recon.py`, `train_dino_ae.py` | DINO-AE: frozen DINOv2 extractor, tiny feature-AE, train+eval |
| `visualize_features.py` | PCA visualization of DINOv2 features |

## Docs
- `COMPARISON.md` — full method + code-level comparison, per-category results
- `PROGRESS_SUMMARY.md` — plain-language project summary
- `READING_GUIDE.md` — annotated bibliography
- `UNIAD_SETUP.md` — how to reproduce the UniAD baseline

## Setup
```bash
python -m venv venv && source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Download MVTec-AD into `mvtec_anomaly_detection/` (not included — research-only license).

## Run
```bash
python train_dino_ae.py --epochs 200                    # DINO-AE (main method), all 15
python baseline_vae/main.py --img-size 256 --epochs 100 # Pixel-VAE baseline, all 15
```

*Trained on an RTX 4060 (8 GB).*
