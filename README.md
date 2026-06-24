# DINO-AE: Features over Architecture

*A lightweight DINOv2 autoencoder for unsupervised anomaly detection on [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad).*

Train on **normal images only**; at test time, detect whether an image is defective
and localize **where**. The method reconstructs **frozen DINOv2 features** with a tiny
convolutional autoencoder — anything it fails to reconstruct is flagged as an anomaly.

| method | image AUROC | pixel AUROC | reconstructs | model |
|---|---|---|---|---|
| UniAD (SOTA reference) | 0.942 | 0.960 | EfficientNet features | transformer + 3 tricks |
| **DINO-AE (this repo)** | **0.972** | **0.966** | **DINOv2 features** | **tiny conv AE ×15** |

A small autoencoder on frozen foundation features **beats** the purpose-built transformer SOTA — the strength of the features matters more than the complexity of the model.

## How it works
```
image → [frozen DINOv2 ViT-B/14] → features → [tiny conv AE] → cosine distance → anomaly map → top-k score
```
- **DINOv2** turns each 14×14 patch into a 768-d "meaning" vector. These are computed **once** and cached (the backbone is frozen), so training the AE takes seconds.
- The AE is forced to *summarize, not copy* normal features via two anti-shortcut tricks (inspired by UniAD): a spatial+channel **bottleneck** and norm-scaled **feature noise** (denoising).
- One specialist model per category (15 total), same protocol across all categories.

## Code
| file | role |
|---|---|
| `dataset.py` | MVTec loader (train = good only; test = good + defects with masks) |
| `dino_features.py` | frozen DINOv2 patch-token extractor (+ ImageNet normalization) |
| `feat_recon.py` | the tiny conv autoencoder (bottleneck + feature-noise) |
| `train_dino_ae.py` | extract → cache → train AE → evaluate (AUROC), per category |
| `predict.py` | single-image verdict: NORMAL / DEFECT + anomaly heatmap |
| `common.py` | shared helpers (category list, top-k image score) |
| `visualize_features.py` | PCA visualization of DINOv2 features |

## Setup
```bash
python -m venv venv && source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Download MVTec-AD into `mvtec_anomaly_detection/` (not included — research-only license).

## Run
```bash
# train + evaluate all 15 categories (saves weights with --save-model)
python train_dino_ae.py --epochs 200 --save-model

# verify on a single image: is THIS one defective?
python predict.py --category screw --image path/to/some_screw.png
```
`predict.py` prints the anomaly score, the decision threshold (99th percentile of the
training-good scores — no test labels used), a **NORMAL / DEFECT** verdict, and saves a
heatmap showing where the defect is.

*Trained on an RTX 4060 (8 GB).*
