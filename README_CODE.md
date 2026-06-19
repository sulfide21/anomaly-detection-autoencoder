# VAE Anomaly Detection on MVTec-AD

A Convolutional **Variational Autoencoder**, one model per category, trained on
normal images only. Each test image gets two anomaly scores from the same
reconstruction — naive **MSE** and structure-aware **SSIM** — so you can report
the gap between them (the carpet/texture story from `READING_GUIDE.md`).

## Files
| file | what it is |
|------|------------|
| `model.py`   | the Conv-VAE (encoder → mu/logvar → sample → decoder) |
| `ssim.py`    | differentiable windowed SSIM (training loss **and** anomaly score) |
| `dataset.py` | MVTec loader: train=good only, test=good+defects with masks |
| `main.py`    | train + evaluate, loops over all 15 categories, writes `results/results.csv` |

## Setup
```powershell
.\venv\Scripts\Activate.ps1     # venv already created
```

## Run
```powershell
# quick check on one category (seconds)
python main.py --categories carpet --img-size 64 --epochs 3

# recommended full run on CPU — ~4-7 h, do it overnight
python main.py --img-size 128 --epochs 60 --save-model

# train with MSE reconstruction loss instead of SSIM (for comparison)
python main.py --recon-loss mse --img-size 128 --epochs 60
```

## Output (`results/`)
- `results.csv` — per-category image-level AUROC (mse & ssim) + pixel-level AUROC
- `figures/<cat>/*.png` — input | reconstruction | anomaly map, for a few test images
- `models/<cat>.pt` — weights (only with `--save-model`)

## Key knobs
`--img-size` (divisible by 32) · `--epochs` · `--recon-loss {ssim,mse}` ·
`--beta` (KL weight) · `--latent-dim` · `--channels {1,3}` · `--categories ...`

## Reading the result
Headline number = **mean image-level AUROC** across categories (the guide's
"~0.70" baseline). Compare the `auroc_img_mse` vs `auroc_img_ssim` columns —
that difference, especially on textures (carpet, grid, leather, tile, wood),
is the SSIM-vs-MSE finding to write up.
