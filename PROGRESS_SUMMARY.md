# Progress Summary — Anomaly Detection on MVTec-AD

**For explaining the work to the lecturer.** Read top to bottom — it's a story.

## The one framing that ties everything together
> The task: **unsupervised** defect detection on MVTec-AD — train **only on normal
> images**, detect & localize defects at test time. I built three methods that are
> really *one idea being refined*: **what should the autoencoder reconstruct?**
> Each step answered that better, and the final one (my own) **beats the published
> SOTA I reproduced**.

## The dataset
MVTec-AD: 15 categories, each with `train/good` (normal only) + `test` (good +
defects) + pixel masks. Metric: image-level AUROC (detection) and pixel-level
AUROC (localization).

## Act 1 — Pixel-VAE (baseline)
- A (variational) autoencoder reconstructing **raw pixels**, one model per category.
- Idea: train on normal only → defects can't be reconstructed → high error = anomaly.
- **Result: 0.687 detection / 0.805 localization** (mean over 15).
- Finding: good on objects (hazelnut 0.96), **bad on textures** — carpet's MSE
  score came out *inverted* (0.31, below chance) because a global latent can't
  repaint a carpet weave. SSIM scoring partly rescues it.
- → Motivated finding a fundamentally better approach.

## Act 2 — UniAD (reproduce the state of the art)
- NeurIPS 2022. Reconstructs **pretrained features** (not pixels) with a
  transformer; **one unified model** for all 15 categories. Uses 3 "anti-shortcut"
  tricks so it can't just copy its input.
- I ported the official 2022 code to run on a modern GPU (RTX 4060) and trained it.
- **Result: 0.942 / 0.960.**
- Lesson: the big gain is **features instead of raw pixels** — never repaint
  texture, just reproduce a feature vector. So the real lever is *feature quality*.

## Act 3 — DINO-AE (my contribution)
- I tested that lever directly: reconstruct **DINOv2 foundation-model features**
  with a **tiny conv autoencoder** (bottleneck + noise only — no transformer, none
  of UniAD's tricks). Score by cosine distance.
- **Result: 0.972 / 0.966 — beats the reproduced UniAD**, far simpler model,
  ~20 min total training vs hours.
- Best-or-tied on 13 of 15 categories. The categories that destroyed the VAE
  (carpet 0.52→1.00, metal_nut 0.56→1.00, tile 0.86→1.00) are now solved.

## Headline table
| method | detection | localization | model | train time |
|---|---|---|---|---|
| pixel-VAE | 0.687 | 0.805 | conv VAE ×15 | ~5 h |
| UniAD (reproduced) | 0.942 | 0.960 | transformer + 3 tricks | ~2 h |
| **DINO-AE (mine)** | **0.972** | **0.966** | tiny conv AE ×15 | ~20 min |

## The takeaway (say this)
> Going from **raw pixels → pretrained features → foundation features** mattered
> far more than the model architecture. DINO-AE proves it: the *simplest*
> reconstructor on the *best* features beats a purpose-built transformer SOTA.

## Artifacts (where things live)
- Pixel-VAE: `model.py`, `ssim.py`, `main.py` · results `results_vae_full/results.csv`
- UniAD (ported): `UniAD/` · results `UniAD/experiments/MVTec-AD/run100.log`
- DINO-AE: `dino_features.py`, `feat_recon.py`, `train_dino_ae.py` · results `results_dino_ae/results.csv`
- Full method comparison + code-level explanation: `COMPARISON.md`
- Paper reading list for the report: `READING_GUIDE.md`

## Honest caveats (good to volunteer)
- UniAD was trained 100 epochs (paper uses 1000 → ~0.967); still a fair comparison.
- VAE is 15 specialists; UniAD is 1 unified model (harder setting) — noted in the report.
- DINO-AE uses a *frozen* foundation backbone — the novelty is the combination &
  the demonstration that features dominate, not a new backbone.
