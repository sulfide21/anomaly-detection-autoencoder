# Reading Guide — Anomaly Detection on MVTec-AD

An annotated bibliography for the tugas (autoencoder anomaly detection on MVTec-AD).
Each entry: what problem it solves, the key idea, the headline result, and why it
matters for *our* project. Ordered roughly by how central it is to our work.

Reading priority: **★★★ read first**, **★★ skim**, **★ optional / frontier**.

---

## A. The dataset (cite for the data source)

### ★★★ MVTec-AD — the dataset paper (journal version)
> Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021).
> *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
> Unsupervised Anomaly Detection.* International Journal of Computer Vision (IJCV), 129, 1038–1059.
> (Conference original: CVPR 2019.) https://link.springer.com/article/10.1007/s11263-020-01400-4

5,354 high-res images, 15 categories (10 objects + 5 textures), 70+ defect types,
with pixel-accurate ground-truth masks. Defines the task: **train on normal images
only**, detect & localize defects at test time (unsupervised). The paper also
benchmarks early methods (autoencoders, GANs, pretrained-feature methods) — which is
exactly the family we implemented.

---

## B. Our method family — reconstruction / autoencoder

### ★★★ SSIM Autoencoder (the method we built)
> Bergmann, P., Löwe, S., Fauser, M., Sattlegger, D., & Steger, C. (2019).
> *Improving Unsupervised Defect Segmentation by Applying Structural Similarity to
> Autoencoders.* VISAPP 2019. https://arxiv.org/abs/1807.02011

The core argument: a per-pixel reconstruction error (MSE / Lp) is a *bad* anomaly
metric — it over-reacts to tiny edge misalignments and misses defects that keep the
same brightness. They replace it with **SSIM** (compares luminance + contrast +
structure of local windows). Headline: just changing the loss/score from MSE to SSIM
raised AUC from **0.688 → 0.966** on a texture dataset. **This is the published version
of our carpet finding** (MSE inverted at 0.36, SSIM un-inverted it).

### ★★ DRAEM — reconstruction + synthetic defects
> Zavrtanik, V., Kristan, M., & Skočaj, D. (2021). *DRAEM — A Discriminatively
> Trained Reconstruction Embedding for Surface Anomaly Detection.* ICCV 2021.
> https://arxiv.org/abs/2108.07610

Instead of *only* reconstructing, DRAEM **synthesizes fake defects** on normal images
during training and learns a discriminative normal-vs-anomaly boundary. Gives direct
localization without post-processing and beats plain autoencoders by a large margin.
Connects to our "make your own anomalies" contamination trick — here it's used to
*train*, not just to test.

---

## C. Feature / embedding-based (the ~99% state-of-the-art)

### ★★ PaDiM
> Defard, T., Setkov, A., Loesch, A., & Audigier, R. (2021). *PaDiM: a Patch
> Distribution Modeling Framework for Anomaly Detection and Localization.* ICPR 2021.
> https://arxiv.org/abs/2011.08785

Models the *distribution* of normal patch features (from a pretrained CNN) with a
per-location Gaussian. Test patches far from the normal distribution = anomaly. No
reconstruction at all — it compares deep features.

### ★★ PatchCore
> Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022).
> *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022.
> https://arxiv.org/abs/2106.08265

Keeps a **memory bank** of normal patch features (coreset-subsampled for speed) and
scores a test patch by distance to its nearest normal neighbor. Reaches **~99.6%
AUROC** on MVTec — the standard SOTA baseline. Explains why our pixel-AE (~0.70 mean)
is the simpler, interpretable baseline: the gap is *features vs raw pixels*.

### ★ Reverse Distillation (RD4AD)
> Deng, H., & Li, X. (2022). *Anomaly Detection via Reverse Distillation from
> One-Class Embedding.* CVPR 2022. https://arxiv.org/abs/2201.10703

A teacher–student model with a trainable **one-class bottleneck** that keeps normal
patterns but drops anomaly information. A bridge between our bottleneck/autoencoder
intuition and feature-based methods.

---

## D. Hidden gems / unique topics (the interesting threads)

### ★★★ SoftPatch — anomaly detection with NOISY training data
> Jiang, X., et al. (2022). *SoftPatch: Unsupervised Anomaly Detection with Noisy
> Data.* NeurIPS 2022. https://arxiv.org/abs/2403.14233

**This is our contamination experiment, formalized.** It rejects the usual assumption
that training data is perfectly clean (unrealistic in a real factory), scores each
training *patch* for how "noisy"/anomalous it looks, and down-weights/removes the
suspect ones before building its memory bank. Directly mirrors our "score the training
images, remove the worst" cleanup — but at patch level and at NeurIPS.

### ★★★ MVTec LOCO AD — *logical* anomalies (reframes the task)
> Bergmann, P., et al. (2022). *Beyond Dents and Scratches: Logical Constraints in
> Unsupervised Anomaly Detection and Localization.* IJCV.
> https://link.springer.com/article/10.1007/s11263-022-01578-9

Introduces a new dataset separating **structural** anomalies (scratches, dents — what
our AE handles) from **logical** anomalies (the parts are individually fine but the
*arrangement* is wrong: wrong count, missing/extra object, wrong position). A
pixel-autoencoder is essentially blind to logical anomalies — perfect "limitations /
future work" material.

### ★★ EfficientAD — the AE survives in 2024 SOTA, at 2 ms
> Batzner, K., et al. (2024). *EfficientAD: Accurate Visual Anomaly Detection at
> Millisecond-Level Latencies.* WACV 2024. https://arxiv.org/abs/2303.14535

Runs at **~2 ms/image** and fuses a **student–teacher network with an autoencoder**
(the AE provides global context to catch logical anomalies). Shows the humble
autoencoder we built is still a *component* of cutting-edge methods — not obsolete.

---

## E. Foundation-model frontier (no / little training — the opposite of 15 specialists)

### ★ UniAD — one model for ALL categories
> You, Z., et al. (2022). *A Unified Model for Multi-class Anomaly Detection.*
> NeurIPS 2022 (Spotlight). https://arxiv.org/abs/2206.03687

Trains a *single* model for all 15 categories instead of one-per-category. Identifies
the **"identical shortcut"** problem (a powerful unified model just copies its input,
including anomalies) and fixes it with masked attention + feature jittering. This is
*exactly* the "one unified model vs 15 specialists" question we debated.

### ★ AnomalyCLIP — zero-shot (no training images!)
> Zhou, Q., et al. (2024). *AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot
> Anomaly Detection.* ICLR 2024. https://arxiv.org/abs/2310.18961

Adapts CLIP with learnable **object-agnostic prompts** ("normal" vs "abnormal") so it
detects defects on categories it has *never seen* — no training images at all.

### ★ AnomalyGPT — chat with your defects
> Gu, Z., et al. (2024). *AnomalyGPT: Detecting Industrial Anomalies Using Large
> Vision-Language Models.* AAAI 2024. https://arxiv.org/abs/2308.15366

A Large Vision-Language Model that answers "is this defective and where?" in
multi-turn dialogue, no manual threshold needed. Few-shot in-context learning.

### ★ Segment Any Anomaly (SAA)
> Cao, Y., et al. (2023). *Segment Any Anomaly without Training via Hybrid Prompt
> Regularization.* https://arxiv.org/abs/2305.10724

**Training-free** defect segmentation by combining Segment Anything (SAM) +
Grounding DINO with text prompts. Foundation models, zero fine-tuning.

---

## F. For the background section

### ★★ Survey
> *A survey of deep learning for industrial visual anomaly detection.* Artificial
> Intelligence Review (2025). https://link.springer.com/article/10.1007/s10462-025-11287-7

One citation that maps the whole landscape — handy for the related-work background.

### ★ CutPaste — simple synthetic anomalies
> Li, C.-L., et al. (2021). *CutPaste: Self-Supervised Learning for Anomaly Detection
> and Localization.* CVPR 2021. https://arxiv.org/abs/2104.04015

Cut a patch and paste it elsewhere to fabricate a defect; train a model to spot it.
Same spirit as DRAEM, simpler.

---

## The narrative arc (for the report)

```
pixel AE (Bergmann 2019)  →  + SSIM (Bergmann 2019)  →  + synthetic defects (DRAEM/CutPaste 2021)
  →  feature-based (PaDiM/PatchCore 2021–22)  →  unified multi-class (UniAD 2022)
  →  logical anomalies (MVTec LOCO 2022)  →  noisy-data (SoftPatch 2022)
  →  efficient SOTA (EfficientAD 2024)  →  zero-shot foundation models (AnomalyCLIP/AnomalyGPT/SAA 2023–24)

Our work sits at the first two stages: the honest, interpretable baseline —
and we independently touched the "noisy data" and "unified model" threads.
```
