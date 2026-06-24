"""
predict.py — single-image anomaly verdict for DINO-AE.

Feed ANY image and get a yes/no answer: NORMAL or DEFECT, with the anomaly
score, the decision threshold, and a heatmap showing *where* the defect is.

This is the "does the model actually work" demo. AUROC (in results.csv) proves
the model separates good from defective, but it is threshold-free — it can't
judge a single new image. Here we pick a concrete threshold the honest way:
the Pth percentile of the anomaly scores of the TRAINING images (all "good").
No test labels are used, so the threshold is not cheating.

Everything else (backbone, AE, transform, scoring) is imported from the training
code, so the demo runs the exact same pipeline.

Examples:
  python predict.py --category screw --image mvtec_anomaly_detection/screw/test/scratch_neck/000.png
  python predict.py --category carpet --image some_carpet.jpg --percentile 99
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import MVTecDataset
from dino_features import DinoV2Features
from feat_recon import FeatRecon
from common import _image_score
from train_dino_ae import cosine_distance_map, extract_features


def load_image(path, img_size):
    """Same transform the dataset uses for channels=3: Resize -> RGB -> [0,1] tensor."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img)                                       # 3 x H x W in [0,1]


def anomaly_for_feat(model, feat, topk):
    """feat: 1 x C x g x g -> (image_score float, amap g x g numpy)."""
    rec = model(feat)
    amap = cosine_distance_map(rec, feat)                # 1 x g x g
    score = _image_score(amap, topk).item()
    return score, amap[0].cpu().numpy()


def compute_threshold(backbone, model, args, device):
    """Pth percentile of train-good image scores. Cached per (category, settings)."""
    cache_path = Path(args.out_dir) / "thresholds.json"
    key = f"{args.category}_p{args.percentile}_topk{args.score_topk}_img{args.img_size}"
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if not args.no_cache and key in cache:
            return cache[key]

    print(f"[threshold] computing {args.percentile}th percentile of train-good scores...")
    train_ds = MVTecDataset(args.data_root, args.category, "train", args.img_size, channels=3)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    feats = extract_features(backbone, loader, device)   # N x C x g x g
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(feats), args.batch_size):
            f = feats[i:i + args.batch_size].to(device)
            rec = model(f)
            amap = cosine_distance_map(rec, f)
            scores.extend(_image_score(amap, args.score_topk).cpu().numpy())
    thr = float(np.percentile(scores, args.percentile))
    cache[key] = thr
    cache_path.write_text(json.dumps(cache, indent=2))
    print(f"[threshold] {args.category}: {thr:.4f}  "
          f"(from {len(scores)} good images, range {min(scores):.4f}-{max(scores):.4f})")
    return thr


def save_heatmap(img_tensor, amap, score, thr, verdict, out_path):
    """input | heatmap | overlay, saved to out_path. Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[heatmap] matplotlib not installed — skipping image (verdict still valid)")
        return
    base = img_tensor.permute(1, 2, 0).numpy()           # H x W x 3
    g = amap.shape[-1]
    up = F.interpolate(torch.tensor(amap)[None, None], size=base.shape[0],
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(base);                 ax[0].set_title("input")
    ax[1].imshow(up, cmap="jet");       ax[1].set_title("anomaly map")
    ax[2].imshow(base);                 ax[2].imshow(up, cmap="jet", alpha=0.45)
    ax[2].set_title(f"{verdict}  (score {score:.3f} / thr {thr:.3f})")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Single-image anomaly verdict for DINO-AE.")
    ap.add_argument("--category", required=True, help="which trained model to use, e.g. screw")
    ap.add_argument("--image", required=True, help="path to the image to test")
    ap.add_argument("--data-root", default="mvtec_anomaly_detection")
    ap.add_argument("--out-dir", default="results_dino_ae", help="where models/ and outputs live")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--model-name", default="dinov2_vitb14")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--noise-std", type=float, default=0.2)
    ap.add_argument("--score-topk", type=float, default=0.01)
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="threshold = this percentile of train-good scores")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-cache", action="store_true", help="recompute the threshold")
    ap.add_argument("--no-heatmap", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Path(args.out_dir) / "models" / f"{args.category}.pt"
    if not weights.exists():
        raise SystemExit(f"No weights for '{args.category}' at {weights}. "
                         f"Train with: python train_dino_ae.py --save-model")
    if not Path(args.image).exists():
        raise SystemExit(f"Image not found: {args.image}")

    print(f"[setup] category={args.category}  device={device}")
    backbone = DinoV2Features(args.model_name, args.img_size).to(device)
    model = FeatRecon(in_dim=backbone.embed_dim, bottleneck=args.bottleneck,
                      noise_std=args.noise_std).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    thr = compute_threshold(backbone, model, args, device)

    img = load_image(args.image, args.img_size)          # 3 x H x W
    with torch.no_grad():
        feat = backbone(img[None].to(device))            # 1 x C x g x g
        score, amap = anomaly_for_feat(model, feat, args.score_topk)

    verdict = "DEFECT" if score > thr else "NORMAL"
    ratio = score / thr if thr > 0 else float("inf")
    print("\n" + "=" * 44)
    print(f"  image     : {args.image}")
    print(f"  category  : {args.category}")
    print(f"  score     : {score:.4f}")
    print(f"  threshold : {thr:.4f}  ({args.percentile:.0f}th pct of good)")
    print(f"  ratio     : {ratio:.2f}x threshold")
    print(f"  VERDICT   : {verdict}")
    print("=" * 44 + "\n")

    if not args.no_heatmap:
        p = Path(args.image)
        stem = f"{p.parent.name}_{p.stem}"               # e.g. scratch_neck_000, avoids name clashes
        out_path = Path(args.out_dir) / "predictions" / f"{args.category}_{stem}.png"
        save_heatmap(img, amap, score, thr, verdict, out_path)


if __name__ == "__main__":
    main()
