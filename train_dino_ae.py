"""
DINO-AE: anomaly detection by reconstructing frozen DINOv2 features.

Pipeline:  image -> [frozen DINOv2 ViT-B/14] -> features -> [minimal conv AE]
           -> cosine distance per token -> anomaly map -> top-k image score.

One model per category (same protocol as our VAE), so results drop straight into
the VAE-vs-UniAD comparison. Reuses our eval/AUROC helpers from main.py.

Features are cached once per category (the backbone is frozen, so they never
change) — the tiny AE then trains on the cache in seconds.

Example:
  python train_dino_ae.py --categories carpet --epochs 200      # quick check
  python train_dino_ae.py --epochs 200 --save-model             # all 15
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from dataset import MVTecDataset
from dino_features import DinoV2Features
from feat_recon import FeatRecon
from main import ALL_CATEGORIES, _image_score


def cosine_distance_map(rec, target):
    """1 - cosine similarity per spatial token. rec/target: B x C x H x W -> B x H x W."""
    rec = F.normalize(rec, dim=1)
    target = F.normalize(target, dim=1)
    return 1.0 - (rec * target).sum(dim=1)


@torch.no_grad()
def extract_features(backbone, loader, device, with_meta=False):
    feats, labels, masks = [], [], []
    for batch in loader:
        if with_meta:
            x, label, mask = batch
            labels.append(label)
            masks.append(mask)
        else:
            x = batch
        feats.append(backbone(x.to(device)).cpu())
    feats = torch.cat(feats)
    if with_meta:
        return feats, torch.cat(labels), torch.cat(masks)
    return feats


def train_ae(model, feats, args, device):
    loader = DataLoader(TensorDataset(feats), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for (f,) in loader:
            f = f.to(device)
            rec = model(f)
            loss = cosine_distance_map(rec, f).mean()   # denoise back to clean f
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * f.size(0)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"    epoch {epoch:3d}/{args.epochs}  cos-loss {running/len(feats):.4f}")


@torch.no_grad()
def evaluate(model, feats, labels, masks, args, device):
    model.eval()
    img_scores, pix_scores, pix_labels = [], [], []
    for i in range(0, len(feats), args.batch_size):
        f = feats[i:i + args.batch_size].to(device)
        rec = model(f)
        amap = cosine_distance_map(rec, f)              # B x g x g
        img_scores.extend(_image_score(amap, args.score_topk).cpu().numpy())
        up = F.interpolate(amap.unsqueeze(1), size=masks.shape[-1],
                           mode="bilinear", align_corners=False)[:, 0]
        pix_scores.append(up.cpu().numpy().ravel())
        pix_labels.append(masks[i:i + args.batch_size, 0].numpy().ravel())
    labels = labels.numpy()
    auroc_img = roc_auc_score(labels, img_scores)
    pl, ps = np.concatenate(pix_labels), np.concatenate(pix_scores)
    auroc_pix = roc_auc_score(pl, ps) if pl.max() > 0 else float("nan")
    return auroc_img, auroc_pix


def run_category(cat, backbone, args, device, writer):
    print(f"\n=== {cat} ===")
    train_ds = MVTecDataset(args.data_root, cat, "train", args.img_size, channels=3)
    test_ds = MVTecDataset(args.data_root, cat, "test", args.img_size, channels=3)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    t0 = time.time()
    train_feats = extract_features(backbone, train_loader, device)
    test_feats, test_labels, test_masks = extract_features(backbone, test_loader,
                                                           device, with_meta=True)
    print(f"    train={len(train_feats)}  test={len(test_feats)}  "
          f"feat={tuple(train_feats.shape[1:])}")

    model = FeatRecon(in_dim=backbone.embed_dim, bottleneck=args.bottleneck,
                      noise_std=args.noise_std).to(device)
    train_ae(model, train_feats, args, device)
    auroc_img, auroc_pix = evaluate(model, test_feats, test_labels, test_masks,
                                    args, device)
    dt = time.time() - t0
    print(f"    AUROC  img={auroc_img:.3f}  pixel={auroc_pix:.3f}  ({dt:.0f}s)")
    writer.writerow({"category": cat, "n_train": len(train_feats),
                     "n_test": len(test_feats), "auroc_img": round(auroc_img, 4),
                     "auroc_pixel": round(auroc_pix, 4), "seconds": round(dt, 1)})
    if args.save_model:
        mdir = Path(args.out_dir) / "models"
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), mdir / f"{cat}.pt")
    return auroc_img, auroc_pix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="mvtec_anomaly_detection")
    ap.add_argument("--categories", nargs="*", default=ALL_CATEGORIES)
    ap.add_argument("--model-name", default="dinov2_vitb14")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--noise-std", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--score-topk", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--out-dir", default="results_dino_ae")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | backbone: {args.model_name} | img_size={args.img_size}")
    backbone = DinoV2Features(args.model_name, args.img_size).to(device)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out_dir) / "results.csv"
    fields = ["category", "n_train", "n_test", "auroc_img", "auroc_pixel", "seconds"]
    rows = []
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        for cat in args.categories:
            rows.append(run_category(cat, backbone, args, device, writer))
            fcsv.flush()

    if rows:
        mi = np.mean([r[0] for r in rows])
        mp = np.mean([r[1] for r in rows])
        print(f"\n========= MEAN  img={mi:.3f}  pixel={mp:.3f}  =========")
        print(f"results -> {csv_path}")


if __name__ == "__main__":
    main()
