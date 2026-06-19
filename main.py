"""
Train + evaluate a Convolutional VAE for anomaly detection on MVTec-AD.

One model is trained per category (the standard MVTec protocol). For every test
image we compute TWO anomaly scores from the same reconstruction:

  * mse  : mean squared reconstruction error   (the naive score)
  * ssim : mean of (1 - SSIM) over the image    (the structure-aware score)

and report image-level AUROC for both, plus pixel-level AUROC (using the
ground-truth masks). The mse-vs-ssim gap is the experiment the carpet case is
about: MSE can rank texture defects backwards; SSIM usually fixes it.

Examples
--------
  # quick smoke test on one category
  python main.py --categories carpet --img-size 64 --epochs 3

  # the real run: all 15 categories
  python main.py --epochs 100
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset
from model import ConvVAE
from ssim import SSIM

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


def vae_loss(recon, x, mu, logvar, ssim_mod, recon_kind, beta):
    """recon loss (summed over pixels) + beta * KL (summed over latent)."""
    if recon_kind == "ssim":
        _, ssim_map = ssim_mod(recon, x)
        recon_loss = (1.0 - ssim_map).sum(dim=[1, 2, 3]).mean()
    else:  # mse
        recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=[1, 2, 3]).mean()
    kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return recon_loss + beta * kld, recon_loss.item(), kld.item()


def train_one(model, loader, args, device):
    ssim_mod = SSIM(channels=args.channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for x in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rl, kl = vae_loss(recon, x, mu, logvar, ssim_mod,
                                    args.recon_loss, args.beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            avg = running / len(loader.dataset)
            print(f"    epoch {epoch:3d}/{args.epochs}  loss {avg:9.2f} "
                  f"(recon {rl:8.2f}  kl {kl:7.2f})")


def _image_score(amap, topk_frac):
    """Image-level score = mean of the worst `topk_frac` pixels of the map.

    A small localized defect is washed out if you average over the whole image,
    so we score by the most-anomalous region instead. topk_frac >= 1 falls back
    to the plain whole-image mean.
    """
    flat = amap.flatten(1)                               # (B, H*W)
    if topk_frac >= 1.0:
        return flat.mean(1)
    k = max(1, int(topk_frac * flat.shape[1]))
    return flat.topk(k, dim=1).values.mean(1)


@torch.no_grad()
def evaluate(model, loader, args, device, fig_dir=None):
    ssim_mod = SSIM(channels=args.channels).to(device)
    model.eval()
    img_labels, mse_scores, ssim_scores = [], [], []
    pix_labels, pix_scores = [], []
    saved = 0

    for x, label, mask in loader:
        x = x.to(device)
        recon = model.reconstruct(x)

        # per-pixel anomaly map = 1 - SSIM (averaged over channels)
        _, ssim_map = ssim_mod(recon, x)
        amap = (1.0 - ssim_map).mean(dim=1)              # (B, H, W)
        mse_map = ((recon - x) ** 2).mean(dim=1)         # (B, H, W)

        # image-level scores = mean of the worst top-k% pixels
        ssim_scores.extend(_image_score(amap, args.score_topk).cpu().numpy())
        mse_scores.extend(_image_score(mse_map, args.score_topk).cpu().numpy())
        img_labels.extend(label.numpy())

        # pixel-level (SSIM map vs ground-truth mask)
        pix_scores.append(amap.cpu().numpy().ravel())
        pix_labels.append(mask[:, 0].numpy().ravel())

        if fig_dir is not None and saved < args.save_figs:
            saved += _save_fig(x, recon, amap, label, fig_dir, saved)

    img_labels = np.array(img_labels)
    out = {
        "auroc_mse": roc_auc_score(img_labels, mse_scores),
        "auroc_ssim": roc_auc_score(img_labels, ssim_scores),
    }
    pl = np.concatenate(pix_labels)
    ps = np.concatenate(pix_scores)
    out["auroc_pixel_ssim"] = roc_auc_score(pl, ps) if pl.max() > 0 else float("nan")
    return out


def _save_fig(x, recon, amap, label, fig_dir, idx):
    import matplotlib.pyplot as plt
    i = 0
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(x[i, 0].cpu(), cmap="gray"); ax[0].set_title("input")
    ax[1].imshow(recon[i, 0].cpu(), cmap="gray"); ax[1].set_title("reconstruction")
    im = ax[2].imshow(amap[i].cpu(), cmap="jet"); ax[2].set_title("anomaly map (1-SSIM)")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    tag = "anom" if int(label[i]) == 1 else "good"
    fig.suptitle(f"label: {tag}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{idx:02d}_{tag}.png", dpi=90)
    plt.close(fig)
    return 1


def run_category(cat, args, device, writer):
    print(f"\n=== {cat} ===")
    train_ds = MVTecDataset(args.data_root, cat, "train", args.img_size, args.channels)
    test_ds = MVTecDataset(args.data_root, cat, "test", args.img_size, args.channels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)
    print(f"    train(good)={len(train_ds)}  test={len(test_ds)}")

    model = ConvVAE(in_channels=args.channels, img_size=args.img_size,
                    latent_dim=args.latent_dim).to(device)

    t0 = time.time()
    train_one(model, train_loader, args, device)

    fig_dir = None
    if args.save_figs > 0:
        fig_dir = Path(args.out_dir) / "figures" / cat
        fig_dir.mkdir(parents=True, exist_ok=True)
    res = evaluate(model, test_loader, args, device, fig_dir)
    dt = time.time() - t0

    print(f"    AUROC  img-mse={res['auroc_mse']:.3f}  "
          f"img-ssim={res['auroc_ssim']:.3f}  "
          f"pix-ssim={res['auroc_pixel_ssim']:.3f}  ({dt:.0f}s)")
    writer.writerow({"category": cat, "n_train": len(train_ds), "n_test": len(test_ds),
                     "auroc_img_mse": round(res["auroc_mse"], 4),
                     "auroc_img_ssim": round(res["auroc_ssim"], 4),
                     "auroc_pixel_ssim": round(res["auroc_pixel_ssim"], 4),
                     "seconds": round(dt, 1)})
    if args.save_model:
        mdir = Path(args.out_dir) / "models"
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), mdir / f"{cat}.pt")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="mvtec_anomaly_detection")
    ap.add_argument("--categories", nargs="*", default=ALL_CATEGORIES)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="KL weight. Lower = sharper reconstruction (less collapse "
                         "to a generic texture); beta=0 is a plain autoencoder.")
    ap.add_argument("--score-topk", type=float, default=0.01,
                    help="image score = mean of the worst this-fraction of pixels "
                         "(0.01 = top 1%%). Use 1.0 for plain whole-image mean.")
    ap.add_argument("--recon-loss", default="ssim", choices=["ssim", "mse"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-figs", type=int, default=6,
                    help="how many qualitative test figures to save per category")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | img_size={args.img_size} | recon_loss={args.recon_loss}"
          f" | epochs={args.epochs}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out_dir) / "results.csv"
    fields = ["category", "n_train", "n_test", "auroc_img_mse",
              "auroc_img_ssim", "auroc_pixel_ssim", "seconds"]
    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cat in args.categories:
            res = run_category(cat, args, device, writer)
            rows.append((cat, res))
            f.flush()

    # summary
    if rows:
        mse = np.mean([r["auroc_mse"] for _, r in rows])
        ssim = np.mean([r["auroc_ssim"] for _, r in rows])
        print("\n================ MEAN over categories ================")
        print(f"  image-level AUROC   mse={mse:.3f}   ssim={ssim:.3f}")
        print(f"  results written to  {csv_path}")


if __name__ == "__main__":
    main()
