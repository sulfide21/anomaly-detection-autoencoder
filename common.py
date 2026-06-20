"""Shared helpers used by both pipelines (pixel-VAE and DINO-AE)."""

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


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
