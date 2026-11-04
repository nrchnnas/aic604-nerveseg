"""Dice evaluation for nerve segmentation predictions (works for any model).

Usage:
  python nerveseg/evaluate.py --pred <folder of predicted masks> --gt <folder of true masks> [--out results.csv]
                              [--baseline other_results.csv] [--poor-below 0.5]

Both folders hold PNG masks with matching file names (e.g. nerve_012.png). Any nonzero pixel counts
as nerve. For nnU-Net: --gt nerveseg/nnunet/nnUNet_raw/Dataset501_NerveRequired/labelsTs

Reports per-image Dice and IoU, mean, median, and a bootstrap 95% CI of the mean (10,000 resamples
of test images, seed 1). For the required split it also splits Dice by whether the test image has a
near-identical training image (correlation > 0.95, from nerveseg/data/dataset_report.json), which
shows how much the near-duplicates inflate the score.

Failure types (our definitions; every image has exactly one nerve):
  missed_empty      the prediction is empty
  missed_elsewhere  the prediction does not touch the true nerve (Dice 0)
  poor              it overlaps, but Dice is below --poor-below (default 0.5)
  found             Dice at or above --poor-below
Size groups: small / medium / large nerve, split at the terciles of the true nerve area over all 300
images (data/meta.csv: 1308 and 2289 px), so the groups are the same for every experiment and split.
Regions are 8-connected, as in postprocess.py.

--baseline: paired comparison with another result CSV of the same images (this minus baseline),
with a bootstrap 95% CI of the mean difference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
REPORT = HERE / "data" / "dataset_report.json"
META = HERE / "data" / "meta.csv"
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
FAILURE_TYPES = ("found", "poor", "missed_elsewhere", "missed_empty")
SIZE_GROUPS = ("small", "medium", "large")


def dice_iou(p: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(p, g).sum()
    s = p.sum() + g.sum()
    union = np.logical_or(p, g).sum()
    dice = 1.0 if s == 0 else 2 * inter / s
    iou = 1.0 if union == 0 else inter / union
    return float(dice), float(iou)


def bootstrap_ci(x: np.ndarray, n: int = 10_000, seed: int = 1) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(name: str, x: np.ndarray) -> str:
    if len(x) == 0:
        return f"{name}: no images"
    lo, hi = bootstrap_ci(x)
    return (f"{name}: n={len(x)}  mean Dice {x.mean():.4f}  (95% CI {lo:.4f} to {hi:.4f})  "
            f"median {np.median(x):.4f}  min {x.min():.4f}")


def size_cuts() -> tuple[float, float] | None:
    if not META.exists():
        return None
    with open(META) as f:
        areas = np.array([int(r["area_px"]) for r in csv.DictReader(f)])
    lo, hi = np.percentile(areas, [100 / 3, 200 / 3])
    return float(lo), float(hi)


def failure_type(dice: float, pred_px: int, inter: int, poor_below: float) -> str:
    if pred_px == 0:
        return "missed_empty"
    if inter == 0:
        return "missed_elsewhere"
    return "poor" if dice < poor_below else "found"


def score(pred: Path, gt: Path, poor_below: float = 0.5) -> list[dict]:
    """One row per true mask in gt. A missing prediction is scored as empty (with a warning)."""
    gts = sorted(Path(gt).glob("*.png"))
    assert gts, f"no PNG masks in {gt}"
    cuts = size_cuts()
    rows = []
    for g in gts:
        pf = Path(pred) / g.name
        gm = np.asarray(Image.open(g)) > 0
        if pf.exists():
            pm = np.asarray(Image.open(pf)) > 0
            assert pm.shape == gm.shape, f"{g.name}: shape {pm.shape} vs {gm.shape}"
        else:
            print(f"warning: no prediction for {g.name}, scored as empty")
            pm = np.zeros_like(gm)
        d, j = dice_iou(pm, gm)
        gt_px, pred_px = int(gm.sum()), int(pm.sum())
        row = {"case": g.stem, "dice": d, "iou": j, "gt_px": gt_px, "pred_px": pred_px,
               "n_pred_regions": int(ndimage.label(pm, structure=EIGHT_CONNECTED)[1]),
               "failure": failure_type(d, pred_px, int(np.logical_and(pm, gm).sum()), poor_below)}
        if cuts:
            row["size_group"] = "small" if gt_px < cuts[0] else "medium" if gt_px < cuts[1] else "large"
        rows.append(row)

    # nearest training image correlation, when the cases are the required split's test set
    if REPORT.exists():
        maxr = json.loads(REPORT.read_text())["leakage_required"]["per_test_max_r"]
        ids = [str(int(r["case"].split("_")[-1])) for r in rows]
        if set(ids) == set(maxr):
            for r, i in zip(rows, ids):
                r["nearest_train_r"] = maxr[i]
    return rows


def summary(rows: list[dict]) -> dict:
    d = np.array([float(r["dice"]) for r in rows])
    lo, hi = bootstrap_ci(d)
    return {"n": len(d), "mean_dice": float(d.mean()), "ci95": [lo, hi], "median_dice": float(np.median(d)),
            "min_dice": float(d.min()), "mean_iou": float(np.mean([float(r["iou"]) for r in rows]))}


def report(rows: list[dict], poor_below: float = 0.5) -> None:
    dice = np.array([float(r["dice"]) for r in rows])
    print(summarize("all images", dice))
    print(f"mean IoU {np.mean([float(r['iou']) for r in rows]):.4f}")

    if "nearest_train_r" in rows[0]:
        leaked = np.array([float(r["nearest_train_r"]) > 0.95 for r in rows])
        print(summarize("  with a near-identical training image (r > 0.95)", dice[leaked]))
        print(summarize("  without one", dice[~leaked]))

    counts = {t: sum(r["failure"] == t for r in rows) for t in FAILURE_TYPES}
    print(f"failure types (poor = 0 < Dice < {poor_below}): "
          + ", ".join(f"{t} {counts[t]}" for t in FAILURE_TYPES))
    multi = sum(int(r["n_pred_regions"]) > 1 for r in rows)
    print(f"images with more than one predicted region: {multi} of {len(rows)}")

    if "size_group" in rows[0]:
        lo, hi = size_cuts()
        bounds = {"small": f"< {lo:.0f} px", "medium": f"{lo:.0f} to {hi:.0f} px", "large": f">= {hi:.0f} px"}
        for g in SIZE_GROUPS:
            sel = np.array([r["size_group"] == g for r in rows])
            print(summarize(f"  {g} nerves ({bounds[g]})", dice[sel]))
    area = np.array([int(r["gt_px"]) for r in rows])
    if len(rows) > 2 and np.ptp(dice) > 0:
        rho, p = stats.spearmanr(area, dice)
        print(f"Spearman rho, true nerve area vs Dice: {rho:.3f} (p = {p:.3g}, n = {len(rows)})")


def paired(rows: list[dict], base_rows: list[dict]) -> dict:
    """Mean of (this - baseline) Dice over the cases both have, with a bootstrap 95% CI."""
    base = {r["case"]: float(r["dice"]) for r in base_rows}
    shared = [r for r in rows if r["case"] in base]
    if not shared:
        raise SystemExit("no shared cases between the two results")
    diff = np.array([float(r["dice"]) - base[r["case"]] for r in shared])
    lo, hi = bootstrap_ci(diff)
    eps = 1e-9
    return {"n": len(diff), "mean_diff": float(diff.mean()), "ci95": [lo, hi],
            "better": int((diff > eps).sum()), "worse": int((diff < -eps).sum()),
            "equal": int((np.abs(diff) <= eps).sum()),
            "n_only_in_one": len(rows) + len(base_rows) - 2 * len(shared)}


def paired_text(p: dict) -> str:
    return (f"paired difference (this - baseline): n={p['n']}  mean {p['mean_diff']:+.4f}  "
            f"(95% CI {p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f})  better {p['better']}, worse {p['worse']}, "
            f"equal {p['equal']}" + (f"  [{p['n_only_in_one']} cases in only one file, ignored]"
                                     if p["n_only_in_one"] else ""))


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--baseline", type=Path, help="per-image CSV of another result on the same images")
    ap.add_argument("--poor-below", type=float, default=0.5)
    a = ap.parse_args()

    rows = score(a.pred, a.gt, a.poor_below)
    report(rows, a.poor_below)
    if a.baseline:
        print(paired_text(paired(rows, read_csv(a.baseline))))
    if a.out:
        write_csv(rows, a.out)
        print(f"per-image results written to {a.out}")


if __name__ == "__main__":
    main()
