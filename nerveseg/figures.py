"""Report figures from predictions. Works for any model whose masks are PNG files named like the labels.

Examples (from the comp-vision folder):
  # success and failure cases + Dice distribution for one experiment
  python nerveseg/figures.py cases --dataset 501 --pred nerveseg/nnunet/predictions/Dataset501_NerveRequired__nnUNetTrainer__fold0 --out nerveseg/figures/501_default

  # compare experiments (per-image Dice CSVs written by evaluate.py)
  python nerveseg/figures.py compare --out nerveseg/figures/compare_splits.png \
      "required split=nerveseg/nnunet/predictions/Dataset501_NerveRequired__nnUNetTrainer__fold0_dice.csv" \
      "grouped split=nerveseg/nnunet/predictions/Dataset502_NerveGrouped__nnUNetTrainer__fold0_dice.csv"

  # Dice against true nerve size, coloured by failure type (per-image CSV written by evaluate.py)
  python nerveseg/figures.py size --out nerveseg/figures/501_default_size.png \
      nerveseg/nnunet/predictions/Dataset501_NerveRequired__nnUNetTrainer__fold0_dice.csv

  # the same test images under two models, one row each
  python nerveseg/figures.py pair --dataset 501 --cases nerve_151 nerve_152 --out nerveseg/figures/pair.png \
      "W0=nerveseg/nnunet/predictions/Dataset501_NerveRequired__nnUNetTrainer_250epochs__fold0" \
      "W2=nerveseg/nnunet/nnUNet_results/Dataset502_NerveGrouped/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_0/validation"

Images are the ROI-cropped scans (patient header removed), so the figures are safe for slides.
Colours: true outline green, predicted outline magenta.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE / "nnunet" / "nnUNet_raw"
DATASETS = {"501": "Dataset501_NerveRequired", "502": "Dataset502_NerveGrouped"}
GT_COLOR, PRED_COLOR = "#2ca02c", "#e377c2"


def dice(p: np.ndarray, g: np.ndarray) -> float:
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(p, g).sum() / s)


def load(ds: str, pred: Path) -> list[dict]:
    d = RAW / DATASETS[ds]
    rows = []
    for g in sorted((d / "labelsTs").glob("*.png")):
        gm = np.asarray(Image.open(g)) > 0
        pf = pred / g.name
        pm = np.asarray(Image.open(pf)) > 0 if pf.exists() else np.zeros_like(gm)
        img = np.asarray(Image.open(d / "imagesTs" / f"{g.stem}_0000.png").convert("L"))
        rows.append({"case": g.stem, "img": img, "gt": gm, "pred": pm, "dice": dice(pm, gm)})
    return rows


def crop_box(gm: np.ndarray, pm: np.ndarray, pad: int = 60) -> tuple[slice, slice]:
    ys, xs = np.nonzero(gm | pm)
    if len(ys) == 0:
        return slice(None), slice(None)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, gm.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, gm.shape[1])
    return slice(y0, y1), slice(x0, x1)


def draw(ax, r: dict, zoom: bool, box: tuple[slice, slice] | None = None) -> None:
    sy, sx = box if box else crop_box(r["gt"], r["pred"]) if zoom else (slice(None), slice(None))
    ax.imshow(r["img"][sy, sx], cmap="gray")
    for m, c in ((r["gt"], GT_COLOR), (r["pred"], PRED_COLOR)):
        if m[sy, sx].any():
            ax.contour(m[sy, sx].astype(float), levels=[0.5], colors=c, linewidths=1.5)
    ax.set_title(f"{r['case']}  Dice {r['dice']:.2f}", fontsize=9)
    ax.axis("off")


def cmd_cases(a) -> None:
    rows = sorted(load(a.dataset, Path(a.pred)), key=lambda r: r["dice"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    picks = {"Lowest Dice": rows[:3], "Median Dice": rows[n // 2 - 1:n // 2 + 2], "Highest Dice": rows[-3:][::-1]}
    for zoom in (True, False):
        fig, axes = plt.subplots(3, 3, figsize=(10, 8.5) if zoom else (12, 9.5))
        for i, (label, rs) in enumerate(picks.items()):
            for j, r in enumerate(rs):
                draw(axes[i, j], r, zoom)
            axes[i, 0].text(-0.08, 0.5, label, transform=axes[i, 0].transAxes, rotation=90,
                            va="center", ha="right", fontsize=11)
        fig.legend(handles=[plt.Line2D([], [], color=GT_COLOR, label="true outline"),
                            plt.Line2D([], [], color=PRED_COLOR, label="predicted outline")],
                   loc="lower center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0.02, 0.04, 1, 1))
        f = out.with_name(out.name + ("_cases_zoom.png" if zoom else "_cases_full.png"))
        fig.savefig(f, dpi=200)
        plt.close(fig)
        print("wrote", f)

    d = np.array([r["dice"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(d, bins=np.linspace(0, 1, 21), color="#4c72b0", edgecolor="white")
    ax.axvline(d.mean(), color="black", ls="--", lw=1, label=f"mean {d.mean():.3f}")
    ax.set_xlabel("Dice per test image")
    ax.set_ylabel("Number of test images")
    ax.legend(frameon=False)
    fig.tight_layout()
    f = out.with_name(out.name + "_dice_hist.png")
    fig.savefig(f, dpi=200)
    plt.close(fig)
    print("wrote", f)


def cmd_compare(a) -> None:
    labels, data = [], []
    for item in a.items:
        label, path = item.split("=", 1)
        with open(path) as fh:
            data.append([float(r["dice"]) for r in csv.DictReader(fh)])
        labels.append(label)
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(labels), 4))
    ax.boxplot(data, showmeans=True, showfliers=False, meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black"})
    rng = np.random.default_rng(1)
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=10, alpha=0.5, color="#4c72b0", zorder=3)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Dice per test image")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    plt.close(fig)
    print("wrote", a.out)


def cmd_pair(a) -> None:
    """The same test images under several models: one row per model, one column per image. Each column is cropped
    to cover the true nerve and every model's prediction, so a prediction on the wrong structure stays in view."""
    models = [item.split("=", 1) for item in a.items]
    rows = {label: {r["case"]: r for r in load(a.dataset, Path(path))} for label, path in models}
    missing = [(label, c) for label, path in models for c in a.cases if not (Path(path) / f"{c}.png").exists()]
    if missing:
        raise SystemExit(f"no prediction for {missing}")
    boxes = []
    for case in a.cases:
        union = np.logical_or.reduce([rows[label][case]["pred"] for label, _ in models])
        boxes.append(crop_box(rows[models[0][0]][case]["gt"], union, pad=a.pad))
    aspect = float(np.median([(sy.stop - sy.start) / (sx.stop - sx.start) for sy, sx in boxes]))
    fig, axes = plt.subplots(len(models), len(a.cases), squeeze=False,
                             figsize=(2.9 * len(a.cases), (2.9 * aspect + 0.45) * len(models) + 0.9))
    for j, (case, box) in enumerate(zip(a.cases, boxes)):
        for i, (label, _) in enumerate(models):
            draw(axes[i, j], rows[label][case], True, box)
    for i, (label, _) in enumerate(models):
        axes[i, 0].text(-0.08, 0.5, label, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=11)
    if a.title:
        fig.suptitle(a.title, fontsize=12)
    fig.legend(handles=[plt.Line2D([], [], color=GT_COLOR, label="true outline"),
                        plt.Line2D([], [], color=PRED_COLOR, label="predicted outline")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.02, 0.05, 1, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    plt.close(fig)
    print("wrote", a.out)


FAILURE_STYLE = {"found": ("#4c72b0", "found (Dice >= {t})"), "poor": ("#dd8452", "poor overlap (0 < Dice < {t})"),
                 "missed_elsewhere": ("#c44e52", "missed: predicted elsewhere"),
                 "missed_empty": ("#7f7f7f", "missed: empty prediction")}


def cmd_size(a) -> None:
    from evaluate import FAILURE_TYPES, size_cuts

    with open(a.csv) as fh:
        rows = list(csv.DictReader(fh))
    if "failure" not in rows[0]:
        raise SystemExit(f"{a.csv} has no failure column; re-run evaluate.py")
    fig, ax = plt.subplots(figsize=(7, 4))
    cuts = size_cuts()
    if cuts:
        for c in cuts:
            ax.axvline(c, color="black", lw=0.8, ls=":")
        ymax = 1.08
        for x, label in ((cuts[0] / 2 + 300, "small"), (sum(cuts) / 2, "medium"), (cuts[1] + 900, "large")):
            ax.text(x, ymax - 0.03, label, ha="center", va="top", fontsize=9, color="#333333")
    for t in FAILURE_TYPES:
        sel = [r for r in rows if r["failure"] == t]
        if not sel:
            continue
        color, label = FAILURE_STYLE[t]
        ax.scatter([int(r["gt_px"]) for r in sel], [float(r["dice"]) for r in sel], s=22, color=color,
                   label=f"{label.format(t=a.poor_below)} ({len(sel)})", alpha=0.85, edgecolor="white", lw=0.4)
    ax.set_xlabel("True nerve area (pixels)")
    ax.set_ylabel("Dice per image")
    ax.set_ylim(-0.03, 1.08)
    if a.title:
        ax.set_title(a.title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    plt.close(fig)
    print("wrote", a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--dataset", choices=list(DATASETS), default="501")
    c.add_argument("--pred", required=True)
    c.add_argument("--out", required=True, help="output path prefix, e.g. nerveseg/figures/501_default")
    k = sub.add_parser("compare")
    k.add_argument("--out", required=True)
    k.add_argument("items", nargs="+", help='"label=path/to/dice.csv"')
    z = sub.add_parser("size")
    z.add_argument("--out", required=True)
    z.add_argument("--title")
    z.add_argument("--poor-below", type=float, default=0.5, help="only for the legend; must match evaluate.py")
    z.add_argument("csv", help="per-image CSV written by evaluate.py")
    q = sub.add_parser("pair")
    q.add_argument("--dataset", choices=list(DATASETS), default="501", help="whose test images and labels")
    q.add_argument("--cases", nargs="+", required=True, help="e.g. nerve_151 nerve_152")
    q.add_argument("--out", required=True)
    q.add_argument("--title")
    q.add_argument("--pad", type=int, default=60, help="pixels around the nerve and predictions")
    q.add_argument("items", nargs="+", help='"row label=folder of predicted masks"')
    a = ap.parse_args()
    {"cases": cmd_cases, "compare": cmd_compare, "size": cmd_size, "pair": cmd_pair}[a.cmd](a)


if __name__ == "__main__":
    main()
