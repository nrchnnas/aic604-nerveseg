"""Keep only the largest predicted region in each mask (every image has exactly one nerve).

Usage:
  python nerveseg/postprocess.py --pred <folder of predicted PNG masks> --out <new folder>

Regions are 8-connected (pixels touching at an edge or a corner belong to the same region). This matches
nnU-Net's own clean-up step, which labels regions with full connectivity (see docs/nnunet_facts.md).
An empty mask stays empty. Only the PNG masks are written; nnU-Net's json/npz/pkl files are not copied.
run.py calls this for its `postprocess` step; the folder then ends in __lcc (largest connected component).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def keep_largest(mask: np.ndarray) -> np.ndarray:
    """Return the mask with every region except the largest set to 0. Ties go to the first region found."""
    lab, n = ndimage.label(mask > 0, structure=EIGHT_CONNECTED)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    return np.where(lab == np.argmax(sizes) + 1, mask, 0).astype(mask.dtype)


def run(pred: Path, out: Path) -> dict:
    files = sorted(Path(pred).glob("*.png"))
    if not files:
        raise SystemExit(f"no PNG masks in {pred}")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)  # always a generated folder; avoids leftovers from an older run
    out.mkdir(parents=True)
    changed, removed = [], 0
    for f in files:
        m = np.asarray(Image.open(f))
        k = keep_largest(m)
        n_removed = int((m > 0).sum() - (k > 0).sum())
        if n_removed:
            changed.append(f.stem)
            removed += n_removed
        Image.fromarray(k).save(out / f.name)
    print(f"largest region kept: {len(files)} masks, {len(changed)} changed, {removed} pixels removed -> {out}")
    return {"n_masks": len(files), "n_changed": len(changed), "pixels_removed": removed, "changed": changed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.pred.resolve() == a.out.resolve():
        raise SystemExit("--out must be a different folder from --pred")
    run(a.pred, a.out)


if __name__ == "__main__":
    main()
