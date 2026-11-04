"""Convert the prepared nerve data into nnU-Net v2 raw datasets.

Run after nerveseg/prepare_data.py:   python nerveseg/nnunet/make_nnunet_dataset.py

Creates, under nerveseg/nnunet/nnUNet_raw/:
- Dataset501_NerveRequired: the instructor's 80/20 split (seed 1).
    imagesTr = the 240 training images, imagesTs = the 60 test images.
- Dataset502_NerveGrouped: the split by near-duplicate cluster.
    imagesTr = 237 train+val images, imagesTs = the 63 test images.
Each dataset also gets labelsTs/ (test masks for our own Dice evaluation; nnU-Net ignores it).

Format (nnU-Net dataset_format.md): grayscale PNG images named <case>_0000.png, label PNGs
named <case>.png with values 0 (background) and 1 (nerve), dataset.json with file_ending ".png".
Images are the ROI-cropped grayscale images from prepare_data.py (patient header removed).

Also writes nerveseg/nnunet/splits/<dataset>_splits_final.json. nnU-Net uses a splits_final.json
placed in nnUNet_preprocessed/<dataset>/ instead of its own random 5-fold split. Copy it there
after nnUNetv2_plan_and_preprocess (run.py does this). The folds keep near-duplicate clusters
together, so validation images never have a near-copy in their training fold:
- Dataset501 fold 0 = splits/required_with_val.json (train 201 / val 39); folds 1 to 4 split the
  remaining clusters of the 240 so that every image is validated once across the 5 folds.
- Dataset502 fold 0 = splits/grouped.json (train 201 / val 36); folds 1 to 4 likewise.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # nerveseg/
DATA = ROOT / "data"
NNU = ROOT / "nnunet"
RAW = NNU / "nnUNet_raw"
SPLIT_OUT = NNU / "splits"
N_FOLDS = 5
SEED = 1

DATASETS = {
    "Dataset501_NerveRequired": {"split_file": "required.json", "fold0": "required_with_val.json"},
    "Dataset502_NerveGrouped": {"split_file": "grouped.json", "fold0": "grouped.json"},
}


def case(i: str) -> str:
    return f"nerve_{int(i):03d}"


def load_clusters() -> dict[str, int]:
    with open(DATA / "meta.csv") as f:
        return {r["id"]: int(r["cluster"]) for r in csv.DictReader(f)}


def write_case(i: str, img_dir: Path, lab_dir: Path) -> None:
    shutil.copyfile(DATA / "images_roi" / f"{i}.png", img_dir / f"{case(i)}_0000.png")
    m = (np.asarray(Image.open(DATA / "masks_roi" / f"{i}.png")) > 0).astype(np.uint8)
    Image.fromarray(m, mode="L").save(lab_dir / f"{case(i)}.png")


def grouped_folds(train_ids: list[str], fold0_val: list[str], cl: dict[str, int]) -> list[dict]:
    """Fold 0 validation is given; remaining clusters are spread over folds 1..N-1, balancing size."""
    fold0_clusters = {cl[i] for i in fold0_val}
    assert all(cl[i] not in fold0_clusters for i in train_ids if i not in fold0_val), "fold 0 val splits a cluster"
    rest = [i for i in train_ids if i not in set(fold0_val)]
    groups: dict[int, list[str]] = {}
    for i in rest:
        groups.setdefault(cl[i], []).append(i)
    order = sorted(groups, key=lambda g: (-len(groups[g]), g))
    rng = np.random.default_rng(SEED)
    vals: list[list[str]] = [list(fold0_val)] + [[] for _ in range(N_FOLDS - 1)]
    for g in order:  # largest clusters first, each to the currently smallest fold (ties broken randomly)
        sizes = np.array([len(v) for v in vals[1:]])
        cands = np.flatnonzero(sizes == sizes.min())
        vals[1 + int(rng.choice(cands))] += groups[g]
    folds = []
    for v in vals:
        vs = set(v)
        folds.append({"train": sorted(case(i) for i in train_ids if i not in vs), "val": sorted(case(i) for i in v)})
    return folds


def main() -> None:
    cl = load_clusters()
    SPLIT_OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, cfg in DATASETS.items():
        sp = json.loads((DATA / "splits" / cfg["split_file"]).read_text())
        f0 = json.loads((DATA / "splits" / cfg["fold0"]).read_text())
        train_ids = sorted(sp["train"] + sp.get("val", []), key=int)
        test_ids = sorted(sp["test"], key=int)
        assert set(f0["test"]) == set(test_ids) and set(f0["train"] + f0["val"]) == set(train_ids)

        d = RAW / name
        dirs = {k: d / k for k in ("imagesTr", "labelsTr", "imagesTs", "labelsTs")}
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        for i in train_ids:
            write_case(i, dirs["imagesTr"], dirs["labelsTr"])
        for i in test_ids:
            write_case(i, dirs["imagesTs"], dirs["labelsTs"])
        (d / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "ultrasound"},
            "labels": {"background": 0, "nerve": 1},
            "numTraining": len(train_ids),
            "file_ending": ".png",
            "name": name,
            "description": f"ROI-cropped grayscale ultrasound, nerve masks. Train/test from nerveseg/data/splits/{cfg['split_file']}.",
        }, indent=1))

        # Saved fold files are canonical (shared across Mac and Windows); --recompute-splits overwrites them.
        sf = SPLIT_OUT / f"{name}_splits_final.json"
        if sf.exists() and "--recompute-splits" not in sys.argv:
            folds = json.loads(sf.read_text())
        else:
            folds = grouped_folds(train_ids, f0["val"], cl)
            sf.write_text(json.dumps(folds, indent=1))
        summary[name] = {"train": len(train_ids), "test": len(test_ids),
                         "fold_val_sizes": [len(f["val"]) for f in folds]}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
