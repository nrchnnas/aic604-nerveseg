"""Prepare the nerve ultrasound dataset.

Run from anywhere:  python nerveseg/prepare_data.py
Safe to re-run; it rebuilds everything under nerveseg/data/ except the raw extraction.

Required by the brief: 80/20 split with seed 1. Steps marked "agreed 4 Nov" go beyond
the brief; Nana approved them on 4 Nov 2026.

What it does
1. Extracts docs/final project/nerve (1).zip into data/raw/ (skips __MACOSX).
2. Pairs each <id>.json with <id>.jpg or <id>.png by number. The imagePath field
   inside 14 JSON files names a different file, but the embedded imageData matches
   the same-numbered image in all 300 cases, so imagePath is ignored.
3. Agreed 4 Nov (not in the brief): crops every image and mask to a fixed region (ROI) that removes the header
   bar with patient details and the scanner side panels, saved as grayscale PNG.
   All nerve polygons lie inside the ROI, so ROI Dice equals full-image Dice.
4. Groups near-duplicate images (pixel correlation of downsampled ROI > CLUSTER_R, chained).
5. Writes splits/required.json: train_test_split(sorted json names, test_size=0.2,
   random_state=1), the same call Lab 3 uses (the brief does not say how seed 1 is applied).
   Extra splits (agreed 4 Nov): the same split with a validation subset of the 80%, and a split by cluster.
6. Reports how many required-split test images have a near-duplicate in training.
7. YOLO + MedSAM plan (agreed 4 Nov): writes YOLO box labels and per-split image lists.
"""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "docs" / "final project" / "nerve (1).zip"
DATA = ROOT / "data"
RAW = DATA / "raw" / "nerve"

ROI = (64, 64, 800, 608)  # x0, y0, x1, y1 in original 864x648 pixels -> 736x544
CLUSTER_R = 0.80  # correlation threshold for chaining near-duplicates into a cluster
DUP_R = 0.95  # threshold reported as "near-identical"
SEED = 1


def extract() -> None:
    if RAW.exists() and any(RAW.glob("*.json")):
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for m in z.namelist():
            if m.startswith("__MACOSX") or Path(m).name.startswith("."):
                continue
            z.extract(m, RAW.parent)


def load_records() -> list[dict]:
    recs = []
    for jf in sorted(RAW.glob("*.json")):
        sid = jf.stem
        imgs = [p for p in (RAW / f"{sid}.jpg", RAW / f"{sid}.png") if p.exists()]
        assert len(imgs) == 1, f"{sid}: expected one image, found {imgs}"
        j = json.loads(jf.read_text())
        assert len(j["shapes"]) == 1 and j["shapes"][0]["shape_type"] == "polygon", sid
        assert (j["imageWidth"], j["imageHeight"]) == (864, 648), sid
        recs.append(
            {
                "id": sid,
                "json": jf.name,
                "image": imgs[0].name,
                "label_raw": j["shapes"][0]["label"],
                "imagePath_in_json": j["imagePath"],
                "polygon": j["shapes"][0]["points"],
            }
        )
    return recs


def build_roi_files(recs: list[dict]) -> np.ndarray:
    x0, y0, x1, y1 = ROI
    img_dir, mask_dir = DATA / "images_roi", DATA / "masks_roi"
    for d in (img_dir, mask_dir):
        pass  # files are overwritten in place
        d.mkdir(parents=True, exist_ok=True)
    feats = []
    for r in recs:
        im = Image.open(RAW / r["image"]).convert("L")
        m = Image.new("L", (864, 648), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in r["polygon"]], fill=255)
        im_roi, m_roi = im.crop(ROI), m.crop(ROI)
        full_area = int((np.asarray(m) > 0).sum())
        a = np.asarray(m_roi) > 0
        assert a.sum() == full_area, f"{r['id']}: nerve extends outside ROI"
        ys, xs = np.nonzero(a)
        r.update(
            area_px=full_area,
            bbox_roi=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        )
        im_roi.save(img_dir / f"{r['id']}.png")
        m_roi.save(mask_dir / f"{r['id']}.png")
        f = np.asarray(im_roi.resize((92, 68)), dtype=np.float32).ravel()
        feats.append((f - f.mean()) / (f.std() + 1e-6))
    X = np.stack(feats)
    return X @ X.T / X.shape[1]


def cluster(C: np.ndarray) -> np.ndarray:
    n = len(C)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    iu, ju = np.nonzero(np.triu(C, 1) > CLUSTER_R)
    for i, j in zip(iu, ju):
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def pick_clusters(ids: list[str], cl: dict, frac: float, rng: np.random.Generator) -> list[str]:
    groups = sorted({cl[i] for i in ids})
    rng.shuffle(groups)
    target, chosen = round(frac * len(ids)), []
    for g in groups:
        if len(chosen) >= target:
            break
        chosen += [i for i in ids if cl[i] == g]
    return sorted(chosen, key=int)


def make_splits(recs: list[dict], cl_arr: np.ndarray) -> dict:
    ids = [r["id"] for r in recs]
    cl = dict(zip(ids, cl_arr.tolist()))

    # Required split: 80/20, seed 1 (instructor). How "seed 1" is applied is not specified;
    # this uses the same call as Lab 3: train_test_split(sorted names, test_size=0.2, random_state=1).
    names = sorted(r["json"] for r in recs)
    tr_names, te_names = train_test_split(names, test_size=0.2, random_state=SEED)
    req_train = sorted((n[:-5] for n in tr_names), key=int)
    req_test = sorted((n[:-5] for n in te_names), key=int)

    # AGREED 4 Nov (not in the brief): hold out part of the 80% for early stopping.
    req_val = pick_clusters(req_train, cl, 0.15, np.random.default_rng(SEED))
    req_train_minus_val = sorted(set(req_train) - set(req_val), key=int)

    # AGREED 4 Nov (not in the brief): split by near-duplicate cluster as a second evaluation.
    rng = np.random.default_rng(SEED)
    g_test = pick_clusters(ids, cl, 0.20, rng)
    rest = [i for i in ids if i not in set(g_test)]
    g_val = pick_clusters(rest, cl, 0.15, rng)
    g_train = sorted(set(rest) - set(g_val), key=int)

    return {
        "required": {"train": req_train, "test": req_test,
                     "note": "instructor's 80/20 seed 1, implemented as train_test_split(sorted json names, test_size=0.2, random_state=1)"},
        "required_with_val": {"train": req_train_minus_val, "val": req_val, "test": req_test,
                                       "note": "agreed 4 Nov, not in the brief: same test set; val is a cluster-grouped subset of the 80%"},
        "grouped": {"train": g_train, "val": g_val, "test": g_test,
                             "note": f"agreed 4 Nov, not in the brief: whole near-duplicate clusters (r>{CLUSTER_R}) assigned to test/val, seed {SEED}"},
    }


def leakage(C: np.ndarray, ids: list[str], split: dict) -> dict:
    idx = {s: k for k, s in enumerate(ids)}
    train = [idx[i] for i in split["train"] + split.get("val", [])]
    out = {}
    for t in split["test"]:
        out[t] = float(C[idx[t], train].max())
    v = np.array(list(out.values()))
    return {
        "n_test": len(v),
        f"test_with_train_neighbour_r>{DUP_R}": int((v > DUP_R).sum()),
        "test_with_train_neighbour_r>0.90": int((v > 0.90).sum()),
        f"test_with_train_neighbour_r>{CLUSTER_R}": int((v > CLUSTER_R).sum()),
        "per_test_max_r": {k: round(x, 4) for k, x in out.items()},
    }


def write_yolo(recs: list[dict], splits: dict) -> None:
    yolo = DATA / "yolo"
    pass  # files are overwritten in place
    (yolo / "images").mkdir(parents=True, exist_ok=True)
    (yolo / "labels").mkdir(parents=True, exist_ok=True)
    W, H = ROI[2] - ROI[0], ROI[3] - ROI[1]
    for r in recs:
        dst = yolo / "images" / f"{r['id']}.png"
        shutil.copyfile(DATA / "images_roi" / f"{r['id']}.png", dst)
        bx0, by0, bx1, by1 = r["bbox_roi"]
        cx, cy, bw, bh = (bx0 + bx1) / 2 / W, (by0 + by1) / 2 / H, (bx1 - bx0) / W, (by1 - by0) / H
        (yolo / "labels" / f"{r['id']}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    for name, sp in splits.items():
        for part in ("train", "val", "test"):
            if part in sp:
                (yolo / f"{name}_{part}.txt").write_text("".join(f"./images/{i}.png\n" for i in sp[part]))
        if "val" not in sp:
            continue  # YOLO needs a val set; not written for the plain 80/20 split
        (yolo / f"{name}.yaml").write_text(
            f"# regenerate with prepare_data.py on each machine (absolute paths)\n"
            f"path: {yolo.resolve()}\n"
            f"train: {name}_train.txt\nval: {name}_val.txt\ntest: {name}_test.txt\n"
            f"names:\n  0: nerve\n"
        )


def main() -> None:
    extract()
    recs = load_records()
    C = build_roi_files(recs)
    np.fill_diagonal(C, -1)
    cl = cluster(C)
    ids = [r["id"] for r in recs]
    splits = make_splits(recs, cl)

    # The split files in data/splits/ are the team's canonical splits. They are reused on every
    # machine (Mac and Windows) so that small numeric differences cannot change them.
    # Pass --recompute-splits to overwrite them.
    (DATA / "splits").mkdir(exist_ok=True)
    for name in list(splits):
        f = DATA / "splits" / f"{name}.json"
        if f.exists() and not RECOMPUTE:
            kept = json.loads(f.read_text())
            if any(sorted(kept.get(p, [])) != sorted(splits[name].get(p, [])) for p in ("train", "val", "test")):
                print(f"note: recomputed {name} split differs from the saved one; keeping the saved file")
            splits[name] = kept
        else:
            f.write_text(json.dumps(splits[name], indent=1))
    write_yolo(recs, splits)
    report = {
        "n_images": len(recs),
        "roi_xyxy": ROI,
        "label_spellings": {s: sum(r["label_raw"] == s for r in recs) for s in {r["label_raw"] for r in recs}},
        "imagePath_mismatches": [r["id"] for r in recs if Path(r["imagePath_in_json"]).stem != r["id"]],
        f"images_with_any_neighbour_r>{DUP_R}": int((C.max(1) > DUP_R).sum()),
        "n_clusters": int(cl.max() + 1),
        "largest_clusters": sorted(np.bincount(cl).tolist(), reverse=True)[:10],
        "split_sizes": {n: {p: len(sp[p]) for p in ("train", "val", "test") if p in sp} for n, sp in splits.items()},
        "leakage_required": leakage(C, ids, splits["required"]),
        "leakage_grouped": {k: v for k, v in leakage(C, ids, splits["grouped"]).items()
                                     if k != "per_test_max_r"},
    }
    (DATA / "dataset_report.json").write_text(json.dumps(report, indent=1))

    with open(DATA / "meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "label_raw", "imagePath_in_json", "area_px", "bbox_roi_x0", "bbox_roi_y0",
                    "bbox_roi_x1", "bbox_roi_y1", "cluster", "nearest_r", "required_split", "grouped_split"])
        where = {n: {i: p for p in ("train", "val", "test") if p in sp for i in sp[p]} for n, sp in splits.items()}
        for k, r in enumerate(recs):
            w.writerow([r["id"], r["image"], r["label_raw"], r["imagePath_in_json"], r["area_px"], *r["bbox_roi"],
                        int(cl[k]), round(float(C[k].max()), 4), where["required"][r["id"]],
                        where["grouped"][r["id"]]])

    print(json.dumps({k: v for k, v in report.items() if k != "leakage_required"}, indent=1))
    print("required-split leakage:", {k: v for k, v in report["leakage_required"].items() if k != "per_test_max_r"})


if __name__ == "__main__":
    import sys

    RECOMPUTE = "--recompute-splits" in sys.argv
    main()
