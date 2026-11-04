"""One command for every nnU-Net step. Works the same on macOS and Windows.

It sets nnU-Net's folder variables itself (and nnUNet_extTrainer, so nnU-Net finds our trainers in
nnunet/trainers/), so there is nothing to `source` first. Run with the project's Python environment active:

  python nerveseg/nnunet/run.py check                       # versions, device, datasets, trainers, power
  python nerveseg/nnunet/run.py preprocess [--planner nnUNetPlannerResEncM]
  python nerveseg/nnunet/run.py timing   -d 501 [-t TRAINER] [-p PLANS]   # 5-epoch run; log copied to docs/timing/
  python nerveseg/nnunet/run.py train    -d 501 [-t TRAINER] [-p PLANS] [-f 0] [--resume]
  python nerveseg/nnunet/run.py predict  -d 501 [-t ...] [--on test|val] [--checkpoint final|best] [--no-tta]
  python nerveseg/nnunet/run.py postprocess -d 501 [same options]           # keep the largest region -> ..__lcc
  python nerveseg/nnunet/run.py evaluate -d 501 [same options] [--lcc]
  python nerveseg/nnunet/run.py select   -d 501 -t TRAINER                  # decisions on the validation images
  python nerveseg/nnunet/run.py test-variants -d 501 -t TRAINER             # every variant on the test images
  python nerveseg/nnunet/run.py compare  -d 501 -t nnUNetTrainer --vs OTHER_TRAINER[:PLANS]
  python nerveseg/nnunet/run.py curves   -d 501 -t TRAINER [--machine NAME] # training curves into figures/
  python nerveseg/nnunet/run.py ensemble -d 501 -t TRAINER [--folds 0 1 2 3 4]  # C: 5-fold ensemble on test + CV Dice
  python nerveseg/nnunet/run.py records  --machine NAME                     # small result files -> docs/results/NAME/

-d: 501 = required 80/20 split, 502 = split by near-duplicate cluster.
-t: nnU-Net trainer class, default nnUNetTrainer (1000 epochs). Shorter: nnUNetTrainer_250epochs, etc.
    Ours: nnUNetTrainerUSAug (+ _5epochs, _250epochs, _500epochs, _750epochs), see trainers/.
-p: plans, default nnUNetPlans. ResEnc M: nnUNetResEncUNetMPlans (after preprocess --planner nnUNetPlannerResEncM).
--device: auto (default) picks cuda (NVIDIA, Windows/Linux), then mps (Apple), then cpu.

Prediction folders: nerveseg/nnunet/predictions/<Dataset>__<trainer>[__<plans>]__fold<f>[__val][__best][__notta][__lcc]
(the plans part only when not nnUNetPlans). The default validation result (final checkpoint, flip averaging on)
is nnU-Net's own fold_<f>/validation/ folder; it is reused, never predicted again.

Selection rule (agreed 4 Nov, fixed before results): on the fold's validation images, a variant is adopted only
if it raises mean Dice. One decision at a time: flip averaging first, then the clean-up on top of that choice.
The checkpoint is not selected: checkpoint_best was picked on these same validation images, so best vs final is
reported on test as evidence about overfitting only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent  # nerveseg/nnunet
NERVESEG = HERE.parent
sys.path.insert(0, str(NERVESEG))  # evaluate.py, postprocess.py

DATASETS = {"501": "Dataset501_NerveRequired", "502": "Dataset502_NerveGrouped"}
DEFAULT_PLANS = "nnUNetPlans"
PLANNERS = {"nnUNetPlannerResEncM": "nnUNetResEncUNetMPlans", "nnUNetPlannerResEncL": "nnUNetResEncUNetLPlans",
            "nnUNetPlannerResEncXL": "nnUNetResEncUNetXLPlans"}
TRAINERS_DIR = HERE / "trainers"
PRED_ROOT = HERE / "predictions"
VAL_ROOT = HERE / "val_inputs"
TIMING_DIR = NERVESEG / "docs" / "timing"
FIGURES = NERVESEG / "figures"


# ---------------------------------------------------------------- environment

def set_env() -> None:
    os.environ["nnUNet_raw"] = str(HERE / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(HERE / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(HERE / "nnUNet_results")
    ext = [p for p in os.environ.get("nnUNet_extTrainer", "").split(os.pathsep) if p]
    if str(TRAINERS_DIR) not in ext:
        os.environ["nnUNet_extTrainer"] = os.pathsep.join([str(TRAINERS_DIR)] + ext)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    for k in ("nnUNet_preprocessed", "nnUNet_results"):
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def power_problem() -> str | None:
    """On a Mac: None if on the charger with Low Power Mode off, else the reason. Other systems: None."""
    if platform.system() != "Darwin":
        return None
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    if "AC Power" not in (batt.splitlines() or [""])[0]:
        return "the Mac is running on battery"
    g = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    for key in ("lowpowermode", "powermode"):  # macOS 26 reports powermode (0 auto, 1 low power, 2 high power)
        m = re.search(rf"^\s*{key}\s+(\d)", g, re.M)
        if m and m.group(1) == "1":
            return f"Low Power Mode is on ({key} 1)"
    return None


def require_power(a) -> None:
    why = power_problem()
    if why:
        msg = f"{why}: timings and training would be throttled. Plug in and turn Low Power Mode off"
        if a.ignore_power:
            print(f"WARNING: {msg} (continuing because of --ignore-power)")
        else:
            sys.exit(f"{msg}, or pass --ignore-power.")


def tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:  # pip puts console scripts next to the interpreter (Scripts\ on Windows, bin/ elsewhere)
        for cand in (Path(sys.executable).parent / name, Path(sys.executable).parent / f"{name}.exe",
                     Path(sys.executable).parent / "Scripts" / f"{name}.exe"):
            if cand.exists():
                return str(cand)
        sys.exit(f"Cannot find {name}. Is nnunetv2 installed in this environment? (pip install nnunetv2==2.8.1)")
    return exe


def run(cmd: list[str], tee: Path | None = None) -> None:
    print(">>", " ".join(cmd), flush=True)
    if tee is None:
        r = subprocess.run(cmd)
        code = r.returncode
    else:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # so the child's output arrives line by line
        with open(tee, "a", encoding="utf-8") as fh, subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env) as p:
            for line in p.stdout:
                sys.stdout.write(line)
                fh.write(line)
            code = p.wait()
    if code != 0:
        sys.exit(code)


def copy_splits() -> None:
    for name in DATASETS.values():
        src = HERE / "splits" / f"{name}_splits_final.json"
        dst_dir = Path(os.environ["nnUNet_preprocessed"]) / name
        if src.exists() and dst_dir.exists():
            shutil.copyfile(src, dst_dir / "splits_final.json")
            print(f"copied {src.name} -> {dst_dir / 'splits_final.json'}")


# ---------------------------------------------------------------- variants and folders

@dataclass(frozen=True)
class Variant:
    ds: str
    trainer: str
    plans: str = DEFAULT_PLANS
    fold: int = 0
    on: str = "test"          # test | val
    checkpoint: str = "final"  # final | best
    tta: bool = True           # nnU-Net's test-time flip averaging
    lcc: bool = False          # keep only the largest connected region

    @property
    def dataset(self) -> str:
        return DATASETS[self.ds]

    def model_dir(self) -> Path:
        return (Path(os.environ["nnUNet_results"]) / self.dataset / f"{self.trainer}__{self.plans}__2d"
                / f"fold_{self.fold}")

    def name(self) -> str:
        n = f"{self.dataset}__{self.trainer}"
        if self.plans != DEFAULT_PLANS:
            n += f"__{self.plans}"
        n += f"__fold{self.fold}"
        if self.on == "val":
            n += "__val"
        if self.checkpoint == "best":
            n += "__best"
        if not self.tta:
            n += "__notta"
        if self.lcc:
            n += "__lcc"
        return n

    def is_nnunet_validation(self) -> bool:
        return self.on == "val" and self.checkpoint == "final" and self.tta and not self.lcc

    def pred_dir(self) -> Path:
        return self.model_dir() / "validation" if self.is_nnunet_validation() else PRED_ROOT / self.name()

    def csv(self) -> Path:
        return PRED_ROOT / f"{self.name()}_dice.csv"

    def label(self) -> str:
        return (f"{self.trainer}" + (f" [{self.plans}]" if self.plans != DEFAULT_PLANS else "")
                + f", {self.checkpoint} checkpoint, flip averaging {'on' if self.tta else 'off'}"
                + (", largest region only" if self.lcc else ""))


def variant_from_args(a, **kw) -> Variant:
    ck = {"checkpoint_final.pth": "final", "checkpoint_best.pth": "best"}.get(a.checkpoint, a.checkpoint)
    v = Variant(a.dataset, a.trainer, a.plans, a.fold, a.on, ck, not a.no_tta, getattr(a, "lcc", False))
    return replace(v, **kw)


def val_cases(dataset: str, fold: int) -> list[str]:
    splits = json.loads((HERE / "splits" / f"{dataset}_splits_final.json").read_text())
    return sorted(splits[fold]["val"])


def prepare_val(dataset: str, fold: int) -> tuple[Path, Path]:
    """Copy the fold's validation images and labels into their own folders (nnUNetv2_predict needs a folder)."""
    raw = Path(os.environ["nnUNet_raw"]) / dataset
    root = VAL_ROOT / f"{dataset}_fold{fold}"
    img, lab = root / "images", root / "labels"
    cases = val_cases(dataset, fold)
    have = sorted(p.stem for p in lab.glob("*.png")) if lab.exists() else []
    if have != cases:
        if root.exists():
            shutil.rmtree(root)  # generated folder
        img.mkdir(parents=True)
        lab.mkdir(parents=True)
        for c in cases:
            shutil.copyfile(raw / "imagesTr" / f"{c}_0000.png", img / f"{c}_0000.png")
            shutil.copyfile(raw / "labelsTr" / f"{c}.png", lab / f"{c}.png")
        print(f"copied {len(cases)} validation images and labels of {dataset} fold {fold} -> {root}")
    return img, lab


def inputs(v: Variant) -> tuple[Path, Path]:
    if v.on == "val":
        return prepare_val(v.dataset, v.fold)
    raw = Path(os.environ["nnUNet_raw"]) / v.dataset
    return raw / "imagesTs", raw / "labelsTs"


# ---------------------------------------------------------------- steps on one variant

def predict(v: Variant, device: str, force: bool = False) -> Path:
    assert not v.lcc, "the clean-up is a separate step (postprocess)"
    out = v.pred_dir()
    if v.is_nnunet_validation():
        if not out.exists():
            sys.exit(f"{out} does not exist. nnU-Net writes it at the end of training; is training finished?")
        print(f"default validation result: reusing nnU-Net's {out}")
        return out
    ckpt = v.model_dir() / f"checkpoint_{v.checkpoint}.pth"
    if not ckpt.exists():
        sys.exit(f"missing {ckpt}")
    images, labels = inputs(v)
    expected = [out / f.name for f in labels.glob("*.png")]
    if not force and expected and all(p.exists() for p in expected) and \
            min(p.stat().st_mtime for p in expected) >= ckpt.stat().st_mtime:
        print(f"up to date, reusing {out} (pass --force to predict again)")
        return out
    if out.exists():
        shutil.rmtree(out)  # generated folder; avoids leftovers from an older model
    run([tool("nnUNetv2_predict"), "-i", str(images), "-o", str(out), "-d", v.ds, "-c", "2d",
         "-f", str(v.fold), "-tr", v.trainer, "-p", v.plans, "-chk", ckpt.name, "-device", pick_device(device)]
        + ([] if v.tta else ["--disable_tta"]))
    return out


def postprocess(v: Variant) -> dict:
    import postprocess as pp

    src, dst = replace(v, lcc=False).pred_dir(), replace(v, lcc=True).pred_dir()
    if not src.exists():
        sys.exit(f"{src} does not exist; predict first")
    return pp.run(src, dst)


def evaluate(v: Variant, baseline: tuple[str, list[dict]] | None = None, poor_below: float = 0.5) -> list[dict]:
    import evaluate as ev

    pred = v.pred_dir()
    if not pred.exists():
        sys.exit(f"{pred} does not exist")
    print(f"\n=== {v.on}: {v.label()}\n    {pred}")
    rows = ev.score(pred, inputs(v)[1], poor_below)
    ev.report(rows, poor_below)
    if baseline:
        print(f"vs {baseline[0]}: " + ev.paired_text(ev.paired(rows, baseline[1])))
    ev.write_csv(rows, v.csv())
    print(f"per-image results written to {v.csv()}")
    return rows


def decide(what: str, base_name: str, base_rows: list[dict], cand_name: str, cand_rows: list[dict]) -> dict:
    """The agreed rule: adopt the candidate only if its mean validation Dice is higher."""
    import evaluate as ev

    sb, sc, p = ev.summary(base_rows), ev.summary(cand_rows), ev.paired(cand_rows, base_rows)
    adopt = sc["mean_dice"] > sb["mean_dice"]
    print(f"\n--- decision: {what}\n    baseline  {base_name}: mean {sb['mean_dice']:.4f}"
          f"\n    candidate {cand_name}: mean {sc['mean_dice']:.4f}\n    {ev.paired_text(p)}"
          f"\n    -> {'ADOPT' if adopt else 'keep baseline'}"
          + ("  (CI includes 0: the difference may be noise)" if p["ci95"][0] <= 0 <= p["ci95"][1] else ""))
    return {"question": what, "baseline": base_name, "candidate": cand_name, "baseline_summary": sb,
            "candidate_summary": sc, "paired_candidate_minus_baseline": p, "adopt_candidate": adopt,
            "ci_includes_zero": p["ci95"][0] <= 0 <= p["ci95"][1]}


def checkpoint_info(path: Path) -> dict:
    import torch

    c = torch.load(path, map_location="cpu", weights_only=False)
    return {"epochs_done": int(c["current_epoch"]), "best_ema_pseudo_dice": c.get("_best_ema"),
            "mirror_axes": list(c.get("inference_allowed_mirroring_axes") or []), "logging": c.get("logging", {})}


# ---------------------------------------------------------------- commands

def cmd_check(a) -> None:
    print("python  ", sys.version.split()[0], "on", platform.system(), platform.machine())
    try:
        import torch

        print("torch   ", torch.__version__, "| device auto ->", pick_device("auto"))
    except ImportError:
        print("torch    NOT INSTALLED")
    try:
        from importlib.metadata import version

        import nnunetv2  # noqa: F401

        print("nnunetv2", version("nnunetv2"), "(project was preprocessed with 2.8.1)")
    except ImportError:
        print("nnunetv2 NOT INSTALLED")
    for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_extTrainer"):
        print(f"{k:20s}", os.environ[k])
    for name in DATASETS.values():
        raw = Path(os.environ["nnUNet_raw"]) / name
        pre = Path(os.environ["nnUNet_preprocessed"]) / name
        n_tr = len(list((raw / "imagesTr").glob("*.png"))) if raw.exists() else 0
        n_ts = len(list((raw / "imagesTs").glob("*.png"))) if raw.exists() else 0
        plans = sorted(p.stem for p in pre.glob("*Plans*.json")) if pre.exists() else []
        print(f"{name}: imagesTr {n_tr}, imagesTs {n_ts}, preprocessed {pre.exists()}, "
              f"splits_final {(pre / 'splits_final.json').exists()}, plans {plans}")
    try:
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name as find

        names = sorted(set(re.findall(r"^(nnUNetTrainer\w+)\s*=|^class (nnUNetTrainer\w+)\(",
                                      "\n".join(f.read_text() for f in TRAINERS_DIR.glob("*.py")), re.M)))
        names = sorted({x for pair in names for x in pair if x})
        import contextlib
        import io

        found = []
        for n in names:
            with contextlib.redirect_stdout(io.StringIO()):
                find(n)
            found.append(n)
        print("our trainers found by nnU-Net:", ", ".join(found))
    except Exception as e:  # noqa: BLE001
        print("our trainers: lookup failed:", e)
    if platform.system() == "Darwin":
        print("power:", power_problem() or "ok (on charger, Low Power Mode off)")
    else:
        print("power: not checked on this system")


def cmd_preprocess(a) -> None:
    if a.planner:
        if a.planner not in PLANNERS:
            sys.exit(f"--planner must be one of {list(PLANNERS)}")
        # reuses the existing dataset fingerprint; only plans and preprocesses for the new planner
        run([tool("nnUNetv2_plan_experiment"), "-d", *DATASETS.keys(), "-pl", a.planner])
        run([tool("nnUNetv2_preprocess"), "-d", *DATASETS.keys(), "-plans_name", PLANNERS[a.planner], "-c", "2d"])
    else:
        run([tool("nnUNetv2_plan_and_preprocess"), "-d", *DATASETS.keys(), "-c", "2d", "--verify_dataset_integrity"])
    copy_splits()


def cmd_train(a, trainer: str | None = None, tee: Path | None = None) -> None:
    require_power(a)
    copy_splits()  # make sure nnU-Net uses the team's folds, not its own random ones
    run([tool("nnUNetv2_train"), a.dataset, "2d", str(a.fold), "-tr", trainer or a.trainer, "-p", a.plans,
         "-device", pick_device(a.device), "--npz"] + (["--c"] if a.resume else []), tee=tee)


def parse_log(log: Path) -> dict:
    stamp = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?): (.*)$")
    from datetime import datetime

    epoch_times, last_epoch_ts, last_ts, notes = [], None, None, []
    for line in log.read_text(errors="replace").splitlines():
        m = stamp.match(line)
        if not m:
            continue
        ts, msg = datetime.fromisoformat(m.group(1)), m.group(2)
        last_ts = ts
        t = re.match(r"Epoch time: ([\d.]+) s", msg)
        if t:
            epoch_times.append(float(t.group(1)))
            last_epoch_ts = ts
        if any(k in msg for k in ("split file", "Desired fold", "This split has", "do_dummy_2d", "USAug")):
            notes.append(msg)
    steady = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    val_s = (last_ts - last_epoch_ts).total_seconds() if last_epoch_ts and last_ts else None
    return {"epoch_times_s": epoch_times, "mean_s_per_epoch_excluding_first": sum(steady) / len(steady) if steady else None,
            "seconds_after_last_epoch": val_s, "log_notes": notes}


def cmd_timing(a) -> None:
    require_power(a)
    trainer = a.trainer if a.trainer.endswith("_5epochs") else f"{a.trainer}_5epochs"
    v = Variant(a.dataset, trainer, a.plans, a.fold)
    tag = time.strftime("%Y%m%d_%H%M%S") + f"__{a.dataset}__{trainer}" + (
        f"__{a.plans}" if a.plans != DEFAULT_PLANS else "")
    out = TIMING_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    (out / "meta.json").write_text(json.dumps({
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": v.dataset, "trainer": trainer, "plans": a.plans,
        "fold": a.fold, "device": pick_device(a.device), "machine": platform.platform(),
        "nnUNet_n_proc_DA": os.environ.get("nnUNet_n_proc_DA", "not set (nnU-Net default)"),
        "power_problem": power_problem()}, indent=2))
    cmd_train(a, trainer=trainer, tee=out / "console.txt")
    md = v.model_dir()
    logs = [p for p in md.glob("training_log_*.txt") if p.stat().st_mtime >= start - 1]
    if not logs:
        sys.exit(f"no new training log in {md}")
    log = max(logs, key=lambda p: p.stat().st_mtime)
    shutil.copyfile(log, out / log.name)
    if (md / "progress.png").exists():
        shutil.copyfile(md / "progress.png", out / "progress.png")
    s = parse_log(log)
    s["wall_clock_s"] = round(time.time() - start, 1)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"\n=== timing ({tag})")
    for n in s["log_notes"]:
        print("  log:", n)
    print("  epoch times (s):", ", ".join(f"{t:.1f}" for t in s["epoch_times_s"]))
    m = s["mean_s_per_epoch_excluding_first"]
    if m:
        print(f"  mean seconds per epoch (epochs 2+): {m:.1f}")
        print(f"  after the last epoch (final validation and saving): {s['seconds_after_last_epoch']:.0f} s")
        for n in (250, 500, 750, 1000):
            print(f"  {n:5d} epochs: about {n * m / 3600:.1f} h of training (lower bound)")
    print(f"  saved to {out}")


def cmd_predict(a) -> None:
    predict(variant_from_args(a), a.device, a.force)


def cmd_postprocess(a) -> None:
    postprocess(variant_from_args(a, lcc=False))


def cmd_evaluate(a) -> None:
    evaluate(variant_from_args(a), poor_below=a.poor_below)


def nnunet_postprocessing_choice(v: Variant) -> dict:
    """nnU-Net's own clean-up decision, pinned to this one model (no ensembling, only this fold)."""
    run([tool("nnUNetv2_find_best_configuration"), v.ds, "-c", "2d", "-tr", v.trainer, "-p", v.plans,
         "-f", str(v.fold), "--disable_ensembling"])
    folder = v.model_dir().parent / f"crossval_results_folds_{v.fold}"
    pp = json.loads((folder / "postprocessing.json").read_text())
    return {"command": f"nnUNetv2_find_best_configuration {v.ds} -c 2d -tr {v.trainer} -p {v.plans} "
                       f"-f {v.fold} --disable_ensembling",
            "decided_on": f"{v.model_dir() / 'validation'} (final checkpoint, flip averaging on)",
            "postprocessing_fns": pp.get("postprocessing_fns", []),
            "keeps_largest_region": any("largest" in f for f in pp.get("postprocessing_fns", [])),
            "file": str(folder / "postprocessing.json")}


def cmd_select(a) -> None:
    base = Variant(a.dataset, a.trainer, a.plans, a.fold, on="val")
    if not (base.model_dir() / "checkpoint_final.pth").exists():
        sys.exit(f"no finished training in {base.model_dir()}")
    predict(base, a.device)
    r0 = evaluate(base, poor_below=a.poor_below)
    notta = replace(base, tta=False)
    predict(notta, a.device, a.force)
    r1 = evaluate(notta, poor_below=a.poor_below)
    d_tta = decide("flip averaging: turn it off?", base.label(), r0, notta.label(), r1)
    chosen, rc = (notta, r1) if d_tta["adopt_candidate"] else (base, r0)
    cleaned = replace(chosen, lcc=True)
    pp_stats = postprocess(chosen)
    r2 = evaluate(cleaned, poor_below=a.poor_below)
    d_lcc = decide("clean-up: keep only the largest region?", chosen.label(), rc, cleaned.label(), r2)
    nn = nnunet_postprocessing_choice(base)
    note = None
    if not chosen.tta:
        note = ("Our rule turned flip averaging off, but nnU-Net's clean-up step decided on its default validation "
                "result with flip averaging on, so the two clean-up decisions use different bases. Our decision counts.")
    elif nn["keeps_largest_region"] != d_lcc["adopt_candidate"]:
        note = ("nnU-Net's step and ours disagree on the same base. nnU-Net uses its own Dice computation; "
                "our decision (evaluate.py) counts.")
    sel = {
        "dataset": base.dataset, "trainer": a.trainer, "plans": a.plans, "fold": a.fold,
        "n_validation_images": len(r0), "decided": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": "adopt a variant only if it raises mean validation Dice; flip averaging first, then the clean-up",
        "mirror_axes_of_this_trainer": checkpoint_info(base.model_dir() / "checkpoint_final.pth")["mirror_axes"],
        "flip_averaging": {**d_tta, "use_flip_averaging": not d_tta["adopt_candidate"]},
        "cleanup": {**d_lcc, "use_cleanup": d_lcc["adopt_candidate"], "base_tta": chosen.tta,
                    "postprocess_stats": {k: v for k, v in pp_stats.items() if k != "changed"}},
        "nnunet_cleanup_step": nn, "note": note,
        "selected": replace(chosen, lcc=d_lcc["adopt_candidate"], on="test").name(),
    }
    path = PRED_ROOT / f"{base.name().removesuffix('__val')}_selection.json"
    path.write_text(json.dumps(sel, indent=2))
    print(f"\n=== selection on {len(r0)} validation images: flip averaging "
          f"{'on' if sel['flip_averaging']['use_flip_averaging'] else 'off'}, clean-up "
          f"{'on' if sel['cleanup']['use_cleanup'] else 'off'}; nnU-Net's own step: clean-up "
          f"{'on' if nn['keeps_largest_region'] else 'off'}")
    if note:
        print("    note:", note)
    print(f"    written to {path}")


def cmd_test_variants(a) -> None:
    import evaluate as ev

    t0 = Variant(a.dataset, a.trainer, a.plans, a.fold, on="test")
    sel_path = PRED_ROOT / f"{t0.name()}_selection.json"
    if not sel_path.exists():
        sys.exit(f"missing {sel_path}: run `select` first, so the choice is made on validation")
    sel = json.loads(sel_path.read_text())
    t1, t2 = replace(t0, tta=False), replace(t0, checkpoint="best")
    chosen = t0 if sel["flip_averaging"]["use_flip_averaging"] else t1
    t3 = replace(chosen, lcc=True)
    for v in (t0, t1, t2):
        predict(v, a.device, a.force)
    postprocess(chosen)
    r0 = evaluate(t0, poor_below=a.poor_below)
    results = [(t0, r0, "default (nnU-Net's inference)")]
    for v, note in ((t1, "flip averaging off"), (t2, "overfitting evidence only, not a selection"),
                    (t3, "clean-up on the flip-averaging setting chosen on validation")):
        results.append((v, evaluate(v, baseline=(t0.label(), r0), poor_below=a.poor_below), note))
    selected = sel["selected"]
    table = []
    print(f"\n=== {t0.dataset} test images, {a.trainer}: every variant (selected on validation: {selected})")
    print(f"  {'variant':70s} {'mean':>7s} {'95% CI':>17s} {'vs default':>10s} {'diff 95% CI':>19s}")
    for v, rows, note in results:
        s = ev.summary(rows)
        p = ev.paired(rows, r0) if v != t0 else None
        mark = "*" if v.name() == selected else " "
        print(f"{mark} {v.label()[:70]:70s} {s['mean_dice']:7.4f} {s['ci95'][0]:8.4f}-{s['ci95'][1]:.4f} "
              + (f"{p['mean_diff']:+10.4f} {p['ci95'][0]:+9.4f}-{p['ci95'][1]:+.4f}" if p else f"{'':>10s}")
              + f"  {note}")
        table.append({"variant": v.name(), "description": v.label(), "note": note, "selected_on_validation":
                      v.name() == selected, **{k: s[k] for k in ("n", "mean_dice", "median_dice", "min_dice",
                                                                 "mean_iou")},
                      "ci95_lo": s["ci95"][0], "ci95_hi": s["ci95"][1],
                      "diff_vs_default": p["mean_diff"] if p else 0.0,
                      "diff_ci95_lo": p["ci95"][0] if p else "", "diff_ci95_hi": p["ci95"][1] if p else ""})
    print("* = the variant chosen on the validation images")
    out = PRED_ROOT / f"{t0.name()}_variants.csv"
    ev.write_csv(table, out)
    print(f"written to {out}")


def cmd_compare(a) -> None:
    if not a.vs:
        sys.exit("compare needs --vs TRAINER[:PLANS]")
    vs_trainer, _, vs_plans = a.vs.partition(":")
    A = Variant(a.dataset, a.trainer, a.plans, a.fold, on="val")
    B = Variant(a.dataset, vs_trainer, vs_plans or DEFAULT_PLANS, a.fold, on="val")
    rA, rB = evaluate(A, poor_below=a.poor_below), evaluate(B, poor_below=a.poor_below)
    d = decide(f"use {B.label()} instead of {A.label()}?", A.label(), rA, B.label(), rB)
    tA, tB = replace(A, on="test"), replace(B, on="test")
    predict(tA, a.device, a.force)
    predict(tB, a.device, a.force)
    import evaluate as ev

    sA = ev.summary(evaluate(tA, poor_below=a.poor_below))
    rtB = evaluate(tB, baseline=(tA.label(), ev.read_csv(tA.csv())), poor_below=a.poor_below)
    res = {"dataset": A.dataset, "fold": a.fold, "rule": "adopt only if mean validation Dice is higher; "
           "both are reported on test", "validation": d,
           "test": {"baseline": tA.label(), "baseline_summary": sA, "candidate": tB.label(),
                    "candidate_summary": ev.summary(rtB),
                    "paired_candidate_minus_baseline": ev.paired(rtB, ev.read_csv(tA.csv()))}}
    path = PRED_ROOT / f"compare__{tA.name()}__vs__{tB.name().removeprefix(A.dataset + '__')}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\n=== validation decision: {'ADOPT ' + B.trainer if d['adopt_candidate'] else 'keep ' + A.trainer}; "
          f"test: {sA['mean_dice']:.4f} vs {res['test']['candidate_summary']['mean_dice']:.4f}\n    written to {path}")


def cmd_ensemble(a) -> None:
    """Option C: the average of several folds' models on the test images, next to the single fold-<f> model, plus
    the cross-validation Dice (every training image predicted by the one model that did not train on it).

    The ensemble has no untouched validation images (each fold's validation images trained the other models), so
    it is not chosen by the validation rule. It is reported on test next to the single model, and the headline
    stays the single fold-0 model."""
    import evaluate as ev

    folds = sorted(set(a.folds))
    if len(folds) < 2 or a.fold not in folds:
        sys.exit(f"--folds needs at least two folds including the single model's fold {a.fold}")
    single = Variant(a.dataset, a.trainer, a.plans, a.fold, on="test", tta=not a.no_tta)
    ckpts = [replace(single, fold=f).model_dir() / "checkpoint_final.pth" for f in folds]
    missing = [str(c) for c in ckpts if not c.exists()]
    if missing:
        sys.exit("not trained yet:\n  " + "\n  ".join(missing))
    tag = "__folds" + "".join(map(str, folds))
    ens_name = single.name().replace(f"__fold{a.fold}", tag, 1)

    # 1. test images: ensemble vs the single model, same checkpoint (final) and same flip-averaging setting
    images, labels = inputs(single)
    out = PRED_ROOT / ens_name
    expected = [out / f.name for f in labels.glob("*.png")]
    newest = max(c.stat().st_mtime for c in ckpts)
    if not a.force and expected and all(p.exists() for p in expected) and \
            min(p.stat().st_mtime for p in expected) >= newest:
        print(f"up to date, reusing {out} (pass --force to predict again)")
    else:
        if out.exists():
            shutil.rmtree(out)  # generated folder
        run([tool("nnUNetv2_predict"), "-i", str(images), "-o", str(out), "-d", a.dataset, "-c", "2d",
             "-f", *map(str, folds), "-tr", a.trainer, "-p", a.plans, "-chk", "checkpoint_final.pth",
             "-device", pick_device(a.device)] + ([] if single.tta else ["--disable_tta"]))
    predict(single, a.device, a.force)
    r_single = evaluate(single, poor_below=a.poor_below)
    print(f"\n=== test: ensemble of folds {folds}\n    {out}")
    r_ens = ev.score(out, labels, a.poor_below)
    ev.report(r_ens, a.poor_below)
    p_test = ev.paired(r_ens, r_single)
    print(f"vs {single.label()}, fold {a.fold}: " + ev.paired_text(p_test))
    ev.write_csv(r_ens, PRED_ROOT / f"{ens_name}_dice.csv")

    # 2. cross-validation: merge the folds' own validation results (nnU-Net's, final checkpoint, flip averaging on)
    ds = DATASETS[a.dataset]
    prefix = f"{ds}__{a.trainer}" + ("" if a.plans == DEFAULT_PLANS else f"__{a.plans}")
    cv_dir = PRED_ROOT / f"{prefix}__cv_folds{''.join(map(str, folds))}"
    if cv_dir.exists():
        shutil.rmtree(cv_dir)  # generated folder
    (cv_dir / "labels").mkdir(parents=True)
    raw = Path(os.environ["nnUNet_raw"]) / ds
    per_fold, cases_seen = {}, set()
    for f in folds:
        vdir = replace(single, fold=f).model_dir() / "validation"
        cases = val_cases(ds, f)
        if not vdir.exists():
            sys.exit(f"missing {vdir} (nnU-Net writes it at the end of training)")
        dup = cases_seen.intersection(cases)
        if dup:
            sys.exit(f"cases in more than one fold's validation set: {sorted(dup)[:5]}")
        cases_seen.update(cases)
        for c in cases:
            src = vdir / f"{c}.png"
            if not src.exists():
                sys.exit(f"missing {src}")
            shutil.copyfile(src, cv_dir / f"{c}.png")
            shutil.copyfile(raw / "labelsTr" / f"{c}.png", cv_dir / "labels" / f"{c}.png")
    r_cv = ev.score(cv_dir, cv_dir / "labels", a.poor_below)
    by_case = {r["case"]: r for r in r_cv}
    for f in folds:
        per_fold[f] = ev.summary([by_case[c] for c in val_cases(ds, f)])
    n_train = len(list((raw / "labelsTr").glob("*.png")))
    print(f"\n=== cross-validation, folds {folds}: {len(r_cv)} of {n_train} training images, each predicted by "
          f"the model that did not train on it (single models, not the ensemble)")
    ev.report(r_cv, a.poor_below)
    for f, s in per_fold.items():
        print(f"  fold {f}: {s['n']:3d} images, mean Dice {s['mean_dice']:.4f} "
              f"(95% CI {s['ci95'][0]:.4f}-{s['ci95'][1]:.4f})")
    ev.write_csv(r_cv, PRED_ROOT / f"{cv_dir.name}_dice.csv")

    res = {"dataset": ds, "trainer": a.trainer, "plans": a.plans, "folds": folds, "flip_averaging": single.tta,
           "rule": "not chosen by the validation rule (no untouched validation images for an ensemble); reported "
                   f"on test next to the single fold-{a.fold} model, which stays the headline",
           "test": {"single": single.name(), "single_summary": ev.summary(r_single), "ensemble": ens_name,
                    "ensemble_summary": ev.summary(r_ens), "paired_ensemble_minus_single": p_test},
           "cross_validation": {"n_images": len(r_cv), "n_training_images": n_train, "summary": ev.summary(r_cv),
                                "per_fold": {str(f): s for f, s in per_fold.items()},
                                "what": "nnU-Net's own validation result of each fold (final checkpoint, flip "
                                        "averaging on), single models"}}
    path = PRED_ROOT / f"{ens_name}_ensemble.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\n=== test: single {res['test']['single_summary']['mean_dice']:.4f}, ensemble "
          f"{res['test']['ensemble_summary']['mean_dice']:.4f}; CV over {len(r_cv)} images "
          f"{res['cross_validation']['summary']['mean_dice']:.4f}\n    written to {path}")


def cmd_records(a) -> None:
    """Copy the small result files into docs/results/<machine>/ so they can be committed. No images: only the
    per-image Dice tables, decisions, training logs and nnU-Net's progress plots (5-epoch runs are in docs/timing)."""
    if not a.machine or not re.fullmatch(r"[A-Za-z0-9_-]+", a.machine):
        sys.exit("records needs --machine NAME (letters, digits, - or _), e.g. mac or workstation")
    dst = NERVESEG / "docs" / "results" / a.machine
    n = 0
    if PRED_ROOT.exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(list(PRED_ROOT.glob("*.csv")) + list(PRED_ROOT.glob("*.json"))):
            shutil.copyfile(f, dst / f.name)
            n += 1
    for fold_dir in sorted(Path(os.environ["nnUNet_results"]).glob("Dataset*/*__2d/fold_*")):
        if "_5epochs" in fold_dir.parent.name:
            continue
        files = list(fold_dir.glob("training_log_*.txt")) + [p for p in [fold_dir / "progress.png"] if p.exists()]
        if not files:
            continue
        sub = dst / "logs" / f"{fold_dir.parent.parent.name}__{fold_dir.parent.name}__{fold_dir.name}"
        sub.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copyfile(f, sub / f.name)
            n += 1
    print(f"copied {n} files to {dst}\nthen: git add {dst.relative_to(NERVESEG.parent)} && git commit && git push")


def cmd_curves(a) -> None:
    v = Variant(a.dataset, a.trainer, a.plans, a.fold)
    md = v.model_dir()
    src = md / "progress.png"
    if not src.exists():
        sys.exit(f"missing {src}")
    FIGURES.mkdir(exist_ok=True)
    if a.machine and not re.fullmatch(r"[A-Za-z0-9_-]+", a.machine):
        sys.exit("--machine: letters, digits, - or _ only")
    dst = FIGURES / ((f"{a.machine}_" if a.machine else "") + f"{a.dataset}_{a.trainer}"
                     + (f"_{a.plans}" if a.plans != DEFAULT_PLANS else "") + f"_fold{a.fold}_progress.png")
    shutil.copyfile(src, dst)
    print(f"copied {src} -> {dst}")
    print("(nnU-Net's plot: training and validation loss, and its pseudo Dice on validation patches during "
          "training; this is not the Dice of evaluate.py)")
    for ck in ("checkpoint_best.pth", "checkpoint_final.pth"):
        if (md / ck).exists():
            info = checkpoint_info(md / ck)
            print(f"{ck}: saved after epoch {info['epochs_done']}, best EMA pseudo Dice so far "
                  f"{info['best_ema_pseudo_dice']:.4f}")
    final = md / "checkpoint_final.pth"
    if final.exists():
        lg = checkpoint_info(final)["logging"]
        ema = lg.get("ema_fg_dice", [])
        if ema:
            k = max(range(len(ema)), key=lambda i: ema[i])
            print(f"EMA pseudo Dice: highest {ema[k]:.4f} at epoch {k + 1} of {len(ema)}, last {ema[-1]:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["check", "preprocess", "timing", "train", "predict", "postprocess", "evaluate",
                                     "select", "test-variants", "compare", "curves", "ensemble", "records"])
    ap.add_argument("-d", "--dataset", choices=list(DATASETS), default="501")
    ap.add_argument("-t", "--trainer", default="nnUNetTrainer")
    ap.add_argument("-p", "--plans", default=DEFAULT_PLANS)
    ap.add_argument("-f", "--fold", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--checkpoint", default="final",
                    choices=["final", "best", "checkpoint_final.pth", "checkpoint_best.pth"])
    ap.add_argument("--on", default="test", choices=["test", "val"],
                    help="test images, or the fold's validation images")
    ap.add_argument("--no-tta", action="store_true", help="turn off nnU-Net's test-time flip averaging")
    ap.add_argument("--lcc", action="store_true", help="evaluate: score the largest-region (cleaned) folder")
    ap.add_argument("--force", action="store_true", help="predict again even if up-to-date predictions exist")
    ap.add_argument("--poor-below", type=float, default=0.5, help="Dice below this (but > 0) counts as 'poor'")
    ap.add_argument("--vs", help="compare: the other model, TRAINER or TRAINER:PLANS")
    ap.add_argument("--planner", help="preprocess: an extra planner, e.g. nnUNetPlannerResEncM")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted training run")
    ap.add_argument("--ignore-power", action="store_true", help="train/time even on battery or in Low Power Mode")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="ensemble: the folds to average")
    ap.add_argument("--machine", help="records: folder name under docs/results/, e.g. mac or workstation; "
                                      "curves: prefix for the figure name")
    a = ap.parse_args()
    set_env()
    {"check": cmd_check, "preprocess": cmd_preprocess, "timing": cmd_timing, "train": cmd_train,
     "predict": cmd_predict, "postprocess": cmd_postprocess, "evaluate": cmd_evaluate, "select": cmd_select,
     "test-variants": cmd_test_variants, "compare": cmd_compare, "curves": cmd_curves, "ensemble": cmd_ensemble,
     "records": cmd_records}[a.step](a)


if __name__ == "__main__":
    main()
