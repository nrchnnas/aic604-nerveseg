# nerveseg: nerve segmentation in ultrasound with nnU-Net (AIC-604)

Team plan, work split and timeline: `PLAN.md`. Facts for the report: `docs/nnunet_facts.md`. Reading list: `docs/resources.md`.

## What is in this folder
| Path | What it is |
|---|---|
| `prepare_data.py` | extracts `docs/final project/nerve (1).zip`, crops each image to the scan area (removes the patient header), makes masks, finds near-duplicate clusters, writes the splits |
| `data/splits/*.json` | the team's splits (shared, do not regenerate): `required` (80/20, seed 1), `required_with_val`, `grouped` |
| `data/dataset_report.json`, `data/meta.csv` | dataset numbers and one row per image |
| `nnunet/make_nnunet_dataset.py` | converts the data into nnU-Net format (Dataset501 and Dataset502) |
| `nnunet/run.py` | one command for every nnU-Net step, same on Mac and Windows |
| `nnunet/trainers/` | our trainer classes (E3 candidate A: `nnUNetTrainerUSAug`); run.py tells nnU-Net to look here |
| `postprocess.py` | clean-up: keeps only the largest predicted region in each mask |
| `nnunet/splits/*_splits_final.json` | the 5 cross-validation folds nnU-Net uses (near-duplicates kept together) |
| `nnunet/env.sh`, `nnunet/env.ps1` | only needed to call `nnUNetv2_*` commands by hand (Mac/Linux, Windows) |
| `evaluate.py` | Dice per image, mean with 95% CI, leakage breakdown, failure types, Dice by nerve size, paired comparisons |
| `figures.py` | success and failure figures, Dice histograms, experiment comparison plots, Dice vs nerve size |
| `docs/env_mac.txt` | exact package versions on Nana's Mac (`pip freeze`) |
| `docs/timing/` | logs and summaries of the 5-epoch timing runs |
| `nnunet/predictions/` | predictions, per-image Dice CSVs, `*_selection.json`, `*_variants.csv` |
| `nnunet/val_inputs/` | copies of each fold's validation images and labels (made by run.py) |
| `figures/` | figures for the report and slides (all use cropped images, no patient details) |
| `data/yolo/` | kept from an earlier plan (YOLO + MedSAM); not used |
| `_to_delete/` | old files; safe to delete |

Datasets:
- `Dataset501_NerveRequired`: the instructor's 80/20 split. 240 training images, 60 test images. Same call as Lab 3: `train_test_split(sorted file names, test_size=0.2, random_state=1)`.
- `Dataset502_NerveGrouped`: whole near-duplicate clusters kept on one side. 237 training, 63 test.

## Setup
Python 3.11 in its own environment. Install PyTorch first, because it differs by machine.

macOS (Apple Silicon):
```
conda create -n nerveseg python=3.11 -y
conda activate nerveseg
pip install torch torchvision
pip install -r nerveseg/requirements.txt
```
nnunetv2 2.8.1 refuses torch 2.9.x; any other recent torch works. Nana's exact versions are in `docs/env_mac.txt`.

Windows (PowerShell, Anaconda Prompt or similar):
```
conda create -n nerveseg python=3.11 -y
conda activate nerveseg
# NVIDIA GPU: copy the CUDA install command from https://pytorch.org/get-started/locally/
# no NVIDIA GPU: pip install torch torchvision   (CPU only; fine for prediction and figures, too slow to train)
pip install -r nerveseg/requirements.txt
```

Then, from the `comp-vision` folder on either system:
```
python nerveseg/nnunet/run.py check
```
It prints the device it will use (cuda, mps or cpu) and whether the datasets and split files are in place.

## Getting the data onto another machine
The git repo holds the code, docs, split files and the cropped data (`data/images_roi`, `data/masks_roi`, `nnunet/nnUNet_raw`; header removed), but not the raw images or the course zip. After cloning, run `python nerveseg/nnunet/run.py preprocess` (plus `--planner nnUNetPlannerResEncM` if needed). The generated folders (`data/`, `nnunet/nnUNet_raw/`, `nnunet/nnUNet_preprocessed/`) are about 0.7 GB. Either copy the whole `nerveseg` folder (Google Drive), or put the course zip at `nerveseg/docs/final project/nerve (1).zip` and rebuild:
```
python nerveseg/prepare_data.py
python nerveseg/nnunet/make_nnunet_dataset.py
python nerveseg/nnunet/run.py preprocess
python nerveseg/nnunet/run.py preprocess --planner nnUNetPlannerResEncM   # only if E3-B (ResEnc M) is used
```
The rebuild keeps the saved split files, so both machines use exactly the same splits; `git status` should show no change in `data/splits/` or `nnunet/splits/`. Raw images contain patient details: do not upload `data/raw/` or the course zip anywhere public.

## On a new machine (for example an NVIDIA workstation)
1. `git clone https://github.com/nrchnnas/aic604-nerveseg` (private). The cropped data comes with it. The course zip and raw images are not in git.
2. Set up the environment ("Setup" above). On NVIDIA, use the CUDA command from pytorch.org; any version except 2.9.x works. Save `pip freeze`, plus the GPU, driver, CUDA and OS versions, as `docs/env_<machine>.txt`.
3. From the `comp-vision` folder:
   ```
   python nerveseg/nnunet/run.py preprocess
   python nerveseg/nnunet/run.py preprocess --planner nnUNetPlannerResEncM
   python nerveseg/nnunet/run.py check
   ```
   `check` should print device `cuda` and list the `nnUNetTrainerUSAug*` classes. `git status` should show no change in `data/splits/`, `nnunet/splits/` or `nnunet/nnUNet_raw/`.
4. Time it before choosing the training length: `run.py timing -d 501`, then `-t nnUNetTrainerUSAug` (E3-A) and `-p nnUNetResEncUNetMPlans` (E3-B). The logs land in `docs/timing/`. Linux gets nnU-Net's `torch.compile` and Windows does not.
5. Train E1, E2 and E3 on that same machine.

## Running experiments (any OS)
```
python nerveseg/nnunet/run.py timing   -d 501                       # 5 epochs; prints seconds per epoch, saves the log in docs/timing/
python nerveseg/nnunet/run.py train    -d 501 -t nnUNetTrainer      # or nnUNetTrainer_250epochs etc.
python nerveseg/nnunet/run.py predict  -d 501 -t nnUNetTrainer
python nerveseg/nnunet/run.py evaluate -d 501 -t nnUNetTrainer
python nerveseg/figures.py cases --dataset 501 --pred nerveseg/nnunet/predictions/Dataset501_NerveRequired__nnUNetTrainer__fold0 --out nerveseg/figures/501_default
```
Repeat with `-d 502` for the grouped split. Use the same `-t` (and `-p`) for train, predict and evaluate. An interrupted run continues with `train ... --resume`.

Compare experiments:
```
python nerveseg/figures.py compare --out nerveseg/figures/compare_splits.png "required=nerveseg/nnunet/predictions/Dataset501_NerveRequired__nnUNetTrainer__fold0_dice.csv" "grouped=nerveseg/nnunet/predictions/Dataset502_NerveGrouped__nnUNetTrainer__fold0_dice.csv"
```

### Analyses on every trained run (rules in PLAN.md, "Analyses on every trained run")
```
python nerveseg/nnunet/run.py curves        -d 501 -t nnUNetTrainer   # nnU-Net's training curves -> figures/, best vs final epoch
python nerveseg/nnunet/run.py select        -d 501 -t nnUNetTrainer   # choices on the validation images -> *_selection.json
python nerveseg/nnunet/run.py test-variants -d 501 -t nnUNetTrainer   # every variant on the test images -> *_variants.csv
python nerveseg/nnunet/run.py compare       -d 501 -t nnUNetTrainer --vs nnUNetTrainerUSAug   # E3 vs default
python nerveseg/figures.py size --out nerveseg/figures/501_default_size.png nerveseg/nnunet/predictions/Dataset501_NerveRequired__nnUNetTrainer__fold0_dice.csv
```
`select` must run before `test-variants`, so the choice is made on validation, not on test. The single steps also exist:
- `predict` takes `--on val`, `--checkpoint best` and `--no-tta`.
- `postprocess` takes the same options.
- `evaluate` takes the same options plus `--lcc`.

For ResEnc M (E3 candidate B), add `-p nnUNetResEncUNetMPlans` after running `python nerveseg/nnunet/run.py preprocess --planner nnUNetPlannerResEncM` once. On the Mac this was done on 4 Nov.

## Before a timing or training run
- Plug in the charger, and keep Low Power Mode off on the charger (System Settings, Battery). On a Mac, run.py refuses to time or train on battery or in Low Power Mode unless you pass `--ignore-power`.
- Keep the computer awake for the whole run (Mac: `caffeinate -i python nerveseg/nnunet/run.py train ...`, lid open) and do not use the GPU for other heavy work. Otherwise seconds per epoch will differ from the timing run.
- The timing runs show 5 epochs only. Over a long run the laptop may slow down as it heats up, so treat the timing estimates as a lower bound.

## Notes per platform
- macOS: training uses the Apple GPU (`mps`). Some operations fall back to the CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by run.py). Results are not bit-for-bit reproducible on MPS.
- Windows: nnU-Net runs on Windows; with an NVIDIA GPU it uses `cuda`. If data loading hangs or errors, set fewer background workers before training: `$env:nnUNet_n_proc_DA = "4"` (PowerShell).
- Everything in this folder uses Python only (no bash), so the same commands work on both systems.

## Dataset facts (measured by prepare_data.py)
- 300 ultrasound images, 864 x 648 (196 JPG, 104 PNG), one nerve outline each. The label is spelled "nerve" and "Nerve".
- 14 annotation files name the wrong image file in `imagePath`; images and labels are paired by number.
- The nerve covers a median 0.31% of the image (0.06% to 1.4%). Outlines are rough (5 to 39 points).
- Image headers show patient ID fragments, partial names and dates; all images used here are cropped to the scan area.
- 212 of 300 images have a near-identical image (pixel correlation above 0.95); 53 clusters at 0.8. In the required split, 43 of 60 test images have a near-identical training image.
