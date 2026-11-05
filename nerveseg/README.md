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
| `nnunet/predictions/` | predictions, per-image Dice CSVs, `*_selection.json`, `*_variants.csv`, `*_ensemble.json` (not in git) |
| `docs/results/<machine>/` | the small result files copied by `run.py records` (Dice tables, decisions, training logs), in git |
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
python nerveseg/nnunet/run.py preprocess --planner nnUNetPlannerResEncM   # needed for option B (ResEnc M)
```
The rebuild keeps the saved split files, so both machines use exactly the same splits; `git status` should show no change in `data/splits/` or `nnunet/splits/`. Raw images contain patient details: do not upload `data/raw/` or the course zip anywhere public.

## On the NVIDIA workstation: options B and C (from Thu 5 Nov)
E1, E2 and E3 train on the Mac. The workstation trains its own default run **W0**, then **B** (ResEnc M) and **C** (5-fold ensemble). B and C are compared only with W0, never with a Mac number, because the two machines also differ in precision (the Mac runs 32-bit; NVIDIA runs mixed 16/32-bit) and in `torch.compile`. W0 vs E1 (same setup, two machines) shows how much Dice moves between two runs. Plan and rules: `PLAN.md`, "Workstation: B and C".

1. Get the code (the cropped data comes with it; the course zip and raw images are not in git and are not needed):
   ```
   git clone https://github.com/nrchnnas/aic604-nerveseg comp-vision
   cd comp-vision
   ```
   Already cloned: `git pull`. Set your git name and email for this clone (`git config user.name ...`, `git config user.email ...`).
2. Environment ("Setup" above; Linux is the same as Windows): Python 3.11, the CUDA install command from pytorch.org (any torch except 2.9.x), then `pip install -r nerveseg/requirements.txt`. Save the versions:
   ```
   pip freeze > nerveseg/docs/env_workstation.txt
   python -c "import platform, torch; print(platform.platform(), 'torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" >> nerveseg/docs/env_workstation.txt
   nvidia-smi >> nerveseg/docs/env_workstation.txt
   ```
3. Preprocess and check (from the `comp-vision` folder):
   ```
   python nerveseg/nnunet/run.py preprocess
   python nerveseg/nnunet/run.py preprocess --planner nnUNetPlannerResEncM
   python nerveseg/nnunet/run.py check
   git status
   ```
   `check` must print device `cuda` and find the `nnUNetTrainerUSAug*` classes. `git status` must show no change in `nerveseg/data/splits/`, `nerveseg/nnunet/splits/` or `nerveseg/nnunet/nnUNet_raw/`.
4. Time it (5 epochs each):
   ```
   python nerveseg/nnunet/run.py timing -d 501
   python nerveseg/nnunet/run.py timing -d 501 -p nnUNetResEncUNetMPlans
   ```
   Each prints seconds per epoch and the hours for 250 epochs. Note each run's GPU memory in `nvidia-smi`. If the two together fit in the card's memory, time them again at the same time (the same two commands in two terminals, started a minute apart). Add a "Workstation" section to `docs/timing/SUMMARY.md` with the time per epoch alone and together.
5. Decide what fits before **Mon 9 Nov 08:00** (report section 4 is written on Monday). All runs are 250 epochs on 501:
   ```
   python nerveseg/nnunet/run.py train -d 501 -t nnUNetTrainer_250epochs                             # W0 = fold 0
   python nerveseg/nnunet/run.py train -d 501 -t nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans   # B
   python nerveseg/nnunet/run.py train -d 501 -t nnUNetTrainer_250epochs -f 1                        # C: folds 1 to 4
   python nerveseg/nnunet/run.py train -d 501 -t nnUNetTrainer_250epochs -f 2
   python nerveseg/nnunet/run.py train -d 501 -t nnUNetTrainer_250epochs -f 3
   python nerveseg/nnunet/run.py train -d 501 -t nnUNetTrainer_250epochs -f 4
   ```
   W0 is needed by both B and C. B costs 1 run plus W0; C costs 4 runs plus W0. If only part fits, do W0 and B, and C only if all 4 folds fit.
   - Runs can train at the same time, each in its own terminal. That changes the speed, not the comparison: B and C are still compared with W0 on this machine.
   - It pays off when each run's seconds per epoch with N runs at once is less than N times its time alone (step 4 measures this for N = 2). On the Mac, two at once gave only about 12% more total work.
   - Limits: the runs' GPU memory together must fit the card; each run starts 12 data-loading workers plus itself, so if the CPU has fewer threads than runs x 13, set `nnUNet_n_proc_DA` lower (for example 6) before starting.
   - Suggested grouping: W0 and B together first (B's comparison needs both), then folds 1 to 4, two at a time (or all four if memory and CPU allow).
   - Sharing the data is safe: the preprocessed data is in blosc2 format (`.b2nd`), which nnU-Net only reads during training (its unpack step does nothing for this format; checked in nnunetv2 2.8.1). Each run writes to its own folder. Start runs about a minute apart, because each start rewrites `splits_final.json` (same content). Do not run `preprocess` while anything trains.
   - With two GPUs, put one run on each: `CUDA_VISIBLE_DEVICES=1 python ...` (Linux) or `$env:CUDA_VISIBLE_DEVICES=1` first (PowerShell).
   - Keep the machine from sleeping (Windows: power plan "never sleep" while plugged in).
   - An interrupted run continues with the same command plus `--resume`.
   - If `torch.compile` fails on Linux, set `nnUNet_compile=f` and note it in `env_workstation.txt`.
6. Analyses, as each run finishes (rules in `PLAN.md`, "Analyses on every trained run"):
   ```
   python nerveseg/nnunet/run.py curves        -d 501 -t nnUNetTrainer_250epochs --machine workstation
   python nerveseg/nnunet/run.py select        -d 501 -t nnUNetTrainer_250epochs
   python nerveseg/nnunet/run.py test-variants -d 501 -t nnUNetTrainer_250epochs
   python nerveseg/nnunet/run.py curves        -d 501 -t nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --machine workstation
   python nerveseg/nnunet/run.py select        -d 501 -t nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
   python nerveseg/nnunet/run.py test-variants -d 501 -t nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
   python nerveseg/nnunet/run.py compare       -d 501 -t nnUNetTrainer_250epochs --vs nnUNetTrainer_250epochs:nnUNetResEncUNetMPlans   # B vs W0
   python nerveseg/nnunet/run.py ensemble      -d 501 -t nnUNetTrainer_250epochs   # C vs W0 on test, plus the 5-fold CV Dice
   ```
   `select` must run before `test-variants`. `ensemble` needs folds 0 to 4 finished.
7. Bring the results back through git (no images, only Dice tables, decisions, logs and plots):
   ```
   python nerveseg/nnunet/run.py records --machine workstation
   git pull --rebase
   git add nerveseg/docs/results/workstation nerveseg/docs/env_workstation.txt nerveseg/docs/timing nerveseg/figures
   git commit -m "Workstation: <what finished>"
   git push
   ```
   Do this after the timing runs, and again whenever a run finishes, so the Mac side can follow.

### Using Claude Code on the workstation
Open Claude Code in the `comp-vision` folder and paste:
> Read nerveseg/PLAN.md and nerveseg/README.md, section "On the NVIDIA workstation: options B and C". Do steps 1 to 4, then report the measured seconds per epoch and an hours table (lower bound), and recommend what fits before Mon 9 Nov 08:00, including which runs to train at the same time (step 5). Do not start the long training runs until I agree. Keep every comparison on this machine (B and C vs W0). Records go in the repo (step 7), not a scratchpad. Never commit the course zip or nerveseg/data/raw/.

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
