# nnU-Net facts for the report

Everything here was read from nnunetv2 2.8.1 (the version used to preprocess our data) or from the plan files nnU-Net wrote for our datasets, on 4 Nov 2026. Each item says where it comes from. "Paper" means Isensee et al., Nature Methods 2021 (see `resources.md`). Items marked "check in log" should be confirmed from the training log on the machine that trains.

## What nnU-Net is (paper)
- A segmentation method that configures itself for a new dataset: preprocessing, network architecture, training and post-processing.
- Design choices are grouped into fixed parameters (same for every dataset), rule-based parameters (computed from a "dataset fingerprint" with fixed rules) and empirical parameters (chosen from cross-validation results, such as post-processing and which configuration to use).
- The network itself is a plain U-Net. The paper's claim is that careful configuration matters more than architectural changes; the 2024 follow-up ("nnU-Net Revisited") repeats this claim with new benchmarks.

## What nnU-Net chose for our data (`nnUNet_preprocessed/Dataset501_NerveRequired/nnUNetPlans.json`; same for 502)
| Setting | Value | Note |
|---|---|---|
| Configuration | 2d only | our data is 2D PNG |
| Median image size | 544 x 736 px | our ROI crop |
| Spacing | 1 x 1 | PNG has no physical spacing |
| Patch size | 640 x 768 | larger than the image, so one patch holds the whole image (padded) |
| Batch size | 7 | |
| Network | PlainConvUNet, 8 resolution stages | encoder-decoder with skip connections |
| Features per stage | 32, 64, 128, 256, 512, 512, 512, 512 | |
| Convolutions | 3x3, two per stage in encoder and decoder | |
| Downsampling | stride-2 convolutions (7 times) | smallest feature map 5 x 6 (640/128 x 768/128) |
| Normalisation layer | InstanceNorm2d | |
| Activation | LeakyReLU | |
| Dropout | none | |
| Intensity normalisation | z-score per image | chosen because the channel is not CT |
| Batch Dice | true | Dice computed over the whole batch |
nnU-Net printed that this is its "old default planner" and pointed to its newer residual-encoder presets (`nnUNetPlannerResEncM/L/XL`, documentation/resenc_presets.md).

## Training settings (code: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`)
- Epochs: 1000, each 250 training iterations and 50 validation iterations. Shorter trainer classes exist (`nnUNetTrainer_100epochs`, `_250epochs`, `_500epochs`, ...).
- Optimizer: SGD, initial learning rate 0.01, momentum 0.99 with Nesterov, weight decay 3e-5.
- Learning-rate schedule: polynomial decay, lr = 0.01 x (1 - epoch / epochs) ^ 0.9 (`training/lr_scheduler/polylr.py`).
- Loss: soft Dice (background excluded, smoothing 1e-5, batch Dice) plus cross-entropy, weighted 1:1.
- Deep supervision: the decoder also predicts at lower resolutions; each lower resolution gets half the weight of the one above, the lowest gets weight 0, weights normalised to sum to 1.
- Patch sampling: 33% of samples are forced to contain foreground (nerve). This matters for us because the nerve is about 0.3% of the image.
- Checkpoints: `checkpoint_best.pth` whenever an exponential moving average of validation Dice improves, `checkpoint_final.pth` at the end, `checkpoint_latest.pth` every 50 epochs. There is no early stopping; training always runs all epochs. Prediction uses the final checkpoint unless told otherwise.

## Data augmentation (code, 2D case)
| Transform | Probability | Range |
|---|---|---|
| Rotation | 0.2 | up to +-180 degrees (because our patch aspect ratio 768/640 = 1.2 is below 1.5; above 1.5 it would be +-15) |
| Scaling | 0.2 | 0.7 to 1.4 |
| Elastic deformation | 0 | off |
| Gaussian noise | 0.1 | variance 0 to 0.1 |
| Gaussian blur | 0.2 | sigma 0.5 to 1 |
| Brightness (multiplicative) | 0.15 | 0.75 to 1.25 |
| Contrast | 0.15 | 0.75 to 1.25 |
| Simulated low resolution | 0.25 | scale 0.5 to 1 |
| Gamma, inverted image | 0.1 | 0.7 to 1.5 |
| Gamma | 0.3 | 0.7 to 1.5 |
| Mirroring | random | both axes (left-right and top-bottom) |

Ultrasound note (our inference, not from nnU-Net): in these scans the skin is always at the top and depth increases downward. Top-bottom flips and large rotations create images the scanner never produces. A trainer without them is one candidate "improved" experiment (see PLAN.md). nnU-Net ships `nnUNetTrainerNoMirroring` (turns off all flips, including the plausible left-right one); limiting rotation needs a small custom trainer.

## Inference defaults (`nnUNetv2_predict -h`)
- Sliding window with step size 0.5 of the patch. Our patch covers the whole image, so there is effectively one window.
- Test-time augmentation by mirroring is on by default (`--disable_tta` turns it off). With top-bottom mirroring this also averages over flipped views.

## Our additions (not nnU-Net defaults)
- ROI crop and grayscale (removes patient header).
- Custom `splits_final.json`: 5 folds that keep near-duplicate clusters together (nnU-Net's default is 5 random folds). We train fold 0 unless decided otherwise.
- Two datasets: 501 (instructor's 80/20 split) and 502 (split by cluster).

## Checked in the installed code on Nana's Mac (nnunetv2 2.8.1, 4 Nov)
Read from the package source in the `nerveseg` env (versions: `env_mac.txt`). "Inferred" marks our reading where the code does not say it directly.
- Training lengths shipped: `nnUNetTrainer_1epoch`, `_5epochs`, `_10epochs`, `_20epochs`, `_50epochs`, `_100epochs`, `_250epochs`, `_500epochs`, `_750epochs`, `_2000epochs`, `_4000epochs`, `_8000epochs` (`variants/training_length/nnUNetTrainer_Xepochs.py`); the plain `nnUNetTrainer` is 1000.
- Custom trainers: nnU-Net also searches the folders in the `nnUNet_extTrainer` environment variable (`utilities/find_objects.py`). run.py points it at `nnunet/trainers/`, so nothing is copied into the installed package.
- End-of-training validation (`perform_actual_validation`): sliding window with step 0.5, Gaussian weighting, flip averaging on (`use_mirroring=True`) over the trainer's `inference_allowed_mirroring_axes`. Output: `fold_<f>/validation/`. `--val_best` writes to that same folder (there is only one folder name in the code), so it would overwrite the final-checkpoint result.
- Flip axes: `configure_rotation_dummyDA_mirroring_and_inital_patch_size` returns `(rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes)` and sets `inference_allowed_mirroring_axes = mirror_axes`. The axes flipped in training are the axes averaged at test time.
- Axis order: the PNG reader (`NaturalImage2DIO`) reads an image as rows x columns and stores it as (1, 1, rows, columns). In the 2D configuration, spatial axis 0 is therefore top-bottom and axis 1 is left-right (inferred from the shapes).
- Clean-up decision (`postprocessing/remove_connected_components.py`, `determine_postprocessing`): it tries "remove all but the largest foreground region" on the given validation predictions. It keeps it only if mean foreground Dice goes up (strictly) and no class gets worse. Region labelling: `acvl_utils.remove_all_but_largest_component` calls `skimage.measure.label` with full connectivity. Our masks are (1, H, W) arrays, so regions are 8-connected in the image plane (inferred from the shape). `postprocess.py` and `evaluate.py` use 8-connectivity to match.
- `nnUNetv2_find_best_configuration` has `-f` (default 0 1 2 3 4) and `--disable_ensembling`. With `-f 0` it collects only fold 0's validation predictions into `<trainer>__<plans>__2d/crossval_results_folds_0/` and runs the clean-up decision there. It writes `postprocessing.json`/`.pkl` in that folder and `inference_information.json` at the dataset level. So nnU-Net's own step does work with a single fold.
- Data-loading workers: `nnUNet_n_proc_DA` if set; otherwise 12 for unknown machines, capped at the CPU count (18 here), so 12 on Nana's Mac.
- Checkpoints store `current_epoch` (epochs completed), `_best_ema`, `logging` (loss and pseudo-Dice curves) and `inference_allowed_mirroring_axes`. `checkpoint_best.pth` is saved whenever the EMA pseudo Dice on the validation cases improves.
- Preprocessed arrays, checked on 2 cases: the image is cropped to its nonzero bounding box and z-scored. It equals the z-score of the raw crop within 4e-7. The label equals the raw label, with -1 marking pixels outside the nonzero area.

## ResEnc M plans for our data (`nnUNetResEncUNetMPlans.json`, made 4 Nov with `nnUNetPlannerResEncM`; same for 501 and 502)
| Setting | Value | Compared with the default plans |
|---|---|---|
| Network | ResidualEncoderUNet, 7 stages | default: PlainConvUNet, 8 stages |
| Features per stage | 32, 64, 128, 256, 512, 512, 512 | |
| Encoder | residual blocks per stage 1, 3, 4, 6, 6, 6, 6 | default: 2 plain convolutions per stage |
| Decoder | 1 convolution per stage | default: 2 |
| Patch size | 448 x 576 | default 640 x 768. The patch is now **smaller** than the 544 x 736 image, so the network sees part of the image at a time, and prediction uses several sliding windows. |
| Batch size | 13 | default 7 |
The two plans share the preprocessed data folder (`data_identifier` `nnUNetPlans_2d`). Preprocessing for ResEnc M therefore rewrote those files with the same steps; the default `nnUNetPlans.json` is unchanged (checksum).

## To confirm from the training log
- That it prints it is using the existing split file.
- Seconds per epoch on each machine.
- The number of deep supervision outputs.
