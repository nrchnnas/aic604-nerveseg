# Timing runs

Each run is 5 epochs on Dataset501 fold 0 (250 training iterations per epoch). "Seconds per epoch" is the mean of epochs 2 to 5; epoch 1 includes warm-up. The logs and `summary.json` of each run are in the folders next to this file.

## Mac (Apple M5 Max, 128 GB, macOS 26.4, torch 2.15.0, nnunetv2 2.8.1, device mps; 4 Nov 2026)
All runs on the charger with Low Power Mode off. nnU-Net trains in plain 32-bit on Apple GPUs: no mixed precision and no `torch.compile` (both are cuda-only in nnU-Net's code).

| Run | Network, batch, patch | Seconds per epoch (measured) |
|---|---|---|
| E0: nnUNetTrainer, alone | PlainConvUNet 8 stages, batch 7, 640 x 768 | 301 (272, 316, 287, 313, 290) |
| 501 and 502 at the same time | same | 538 and 537 per run, so about 269 s per epoch of total work (12% more than one run alone) |
| E3-A: nnUNetTrainerUSAug, alone | same network; only augmentation differs | 269 (272, 287, 269, 263, 257) |
| E3-B: ResEnc M plans, alone | ResidualEncoderUNet 7 stages, batch 13, 448 x 576 | 483 (499, 488, 485, 481, 478) |

During E0 the GPU was 69 to 99% busy and the 12 data-loading workers were mostly idle, so the GPU is the limit. E0 and E3-A train the same network. Their 11% difference is probably due to other apps running during E0 (inferred, not measured), so plan with the higher number.

## Hours per training run on the Mac (calculated from the numbers above)
These are lower bounds. Five epochs do not show the laptop slowing down as it heats up over a long run, so leave slack.

| Epochs | Default or E3-A (301 s) | E3-B ResEnc M (483 s) | Two default runs at once (538 s each) |
|---|---|---|---|
| 250 | 21 h | 34 h | 37 h for both |
| 500 | 42 h | 67 h | 75 h for both |
| 750 | 63 h | 101 h | 112 h for both |
| 1000 | 84 h | 134 h | 149 h for both |

5-fold cross-validation multiplies one run by 5 per dataset. At 250 epochs that is about 105 h for one dataset.

## Other machines
Time each new machine with `python nerveseg/nnunet/run.py timing -d 501` (and `-t nnUNetTrainerUSAug`, `-p nnUNetResEncUNetMPlans`), then add a section here.
