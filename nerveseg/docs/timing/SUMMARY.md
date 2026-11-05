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

## Workstation (RTX 4090 24 GB, i9-14900K, 128 GB, Windows 11, torch 2.15.0+cu132, nnunetv2 2.8.1, device cuda; 5 Nov 2026)
Mixed precision is on (cuda). `torch.compile` is off: nnU-Net disables it on Windows (`_do_i_compile`), and the logs have no "Using torch.compile" line. So on this machine the difference from the Mac is mixed precision only, not `torch.compile`. Same plans as the Mac: batch 7 and 640 x 768 for the default, batch 13 and 448 x 576 for ResEnc M. About 3.7 GB of the card was already in use by the desktop and other apps.

| Run | Seconds per epoch (measured) | Peak GPU memory, whole card |
|---|---|---|
| Default, alone | 34.7 (55.2, 34.7, 34.7, 34.8, 34.7) | 11.1 GB |
| ResEnc M, alone | 58.4 (74.5, 58.4, 58.4, 58.4, 58.5) | 14.0 GB |
| Default, sharing the GPU with ResEnc M | about 109 (epochs 3 to 5: 111.6, 107.3, 110.4) | 21.4 GB for both |
| ResEnc M, sharing the GPU with the default | about 166 (epoch 2: 165.8; epoch 1 was 197.9, with warm-up) | same |

Read the "together" rows with care. The second run started a minute later and the first ended earlier, so the two only partly overlapped. The `mean_s_per_epoch_excluding_first` values in those two `summary.json` files (94.0 and 86.4) mix shared epochs with epochs that ran alone. The numbers above use only the epochs that ran fully shared; for ResEnc M that is one epoch, so it is a rough figure.

Running two at once does not pay off here. README's rule needs each run below 2 times its time alone (69 s and 117 s); they took 109 s and 166 s. Memory was not the limit (21.4 of 24.6 GB). So W0 and B train one after the other.

## Hours per training run on the workstation (calculated from the numbers above)
Lower bounds: five epochs do not show slowdown over a long run.

| Epochs | Default (34.7 s) | ResEnc M (58.4 s) |
|---|---|---|
| 250 | 2.4 h | 4.1 h |
| 500 | 4.8 h | 8.1 h |
| 750 | 7.2 h | 12.2 h |
| 1000 | 9.6 h | 16.2 h |

At 250 epochs: W0 and B one after the other about 6.5 h; W0 and B at once about 9 h. The 4 folds of C, one after the other, about 9.6 h. W0, B and C in total about 16 h.

## Other machines
Time each new machine with `python nerveseg/nnunet/run.py timing -d 501` (and `-t nnUNetTrainerUSAug`, `-p nnUNetResEncUNetMPlans`), then add a section here.
