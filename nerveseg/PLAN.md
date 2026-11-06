# Plan: nerve segmentation with nnU-Net

Team: Ping (Pingpan) and Nana. Model: nnU-Net.
Due: report by 11:59 PM Wed 11 Nov 2026; 15-minute in-person presentation on 11 Nov, both speak.
Grading: report 40%, presentation 40%, activity 20%. Report form: dataset 5, model 60, experimental design 25, results 10.
Instructor's email: select one model and explain its architecture, key components, and "the main contributions that enable it to perform effectively on the given task".

Status of this plan: owners and speakers were cleared on 4 Nov so the split can be decided again together (Ping's earlier suggestion: "one person does parts 1+3 and the other does 2"). Items marked OPEN still need both of you.

## Where things stand (5 Nov, 16:00)
- Mac (M5 Max, Apple GPU): environment set up (`docs/env_mac.txt`). E0 measured **301 s per epoch**, which is about 21 h for 250 epochs, 42 h for 500 and 84 h for 1000, per run (lower bound; `docs/timing/`). nnU-Net uses mixed precision and `torch.compile` only on NVIDIA GPUs (`torch.compile` not on Windows), so the Mac trains in plain 32-bit.
- **All reported results come from the NVIDIA workstation** (Nana, 5 Nov): W0, W2, WA, WA2, WB and WC, so every comparison is within one machine. W0 (default nnU-Net, 0.822 test Dice) is the main result. Nothing trained on the Mac is reported.
- Mac timing done (`docs/timing/SUMMARY.md`): two runs at once give 538 s per epoch each, about 12% more total work than one run. E3-A trains the same network as the default (269 s per epoch measured). E3-B (ResEnc M) takes 483 s per epoch, 1.6 times the default. 5-fold CV at 250 epochs is about 105 h per dataset on the Mac.
- Mac runs stopped (Nana, 5 Nov): E1 and E2 (`nnUNetTrainer_250epochs`, 501 and 502 at once) started at 00:09 Thu 5 Nov and were stopped at 15:56 at epoch 94 of 250. They ran at about 500 s per epoch, slowing to about 750 s; the workstation trains the same run in about 2.4 h. E3 never started on the Mac. The partial checkpoints stay in the Mac's `nnUNet_results` (not in git) and are not used.
- **E3 decided 5 Nov (Nana): option A, ultrasound-aware augmentation**, `nnUNetTrainerUSAug_250epochs`, on 501 and 502, trained on the workstation as WA (501) and WA2 (502). Reasons: see "Experiments".
- **B and C also train on the workstation** (WB, WC), with W0 as their baseline: see "Workstation runs". Steps are in README, "On the NVIDIA workstation".
- The analysis code (`select`, `test-variants`, `compare`, `curves`, clean-up, failure types and nerve size) has been run end to end on the 5-epoch models. That was only a code test; its Dice is not a result.

## Who does what
| Work | Owner | Notes |
|---|---|---|
| Data pipeline, splits, training, evaluation, figures |  | all training on one machine: the NVIDIA workstation. Code in this folder. |
| Report 1: dataset description and preprocessing |  | facts in `data/dataset_report.json`, README "Dataset facts" |
| Report 2: nnU-Net architecture and contributions |  | the 60-point section; outline agreed together first |
| Report 2, part: what nnU-Net configured for our data and why it suits this task |  | from `docs/nnunet_facts.md`; this is the "on the given task" part of the email |
| Report 3: experimental design |  | training settings, metrics, splits, overfitting controls |
| Report 4: results and analysis |  | tables, success and failure cases, leakage comparison |
| Slides design |  | |
| Merging the report into one document |  | |

Note: 60% of the report grade sits in section 2, so its outline is agreed together early and the other person reviews the draft on Sat 7.

## Proposed outline for report section 2 (OPEN until agreed)
1. Task and why segmentation (one short paragraph).
2. U-Net: encoder, decoder, skip connections; why this shape suits segmentation.
3. nnU-Net's main contribution: self-configuration. Fixed parameters, rule-based parameters from the dataset fingerprint, empirical parameters. The paper's claim that configuration matters more than architecture.
4. What nnU-Net configured for our data (table from `docs/nnunet_facts.md`).
5. Why it works on this task: whole image in one patch (the network sees the artery beside the nerve), foreground oversampling for a target about 0.3% of the image, Dice + cross-entropy loss and batch Dice for a small target, deep supervision, intensity augmentation for ultrasound brightness differences. Mark which of these are the paper's claims and which are our reasoning.
6. Pretrained weights: none; nnU-Net trains from scratch (one sentence on why).

## Experiments
| ID | What | Dataset | Purpose | Status |
|---|---|---|---|---|
| E0 | 5-epoch timing run | 501 | seconds per epoch, check the split file is used | done on the Mac 4 Nov: 301 s per epoch, split file used (201/39). Repeat on the training machine if it changes. |
| E1 | nnU-Net 2D, fold 0 | 501 (required 80/20) | main result: Dice on the 60 test images | trained on the workstation as W0 (done 14:28 Thu 5). The Mac run was stopped at epoch 94 and is not used. |
| E2 | nnU-Net 2D, fold 0 | 502 (split by cluster) | Dice without near-duplicate leakage | trained on the workstation as W2. The Mac run was stopped at epoch 94 and is not used. |
| E3 | option A, ultrasound-aware augmentation (`nnUNetTrainerUSAug_250epochs`), fold 0 | 501 and 502 | "baseline vs improved" in report 4 | decided 5 Nov; trained on the workstation as WA (501) and WA2 (502) |

Training length: decided 4 Nov, `_250epochs` for every run, so they stay comparable. 250 epochs is a quarter of nnU-Net's default 1000; the report must say so. (Decided when training was planned on the Mac, where 1000 epochs would take about 84 h per run.) Timing runs (4 Nov): E0 alone, then 501 and 502 at the same time (does running two at once give more total work?), then 5 epochs each of E3 candidates A and B. The hours from 5-epoch runs are a lower bound (heat throttling over a long run is not visible in 5 epochs).

Options for E3. Chosen on the argument and the hours per run, **not** on Dice from the 5-epoch runs: five epochs only show that a variant runs and how fast.
- A. Ultrasound-aware augmentation (`nnunet/trainers/nnUNetTrainerUSAug.py`, ready): no top-bottom flips and rotation limited to +-15 degrees, because the skin is always at the top of these scans. Directly about "the given task". Also limits test-time flip averaging to left-right.
- B. nnU-Net's residual-encoder preset ResEnc M (plans made 4 Nov, `-p nnUNetResEncUNetMPlans`): nnU-Net's own recommended upgrade, not specific to our task. On our data its patch is 448 x 576, smaller than the 544 x 736 image, so the network no longer sees the whole image at once (see `docs/nnunet_facts.md`).
- C. Ensemble of 5 folds instead of fold 0: five times the training of E1.

**Decided 5 Nov (Nana): A, on 501 and 502.** Reasons:
- A is about this task: the skin is always at the top of these scans. That supports report section 2 ("on the given task") and section 3 (augmentation as an overfitting control).
- A costs the same as the default.
- Section 2 still explains only one network.

B and C (Nana, 5 Nov): run them on the NVIDIA workstation, see "Workstation runs" below.
- On the Mac, the 4 missing folds of C take about 84 h per dataset, so C cannot finish there before 11 Nov.
- Whatever is not run, the report lists as a limitation with the reason. Training only fold 0, instead of nnU-Net's standard 5-fold ensemble, must be stated for every run except WC (WC is the 5-fold version of W0).

A's caveat for the report: A also limits test-time flip averaging to left-right, so the training augmentation and the inference flips change together. `test-variants` reports both models with flip averaging off, which separates the two effects.

## Workstation runs: all reported results (Nana, 5 Nov; steps and commands in README, "On the NVIDIA workstation")
Every result in the report comes from this machine; the Mac runs were stopped (Nana, 5 Nov). All runs use 250 epochs, on 501 unless stated. W0 is the main result. WA, WB and WC are compared with W0; W2 against W0 is the leakage comparison.

| ID | What | Purpose | Status |
|---|---|---|---|
| W0 | default nnU-Net (`nnUNetTrainer_250epochs`), fold 0 | main result (E1): nnU-Net's Dice on the 60 test images; baseline for A, B and C; fold 0 of C | trained 12:00 to 14:28 Thu 5 Nov (34.7 s per epoch, `docs/timing/SUMMARY.md`); analyses done, in `docs/results/workstation/` |
| WB | option B, ResEnc M (`-p nnUNetResEncUNetMPlans`), fold 0 | "baseline vs improved": nnU-Net's recommended bigger network | trained 14:28 to 18:41 Thu 5 Nov (median 58.5 s per epoch); analyses done, in `docs/results/workstation/`. `compare` keeps W0: validation Dice 0.7226 vs 0.7244 (difference -0.0018, CI -0.0706 to +0.0784); test 0.8298 vs 0.8224. Not at the same time as W0: sharing the GPU was slower (about 9 h vs 6.5 h) |
| WA | option A (`nnUNetTrainerUSAug_250epochs`), fold 0 | option A on 501 (E3), compared with W0 | trained 18:48 to 21:16 Thu 5 Nov (median 34.6 s per epoch); analyses done, in `docs/results/workstation/`. `compare` keeps W0: validation Dice 0.6677 vs 0.7244 (difference -0.0567, CI -0.1496 to +0.0387); test 0.8126 vs 0.8224. Flip averaging lowers A's Dice, so validation turned it off for A: validation 0.7065, test 0.8218 (`docs/results/workstation/flip_views_check.txt` shows the trainer flips the right axis) |
| W2 | default (`nnUNetTrainer_250epochs`), fold 0, on 502 | leakage comparison with W0 (E2) | trained 21:16 to 23:46 Thu 5 Nov (median 34.9 s per epoch); analyses done, in `docs/results/workstation/`. Validation Dice 0.4566 (CI 0.3288 to 0.5775), test 0.8342 (CI 0.8094 to 0.8564); flip averaging on, clean-up off. Do not read W2 against W0 on test as the leakage effect: see "Found after the results" below |
| WA2 | option A, fold 0, on 502 | option A on 502 (E3), compared with W2 | trained 23:46 Thu 5 Nov to 02:14 Fri 6 Nov (median 34.8 s per epoch); analyses done, in `docs/results/workstation/`. `compare` keeps W2: validation Dice 0.4107 vs 0.4566 (difference -0.0459, CI -0.1302 to +0.0366); test 0.7657 vs 0.8342 (difference -0.0685, CI -0.1096 to -0.0340). Flip averaging on, clean-up off |
| WC | option C, default folds 1 to 4, then the average of the 5 folds' models on test (`run.py ensemble`) | "baseline vs improved": nnU-Net's standard 5-fold ensemble; also the 5-fold cross-validation Dice on all 240 training images | fold 0 is W0; folds 1 to 4 trained 02:14 to 12:09 Fri 6 Nov (about 34.8 s per epoch); analyses done, in `docs/results/workstation/`. Test: ensemble 0.8539 (CI 0.8291 to 0.8774), W0 0.8224, difference +0.0315 (CI by image +0.0075 to +0.0643, by cluster +0.007 to +0.072); most of the gain is not from averaging: see "Found after option C" below. Cross-validation over 240 images: 0.6379 (CI by image 0.6001 to 0.6752, by cluster 0.491 to 0.750); folds 0.7244, 0.5876, 0.7232, 0.7585, 0.4156. Fold 3 crashed at 09:39 as validation started, after saving its final checkpoint (python.exe, ntdll.dll, illegal instruction 0xc000001d; suspected CPU instability, not proven). The queue's retry ran validation only, so fold 3 has two logs. Saved masks predicted again on this machine match: at most 3 pixels differ per image, 160 images, `docs/results/workstation/repredict_check.txt` |

Rules, fixed before any results:
- Every reported number comes from the workstation, so every comparison is within one machine: A, B and C against W0; on 502, A against W2.
- A and B vs W0 (and WA2 vs W2): the `compare` rule: on the fold-0 validation images, adopt only if mean Dice rises; both are reported on test. `select` and `test-variants` run for every run.
- C has no untouched validation images (each fold's validation images trained the other four models), so the validation rule cannot choose it. It is reported on test next to W0, with the paired difference and its CI. The 5-fold cross-validation Dice (each training image predicted by the one model that did not train on it) is reported in section 3 as a steadier estimate than 39 images.
- Runs may train at the same time on the workstation. That changes how fast they go, not what is compared (README step 5 has the limits).
- Order: timing first, then W0, B; check that W0 and B are usable in the report; then WA, W2, WA2, C, one at a time. Results are needed by Mon 9 Nov 08:00 (section 4). The whole queue is planned to finish about Fri 6 Nov 12:00.
- Results come back through git: `run.py records --machine workstation` (`docs/results/workstation/`).

Found after the results (5 Nov). The rules above and the decisions made with them stand; the report should state these. Numbers and script: `docs/results/workstation/cluster_check.txt`.
- Validation images are not independent, because near-duplicates sit in clusters. 501 fold 0: 39 images from 11 clusters, 16 of them from cluster 38. 502 fold 0: 36 images from 3 clusters, 22 of them from cluster 12 (mean Dice 0.220, 12 images at 0). Validation decisions lean on these clusters, and CIs that resample clusters are wider.
- W2 against W0 on test (0.8342 vs 0.8224) does not measure leakage, because the test sets differ. 502's test set has 63 images from 11 clusters, 40 of them from cluster 20, and larger nerves (median 3277 px vs 1922 px). Counting each cluster once: W2 0.7457, W0 0.7877.
- Same-image comparison, W0 against W2, on W0's test images that W2 did not train on. With a near-duplicate among the 240 training images (n=16, 7 clusters; see the correction under "Found after option C"): W0 0.8455, W2 0.6337, difference +0.2119, median +0.0491, W0 better on 15 of 16, CI by image +0.033 to +0.404, by cluster -0.098 to +0.484. Without one (n=5): W0 0.6942, W2 0.7578, difference -0.0635, CI by image -0.280 to +0.112. Five of the 16 are cluster 12 images, which are W2's validation images. W2 used default inference, so they chose nothing here.
- The same comparison repeats with option A, a second pair of models. With a near-duplicate (n=16): A 0.8887, WA2 0.5546, difference +0.3340, median +0.1045, A better on all 16, CI by cluster +0.094 to +0.598. Without one (n=5): A 0.7283, WA2 0.7275, difference +0.0008, CI by cluster -0.070 to +0.062.
- Option A costs more without near-duplicates. A minus default on test: -0.0098 on 501 and -0.0685 on 502 (CI by cluster -0.268 to -0.018). With flip averaging off for both: +0.0044 on 501 and -0.0891 on 502 (CI by cluster -0.365 to -0.014). One run each, so this suggests rather than shows that near-duplicates hide the cost of less augmentation.
- W2's best checkpoint is from epoch 3 and was never beaten, because its pseudo Dice comes from those 3 clusters. Its test Dice (0.7433, against 0.8342 for the final checkpoint) is therefore not evidence about overfitting in the usual sense.

Found after option C (6 Nov). The groupings below were chosen after seeing the results, to describe them; no image is dropped from any reported score. Numbers and script: `docs/results/workstation/cv_check.txt`.
- The folds train alike and differ in what they hold out. Train loss ends near -0.974 in every fold; validation loss ends between -0.40 and -0.77. Fold means have SD 0.1425. Fold 4's pseudo Dice never beat epoch 1 (0.4885), as with W2. That comes from its held-out clusters, not a failed training.
- Failures are concentrated. Clusters 14, 12 and 2 hold 42 of the 240 images and 26 of the 30 zero-Dice images. The other 198 images score 0.7471.
- Cluster 14 (12 images, all Dice 0) is the only cluster with its narrow view: 443 image columns hold ultrasound, median 722. No model that held it out had trained on a scan like it. Cluster 12 fails in two models with different training images: 0.129 in fold 1, 0.220 in W2. In clusters 12 and 2 the prediction often sits on the other side of the dark oval from the true nerve (seen on overlays, not measured).
- Nerve position is not a general explanation. Images with the nerve at least 2 SD from the training mean score 0.367 (n=25), but one point per cluster gives Spearman rho -0.074 (p = 0.63), and cluster 14 sits near the mean.
- W0 on test by whether it trained on the image's cluster: yes (n=40, 14 clusters) 0.8799, CI by cluster 0.866 to 0.899; no (n=20, 15 clusters) 0.7074, CI by cluster 0.592 to 0.792. Cross-validation, where no cluster is shared, gives 0.6379. These are different images, so they suggest the size of the effect rather than measure it. The report should give them next to the 0.8224 headline.
- Most of the ensemble's gain is not from averaging. Difference from W0 by where the test image's cluster sits. W0 trained on it (n=40): +0.0030, CI by cluster -0.002 to +0.007. Only in fold 0's validation, so four of the five models trained on it and W0 did not (n=11, 6 clusters): +0.1282, CI +0.047 to +0.333, about three quarters of the total gain. No model trained on it (n=9, 9 clusters): +0.0398, CI -0.040 to +0.128. Median gain over all 60 is +0.0041. The ensemble does remove the worst failures: minimum 0.5594 vs 0.0153, images with more than one predicted region 1 vs 6.
- Correction to the near-duplicate split. `nearest_train_r` compares a test image with all 240 training images, but W0 and A trained on 201. So 10 of the 43 "near-identical" test images have their twin only in fold 0's validation images (W0 0.7149, against 0.8830 for the other 33). No score changes. The 43 and 17 split stays as the fixed metric; the report should say "among the 240 training images". Same-image comparison split instead by whether fold 0 trained on the cluster: W0 minus W2, trained on (n=16, 6 clusters) +0.2474, median +0.0491, W0 better on all 16, CI by cluster +0.041 to +0.537; not trained on (n=5) -0.1772, CI -0.427 to +0.062. A minus WA2, trained on (n=16) +0.3292, median +0.1002, A better on all 16, CI +0.086 to +0.642; not trained on (n=5) +0.0163, CI -0.065 to +0.088.
- Overfitting. Train loss reaches -0.97 in every fold while validation stays well behind (gap 0.20 to 0.57), and the size of the gap follows the held-out clusters, not the epoch count. Validation scores flatten rather than fall: EMA pseudo Dice ends within 0.03 of its peak in folds 0 to 3. Stopping earlier would not close this. Best minus final checkpoint on test: W0 +0.0172 (CI -0.0059 to +0.0488), B -0.0026 (CI -0.0098 to +0.0042), A +0.0111 (CI +0.0021 to +0.0239), WA2 +0.0119 (CI +0.0025 to +0.0229), W2 -0.0909 (CI -0.1238 to -0.0609; see above). So at most a point or two, with a large loss in one case. The remedy is more independent scans: the 300 images come from 53 near-duplicate clusters, 25 of them single images, and the dataset is fixed. Neither A (-0.0567) nor B (-0.0018) raised validation Dice, and both CIs include 0. The report should say overfitting was monitored and limited, not avoided.

## Analyses on every trained run (agreed by Nana 4 Nov; run for every workstation run)
Rules, fixed before any results:
- Choices are made on the fold-0 validation images (501: 39, 502: 36), never on test. Every variant is still reported on test.
- A variant is adopted only if it raises mean validation Dice. We report the paired 95% CI with it. With 39 images, a gain of 1 to 2 Dice points is probably noise, and the report says so.
- One decision at a time: first test-time flip averaging (on/off), then the clean-up on top of that choice. Not all combinations.
- The checkpoint is not selected. The headline uses the final checkpoint (nnU-Net's default). nnU-Net picked `checkpoint_best` with a validation Dice measured on these same 39 images, so best vs final is reported on test only as evidence about overfitting.
- The default validation result is nnU-Net's own `fold_0/validation/` (final checkpoint, flip averaging on), scored with our `evaluate.py`, not nnU-Net's `summary.json`. All comparisons therefore use one Dice definition. Do not use nnU-Net's `--val_best` (it overwrites that folder).
- Option A or B vs the default trainer: same rule on the validation images, at default inference; both are reported on test (`run.py compare`).

What each run gets:
1. Training curves (`run.py curves`): nnU-Net's loss and pseudo-Dice plot, and the epoch where `checkpoint_best` was saved. For "how overfitting was avoided".
2. Best vs final checkpoint on test (`run.py test-variants`): overfitting evidence only.
3. Flip averaging on/off: decided on validation (`run.py select`), both reported on test.
4. Clean-up, keeping only the largest predicted region (every image has one nerve). Decided on validation with our `postprocess.py`, and both reported on test. nnU-Net's own step (`nnUNetv2_find_best_configuration <id> -c 2d -tr <trainer> -f 0 --disable_ensembling`, pinned to one experiment) is also run and recorded. It always decides on its default validation result with flip averaging on. If our rule turns flip averaging off, the two decisions used different bases; the report says so, and our decision counts.
5. Failure types and nerve size (`evaluate.py`, `figures.py size`):
   - complete misses (empty, or predicted elsewhere);
   - poor overlap (Dice below 0.5);
   - found;
   - Dice by nerve size (terciles over all 300 images: below 1308 px, 1308 to 2289 px, above that), with Spearman rho.

Metrics: Dice per test image; mean with bootstrap 95% CI, median, minimum; IoU. On 501, Dice is also reported separately for the 43 test images that have a near-identical training image and the 17 that do not.

## Timeline
| Day | Nana | Ping |
|---|---|---|
| Wed 4 | set up environment, run E0 | read the nnU-Net paper (resources 1, 4) |
| Thu 5 | workstation: set up, timing, W0 done and analysed, WB training; Mac runs stopped | send section 2 outline; agree it together |
| Fri 6 | workstation queue (WA, W2, WA2, WC) finished 12:09; analyses done | write section 2 |
| Sat 7 | results tables and figures for all runs | section 2 draft to Nana |
| Sun 8 | draft sections 1 and 3 | revise section 2; slide template |
| Mon 9 | section 4; merge report v1 | slides for intro and model |
| Tue 10 | both: full slide deck, rehearse twice with a timer, fix gaps | both: full slide deck, rehearse twice with a timer, fix gaps |
| Wed 11 | present; submit report by 11:59 PM | present; submit report by 11:59 PM |

## Presentation (15 minutes)
| Slide | Content | Speaker | Time |
|---|---|---|---|
| 1 | Title, task |  | 0:45 |
| 2 | Why nerve segmentation in ultrasound; what the data looks like |  | 1:15 |
| 3 | Dataset issues: tiny target, patient details removed, near-duplicates |  | 1:30 |
| 4 | U-Net basics |  | 1:30 |
| 5 | nnU-Net's idea: fixed, rule-based, empirical parameters |  | 2:00 |
| 6 | What nnU-Net chose for our data and why it fits |  | 1:30 |
| 7 | Training setup (loss, optimizer, augmentation) |  | 1:15 |
| 8 | Experimental design: splits, metrics, overfitting controls |  | 1:15 |
| 9 | Results table |  | 1:15 |
| 10 | Successes and failures |  | 1:15 |
| 11 | Leakage: required vs grouped split; baseline vs improved |  | 1:00 |
| 12 | Conclusion and limitations |  | 0:30 |
Speakers: OPEN. The brief says everyone speaks, so aim for about 7:30 each. Both should be able to answer questions on either half.

## OPEN
- Work split and speakers (owner and speaker columns cleared 4 Nov; decide together).
- Section 2 outline (Thu 5).
- Decided 5 Nov (Nana): every reported result comes from the workstation and W0 is the main result; the Mac runs were stopped. E3 is option A on 501 and 502 (WA, WA2); A, B and C are all compared with W0. Training length 250 epochs for all runs (4 Nov).
- Who merges the report; slide design owner.
- Decided 4 Nov: headline uses `checkpoint_final`; `checkpoint_best` reported only as overfitting evidence.
