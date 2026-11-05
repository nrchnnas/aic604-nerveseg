# Plan: nerve segmentation with nnU-Net

Team: Ping (Pingpan) and Nana. Model: nnU-Net.
Due: report by 11:59 PM Wed 11 Nov 2026; 15-minute in-person presentation on 11 Nov, both speak.
Grading: report 40%, presentation 40%, activity 20%. Report form: dataset 5, model 60, experimental design 25, results 10.
Instructor's email: select one model and explain its architecture, key components, and "the main contributions that enable it to perform effectively on the given task".

Status of this plan: owners and speakers were cleared on 4 Nov so the split can be decided again together (Ping's earlier suggestion: "one person does parts 1+3 and the other does 2"). Items marked OPEN still need both of you.

## Where things stand (5 Nov, 11:15)
- Mac (M5 Max, Apple GPU): environment set up (`docs/env_mac.txt`). E0 measured **301 s per epoch**, which is about 21 h for 250 epochs, 42 h for 500 and 84 h for 1000, per run (lower bound; `docs/timing/`). nnU-Net uses mixed precision and `torch.compile` only on NVIDIA GPUs (`torch.compile` not on Windows), so the Mac trains in plain 32-bit.
- **E1, E2 and E3 all train on the Mac** so they are comparable with each other. The NVIDIA workstation trains B and C with its own baseline W0 (see "Workstation: B and C").
- Mac timing done (`docs/timing/SUMMARY.md`): two runs at once give 538 s per epoch each, about 12% more total work than one run. E3-A trains the same network as the default (269 s per epoch measured). E3-B (ResEnc M) takes 483 s per epoch, 1.6 times the default. 5-fold CV at 250 epochs is about 105 h per dataset on the Mac.
- **E1 and E2 started on the Mac at 00:09 on Thu 5 Nov**, both at once, with `nnUNetTrainer_250epochs` (Nana's decision). Measured: about 500 s per epoch each for epochs 1 to 40. From about 05:45 on Thu 5 the time rose slowly to about 600 s (72 of 250 epochs done at 11:05). Cause unknown: macOS recorded no thermal warning and memory is not short. At 600 s they finish around 17:00 on Fri 6 Nov; if they return to 500 s, around 12:00. They are the backup if training moves to the NVIDIA workstation. 250 epochs is a quarter of nnU-Net's default 1000; the report must say so.
- **E3 decided 5 Nov (Nana): option A, ultrasound-aware augmentation**, `nnUNetTrainerUSAug_250epochs`, on 501 and 502 at the same time. It starts by itself when E1 and E2 have both finished and written `checkpoint_final.pth`. At 500 to 600 s per epoch each it should finish between about 23:00 on Sat 7 and 11:00 on Sun 8 Nov (estimate). Reasons: see "Experiments".
- **B and C go to the NVIDIA workstation** (Nana, 5 Nov), with their own default run W0 there: see "Workstation: B and C". The handoff is in README, "On the NVIDIA workstation".
- The analysis code (`select`, `test-variants`, `compare`, `curves`, clean-up, failure types and nerve size) has been run end to end on the 5-epoch models. That was only a code test; its Dice is not a result.

## Who does what
| Work | Owner | Notes |
|---|---|---|
| Data pipeline, splits, training, evaluation, figures |  | all training on one machine (Mac or the NVIDIA workstation). Code in this folder. |
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
| E1 | nnU-Net 2D, fold 0 | 501 (required 80/20) | headline Dice on the 60 test images | training on the Mac since 00:09 Thu 5 Nov (250 epochs); finishes about 12:00 to 17:00 Fri 6 |
| E2 | nnU-Net 2D, fold 0 | 502 (split by cluster) | Dice without near-duplicate leakage | training on the Mac since 00:09 Thu 5 Nov (250 epochs); finishes about 12:00 to 17:00 Fri 6 |
| E3 | option A, ultrasound-aware augmentation (`nnUNetTrainerUSAug_250epochs`), fold 0 | 501 and 502 | "baseline vs improved" in report 4 | decided 5 Nov; starts by itself after E1 and E2, finishes about Sat 7 23:00 to Sun 8 11:00 |

Training length: decided 4 Nov, `_250epochs` for every run (E1, E2, E3), so they stay comparable. 1000 epochs (nnU-Net's default) would take about 84 h per run on the Mac. Timing runs (4 Nov): E0 alone, then 501 and 502 at the same time (does running two at once give more total work?), then 5 epochs each of E3 candidates A and B. The hours from 5-epoch runs are a lower bound (heat throttling over a long run is not visible in 5 epochs).

Options for E3. Chosen on the argument and the hours per run, **not** on Dice from the 5-epoch runs: five epochs only show that a variant runs and how fast.
- A. Ultrasound-aware augmentation (`nnunet/trainers/nnUNetTrainerUSAug.py`, ready): no top-bottom flips and rotation limited to +-15 degrees, because the skin is always at the top of these scans. Directly about "the given task". Also limits test-time flip averaging to left-right.
- B. nnU-Net's residual-encoder preset ResEnc M (plans made 4 Nov, `-p nnUNetResEncUNetMPlans`): nnU-Net's own recommended upgrade, not specific to our task. On our data its patch is 448 x 576, smaller than the 544 x 736 image, so the network no longer sees the whole image at once (see `docs/nnunet_facts.md`).
- C. Ensemble of 5 folds instead of fold 0: five times the training of E1.

**Decided 5 Nov (Nana): A, on 501 and 502.** Reasons:
- A is about this task: the skin is always at the top of these scans. That supports report section 2 ("on the given task") and section 3 (augmentation as an overfitting control).
- A costs the same as the default and finishes on Saturday.
- Section 2 still explains only one network.

B and C (Nana, 5 Nov): run them on the NVIDIA workstation, see "Workstation: B and C" below.
- On the Mac, the 4 missing folds of C take about 84 h per dataset, so C cannot finish there before 11 Nov.
- Fallback if the workstation does not work out: B on 501 on the Mac, queued after A (about 34 h, so done about Mon 9 09:00 to 21:00, after the Mon 9 08:00 results deadline). Report it only if it finishes and is analysed in time. Decide before E3 finishes.
- Whatever is not run, the report lists as a limitation with the reason (one run takes 21 to 34 h on the Mac). Training only fold 0, instead of nnU-Net's standard 5-fold ensemble, must be stated anyway for E1 to E3.

A's caveat for the report: A also limits test-time flip averaging to left-right, so the training augmentation and the inference flips change together. `test-variants` reports both models with flip averaging off, which separates the two effects.

## Workstation: B and C (Nana, 5 Nov; steps and commands in README, "On the NVIDIA workstation")
E1, E2 and E3 stay on the Mac; E1 stays the headline. The workstation trains, all with 250 epochs on 501:

| ID | What | Purpose | Status |
|---|---|---|---|
| W0 | default nnU-Net (`nnUNetTrainer_250epochs`), fold 0 | baseline for B and C on the same machine; fold 0 of C. W0 vs E1 (same setup, two machines) shows how much Dice moves between two runs, which helps judge whether a difference of a few points means anything. | after timing |
| WB | option B, ResEnc M (`-p nnUNetResEncUNetMPlans`), fold 0 | "baseline vs improved": nnU-Net's recommended bigger network | after W0 |
| WC | option C, default folds 1 to 4, then the average of the 5 folds' models on test (`run.py ensemble`) | "baseline vs improved": nnU-Net's standard 5-fold ensemble; also the 5-fold cross-validation Dice on all 240 training images | after WB, only if all 4 folds fit |

Rules, fixed before any results:
- Compare only within one machine: B and C against W0. A workstation number is never put next to a Mac number as if only the model changed (precision and `torch.compile` differ).
- B vs W0: the same rule as E3 (`compare`): on the 39 fold-0 validation images, adopt only if mean Dice rises; both are reported on test. `select` and `test-variants` run for W0 and for B, as for every trained run.
- C has no untouched validation images (each fold's validation images trained the other four models), so the validation rule cannot choose it. It is reported on test next to W0, with the paired difference and its CI. The 5-fold cross-validation Dice (each training image predicted by the one model that did not train on it) is reported in section 3 as a steadier estimate than 39 images.
- Runs may train at the same time on the workstation. That changes how fast they go, not what is compared (README step 5 has the limits).
- Order: timing first, then W0, B, C. Results are needed by Mon 9 Nov 08:00 (section 4). After the timing, plan only what fits by then, with slack. 502 only if everything else is done early.
- Results come back through git: `run.py records --machine workstation` (`docs/results/workstation/`).

## Analyses on every trained run (agreed by Nana 4 Nov; run for E1 and E2, and E3)
Rules, fixed before any results:
- Choices are made on the fold-0 validation images (501: 39, 502: 36), never on test. Every variant is still reported on test.
- A variant is adopted only if it raises mean validation Dice. We report the paired 95% CI with it. With 39 images, a gain of 1 to 2 Dice points is probably noise, and the report says so.
- One decision at a time: first test-time flip averaging (on/off), then the clean-up on top of that choice. Not all combinations.
- The checkpoint is not selected. The headline uses the final checkpoint (nnU-Net's default). nnU-Net picked `checkpoint_best` with a validation Dice measured on these same 39 images, so best vs final is reported on test only as evidence about overfitting.
- The default validation result is nnU-Net's own `fold_0/validation/` (final checkpoint, flip averaging on), scored with our `evaluate.py`, not nnU-Net's `summary.json`. All comparisons therefore use one Dice definition. Do not use nnU-Net's `--val_best` (it overwrites that folder).
- E3 vs the default trainer: same rule on the validation images, at default inference; both are reported on test (`run.py compare`).

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
| Thu 5 | E1 and E2 training (started 00:09); workstation: set up, time, start W0 | send section 2 outline; agree it together |
| Fri 6 | E1 and E2 finish about 12:00 to 17:00, E3 starts by itself; run E1/E2 analyses | write section 2 |
| Sat 7 | E1/E2 results and figures; E3 training (finishes Sat 7 23:00 to Sun 8 11:00) | section 2 draft to Nana |
| Sun 8 | E3 analyses (`compare`); draft sections 1 and 3 | revise section 2; slide template |
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
- Workstation (W0, B, C): what fits before Mon 9 08:00 is decided after its timing on Thu 5.
- Fallback B on the Mac after E3 (done about Mon 9 09:00 to 21:00, after the results deadline): only if the workstation does not work out; decide before E3 finishes.
- Decided 5 Nov: E3 is option A on 501 and 502 (Nana); B and C on the NVIDIA workstation, compared with W0 there. Training length 250 epochs for all runs (4 Nov). 5-fold does not fit on the Mac (about 84 h per dataset for the 4 missing folds).
- Who merges the report; slide design owner.
- Decided 4 Nov: headline uses `checkpoint_final`; `checkpoint_best` reported only as overfitting evidence.
