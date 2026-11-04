# Team 4 plan: nerve segmentation with nnU-Net

Team: Ping (Pingpan) and Nana. Model: nnU-Net (team 4 on the sheet).
Due: report by 11:59 PM Wed 11 Nov 2026; 15-minute in-person presentation on 11 Nov, both speak.
Grading: report 40%, presentation 40%, activity 20%. Report form: dataset 5, model 60, experimental design 25, results 10.
Instructor's email: select one model and explain its architecture, key components, and "the main contributions that enable it to perform effectively on the given task".

Status of this plan: work split agreed in principle by Ping ("one person does parts 1+3 and the other does 2") and adjusted by Nana on 4 Nov. Items marked OPEN still need both of you.

## Who does what
| Work | Owner | Notes |
|---|---|---|
| Data pipeline, splits, training, evaluation, figures | Nana | runs on Nana's Mac (M5 Max). Code in this folder. |
| Report 1: dataset description and preprocessing | Nana | facts in `data/dataset_report.json`, README "Dataset facts" |
| Report 2: nnU-Net architecture and contributions | Ping | the 60-point section; outline agreed with Nana first |
| Report 2, part: what nnU-Net configured for our data and why it suits this task | Nana (with Ping) | from `docs/nnunet_facts.md`; this is the "on the given task" part of the email |
| Report 3: experimental design | Nana | training settings, metrics, splits, overfitting controls |
| Report 4: results and analysis | Nana | tables, success and failure cases, leakage comparison |
| Slides design | Ping | OPEN: confirm |
| Merging the report into one document | OPEN | whoever has more time on Mon 9 |

Why this split: sections 1, 3 and 4 all come from the pipeline and experiments on Nana's machine, so one person holding them avoids passing facts back and forth. Section 2 does not depend on results, so Ping can start immediately and nobody waits. Risk: 60% of the report grade sits in one section, so the outline is agreed together early and Nana reviews the draft on Sat 7.

## Proposed outline for report section 2 (for Ping; OPEN until agreed)
1. Task and why segmentation (one short paragraph).
2. U-Net: encoder, decoder, skip connections; why this shape suits segmentation.
3. nnU-Net's main contribution: self-configuration. Fixed parameters, rule-based parameters from the dataset fingerprint, empirical parameters. The paper's claim that configuration matters more than architecture.
4. What nnU-Net configured for our data (table from `docs/nnunet_facts.md`).
5. Why it works on this task: whole image in one patch (the network sees the artery beside the nerve), foreground oversampling for a target about 0.3% of the image, Dice + cross-entropy loss and batch Dice for a small target, deep supervision, intensity augmentation for ultrasound brightness differences. Mark which of these are the paper's claims and which are our reasoning.
6. Pretrained weights: none; nnU-Net trains from scratch (one sentence on why).

## Experiments
| ID | What | Dataset | Purpose | Status |
|---|---|---|---|---|
| E0 | 5-epoch timing run | 501 | seconds per epoch, check the split file is used | to do first |
| E1 | nnU-Net 2D, fold 0 | 501 (required 80/20) | headline Dice on the 60 test images | |
| E2 | nnU-Net 2D, fold 0 | 502 (split by cluster) | Dice without near-duplicate leakage | |
| E3 | improved variant, fold 0 | 501 (and 502 if time) | "baseline vs improved" in report 4 | OPEN: Nana wants it decided 4 Nov, after the timing runs |

Training length: chosen after the timing runs (4 Nov) so that E1 and E2 finish by Sat 7. If 1000 epochs do not fit, use one shorter length (`_250epochs`, `_500epochs` or `_750epochs`) for all runs so they stay comparable. Timing runs: E0 alone, then 501 and 502 at the same time (does running two at once give more total work?), then 5 epochs each of E3 candidates A and B. The hours from 5-epoch runs are a lower bound (heat throttling over a long run is not visible in 5 epochs).

Options for E3 (pick one). Choose on the argument and the hours per run, **not** on Dice from the 5-epoch runs: five epochs only show that a variant runs and how fast.
- A. Ultrasound-aware augmentation (`nnunet/trainers/nnUNetTrainerUSAug.py`, ready): no top-bottom flips and rotation limited to +-15 degrees, because the skin is always at the top of these scans. Directly about "the given task". Also limits test-time flip averaging to left-right.
- B. nnU-Net's residual-encoder preset ResEnc M (plans made 4 Nov, `-p nnUNetResEncUNetMPlans`): nnU-Net's own recommended upgrade, not specific to our task. On our data its patch is 448 x 576, smaller than the 544 x 736 image, so the network no longer sees the whole image at once (see `docs/nnunet_facts.md`).
- C. Ensemble of 5 folds instead of fold 0: five times the training of E1. Decide with the timing numbers (5-fold CV is also OPEN, see below).

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
| Thu 5 | start E1 | send section 2 outline; agree it together |
| Fri 6 | E1 finishes (depends on E0 timing), start E2 | write section 2 |
| Sat 7 | E1/E2 results and figures; choose E3 together | section 2 draft to Nana |
| Sun 8 | run E3; draft sections 1 and 3 | revise section 2; slide template |
| Mon 9 | section 4; merge report v1 | slides for intro and model |
| Tue 10 | both: full slide deck, rehearse twice with a timer, fix gaps | |
| Wed 11 | present; submit report by 11:59 PM | |

## Presentation (15 minutes)
| Slide | Content | Speaker | Time |
|---|---|---|---|
| 1 | Title, task | Ping | 0:45 |
| 2 | Why nerve segmentation in ultrasound; what the data looks like | Nana | 1:15 |
| 3 | Dataset issues: tiny target, patient details removed, near-duplicates | Nana | 1:30 |
| 4 | U-Net basics | Ping | 1:30 |
| 5 | nnU-Net's idea: fixed, rule-based, empirical parameters | Ping | 2:00 |
| 6 | What nnU-Net chose for our data and why it fits | Ping | 1:30 |
| 7 | Training setup (loss, optimizer, augmentation) | Ping | 1:15 |
| 8 | Experimental design: splits, metrics, overfitting controls | Nana | 1:15 |
| 9 | Results table | Nana | 1:15 |
| 10 | Successes and failures | Nana | 1:15 |
| 11 | Leakage: required vs grouped split; baseline vs improved | Nana | 1:00 |
| 12 | Conclusion and limitations | both | 0:30 |
Ping about 7:30, Nana about 7:30. Both should be able to answer questions on either half.

## OPEN
- Section 2 outline (Thu 5).
- E3 choice (Nana: decide 4 Nov, after the timing runs).
- Training length and 5-fold CV: decide after the timing runs (Nana, 4 Nov: do not rule out 5-fold before seeing them).
- Who merges the report; slide design owner.
- Decided 4 Nov: headline uses `checkpoint_final`; `checkpoint_best` reported only as overfitting evidence.
