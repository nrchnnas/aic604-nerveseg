# AIC-604 final project, team 4: nerve segmentation in ultrasound with nnU-Net

Segmenting the nerve in 300 ultrasound images (one nerve outline per image) with nnU-Net (2D), evaluated with Dice on the course's 80/20 split (seed 1). A second split keeps near-duplicate images together, to show how much they inflate Dice.

| Where | What |
|---|---|
| [`nerveseg/README.md`](nerveseg/README.md) | setup (Mac, Windows, NVIDIA), how to run every step, dataset facts |
| [`nerveseg/PLAN.md`](nerveseg/PLAN.md) | current status, experiments E0 to E3, analysis rules, timeline, presentation plan, open decisions |
| [`nerveseg/docs/nnunet_facts.md`](nerveseg/docs/nnunet_facts.md) | what nnU-Net configured for our data, training settings, augmentation (checked in its code) |
| [`nerveseg/docs/resources.md`](nerveseg/docs/resources.md) | reading list |
| [`nerveseg/docs/final project/`](nerveseg/docs/final%20project/) | the course brief (`report (1).pdf`) |
| [`nerveseg/docs/timing/`](nerveseg/docs/timing/) | timing runs (seconds per epoch per machine) |

Quick start, from this folder, with the Python environment set up as in `nerveseg/README.md`:
```
python nerveseg/nnunet/run.py preprocess
python nerveseg/nnunet/run.py check
python nerveseg/nnunet/run.py timing -d 501
```

Data: the repo holds the cropped images and masks (the scanner header, which shows patient details, is removed). The raw images and the course zip are not in git and must not be uploaded anywhere public.

Due: report by 11:59 PM on 11 Nov 2026, presentation on 11 Nov.
