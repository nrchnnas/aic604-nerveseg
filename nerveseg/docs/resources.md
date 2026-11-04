# Reading list

Marked by how each was checked: "verified 4 Nov" means the reference was looked up this session; "standard" means a well-known reference not looked up this session (check the link before citing).

## For report section 2 (nnU-Net)
1. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203-211 (2021). https://doi.org/10.1038/s41592-020-01008-z (verified 4 Nov). The main paper: fixed, rule-based and empirical parameters; results on 23 public datasets.
2. Isensee F, Wald T, Ulrich C, Baumgartner M, Roy S, Maier-Hein K, Jaeger PF. nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. MICCAI 2024. https://arxiv.org/abs/2404.09556 (verified 4 Nov). Argues that well-configured CNN U-Nets still match or beat newer Transformer and Mamba designs; introduces the residual-encoder presets (M, L, XL).
3. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597 (standard). The base architecture.
4. Fabian Isensee's talk on nnU-Net, "Beyond the Patterns" series, FAU, 2021 (53 min). https://www.fau.tv/clips/29-beyond-the-patterns-fabian-isensee-nnu-net-a-self-configuring-method-for-deep-learning-based-biomedical-image-segmentation-w20 (found 4 Nov). A good first watch before reading the paper.

## Building blocks mentioned in the configuration
5. Ulyanov D, Vedaldi A, Lempitsky V. Instance Normalization: The Missing Ingredient for Fast Stylization. 2016. https://arxiv.org/abs/1607.08022 (standard).
6. Milletari F, Navab N, Ahmadi SA. V-Net (introduces the soft Dice loss). 3DV 2016. https://arxiv.org/abs/1606.04797 (standard).
7. Lee CY, Xie S, Gallagher P, Zhang Z, Tu Z. Deeply-Supervised Nets. AISTATS 2015. https://arxiv.org/abs/1409.5185 (standard).

## nnU-Net documentation (GitHub, MIC-DKFZ/nnUNet, folder `documentation/`)
- `how_to_use_nnunet.md`: commands, `-device cpu|cuda|mps` (verified 4 Nov).
- `dataset_format.md`: raw data format, PNG support (verified 4 Nov).
- `manual_data_splits.md`: custom `splits_final.json` (verified 4 Nov).
- `resenc_presets.md`: residual-encoder presets (named in nnU-Net's own planning output).

## Course material
- Lecture 1, slides 5 to 8: project rules (80/20 seed 1, Dice, one model per team).
- Lecture 5: IoU (the course's overlap metric); Dice is closely related: Dice = 2 x IoU / (1 + IoU).
- Lab 3: `train_test_split(..., test_size=0.2, random_state=1)`, which our required split copies.
