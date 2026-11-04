"""E3 candidate A: nnU-Net's default trainer with ultrasound-aware spatial augmentation.

Only two settings differ from nnUNetTrainer, both in configure_rotation_dummyDA_mirroring_and_inital_patch_size:
- rotation limited to +-15 degrees (nnU-Net's default for our 640 x 768 patch is +-180, because the patch aspect
  ratio 1.2 is below its 1.5 threshold);
- mirroring only left-right (spatial axis 1, image columns). nnU-Net's default flips both axes, including
  top-bottom (axis 0, image rows), which turns the skin to the bottom of the scan.
The mirror axes are also the axes nnU-Net uses for test-time flip averaging, so prediction with this trainer
averages over left-right flips only.

Reason (our inference, not from nnU-Net): in these scans the skin is always at the top and depth increases
downward, so top-bottom flips and large rotations make images the scanner never produces.

nnU-Net finds this class through the nnUNet_extTrainer environment variable, which run.py sets to this folder.
Everything else (losses, optimiser, the other augmentations, the schedule) is inherited unchanged.
"""

import numpy as np
import torch

from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerUSAug(nnUNetTrainer):
    MAX_ROTATION_DEG = 15.0
    MIRROR_AXES = (1,)  # left-right only

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 2:
            raise RuntimeError("nnUNetTrainerUSAug is written for the 2d configuration only")
        r = self.MAX_ROTATION_DEG / 360 * 2.0 * np.pi
        rotation_for_DA = (-r, r)
        mirror_axes = self.MIRROR_AXES
        # same call as nnUNetTrainer, with our rotation range
        initial_patch_size = get_patch_size(patch_size[-2:], rotation_for_DA, rotation_for_DA, rotation_for_DA,
                                            (0.85, 1.25))
        self.print_to_log_file("do_dummy_2d_data_aug: False")
        self.print_to_log_file(f"nnUNetTrainerUSAug: rotation +-{self.MAX_ROTATION_DEG} degrees, "
                               f"mirror axes {mirror_axes} (left-right only)")
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, False, initial_patch_size, mirror_axes


def _with_epochs(n: int):
    class _T(nnUNetTrainerUSAug):
        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device('cuda')):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = n

    _T.__name__ = _T.__qualname__ = f"nnUNetTrainerUSAug_{n}epochs"
    return _T


# Same lengths as nnU-Net's own nnUNetTrainer_<n>epochs classes (the plain class is 1000 epochs).
nnUNetTrainerUSAug_5epochs = _with_epochs(5)
nnUNetTrainerUSAug_250epochs = _with_epochs(250)
nnUNetTrainerUSAug_500epochs = _with_epochs(500)
nnUNetTrainerUSAug_750epochs = _with_epochs(750)
