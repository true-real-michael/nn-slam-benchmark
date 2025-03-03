from pathlib import Path

import numpy as np
import torchvision

from nnsb.utils import transform_image_for_vpr
from nnsb.vpr_systems.vpr_system import VPRSystem

MIXVPR_RESIZE = 320


class MixVPR(VPRSystem):
    def __init__(self, model_path: Path, runner_type):
        self.runner = runner_type(model_path)
        self.resize = MIXVPR_RESIZE

    def get_image_descriptor(self, image: np.ndarray) -> np.ndarray:
        image = transform_image_for_vpr(
            image,
            self.resize,
            torchvision.transforms.InterpolationMode.BICUBIC,
        )[None, :]
        result = self.runner(image)
        return result[0]
