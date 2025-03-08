from pathlib import Path

import numpy as np

from nnsb.utils import transform_image_for_vpr
from nnsb.vpr_systems.vpr_system import VPRSystem

SELA_RESIZE = 224


class Sela(VPRSystem):
    def __init__(self, model_path: Path, backend_type):
        super().__init__()
        self.backend = backend_type(model_path)
        self.resize = SELA_RESIZE

    def get_image_descriptor(self, image: np.ndarray) -> np.ndarray:
        image = transform_image_for_vpr(image, self.resize)[None, :]
        result = self.backend(image)
        return result[0]
