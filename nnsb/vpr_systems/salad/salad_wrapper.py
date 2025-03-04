from pathlib import Path

import numpy as np
from torchvision import transforms as tvf

from nnsb.utils import transform_image_for_vpr
from nnsb.vpr_systems.vpr_system import VPRSystem

SELA_RESIZE = 224


class SALAD(VPRSystem):
    def __init__(self, model_path: Path, backend_type):
        super().__init__()
        self.backend = backend_type(model_path)
        self.resize = SELA_RESIZE

    def get_image_descriptor(self, image: np.ndarray) -> np.ndarray:
        image = transform_image_for_vpr(image, self.resize).to(self.device)
        _, h, w = image.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_cropped = tvf.CenterCrop((h_new, w_new))(image)[None, ...]
        result = self.backend(img_cropped)
        return result[0]
