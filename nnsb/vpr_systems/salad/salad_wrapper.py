from pathlib import Path

import numpy as np

from nnsb.utils import transform_image_for_vpr
from nnsb.vpr_systems.vpr_system import VPRSystem


class SALAD(VPRSystem):
    def __init__(self, model_path: Path, backend_type, resize: int):
        super().__init__()
        self.backend = backend_type(model_path)
        self.backend.model.aggregator = self.backend.model.aggregator.to(
            self.device
        ).eval()
        self.resize = resize // 14 * 14

    def get_image_descriptor(self, image: np.ndarray) -> np.ndarray:
        image = transform_image_for_vpr(image, self.resize).to(self.device)[None, ...]
        result = self.backend(image)
        return result[0]
