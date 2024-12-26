from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite

from aero_vloc.vpr_systems import VPRSystem


class RKNN(VPRSystem):
    def __init__(self, model_path: Path):
        super().__init__(0)
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path.as_posix())
        if ret != 0:
            raise ValueError(f"Failed to load model, model path: {model_path}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise ValueError(f"Failed to init runtime")

    def get_image_descriptor(self, image: np.ndarray):
        return self.rknn.inference([image])[0][0, :]
