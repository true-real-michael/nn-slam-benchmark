from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from aero_vloc.vpr_systems import VPRSystem


class RKNN(VPRSystem):
    def __init__(self, model_path: Path, resize: int = 800):
        super().__init__(0)
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path.as_posix())
        if ret != 0:
            raise ValueError(f"Failed to load model, model path: {model_path}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise ValueError("Failed to init runtime")
        self.resize = resize

    def __del__(self):
        self.rknn.release()

    def get_image_descriptor(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resize, self.resize))
        image = np.expand_dims(image, 0)

        return self.rknn.inference([image])[0][0, :]
