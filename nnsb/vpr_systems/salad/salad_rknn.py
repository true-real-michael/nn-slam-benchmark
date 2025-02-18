from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import torch

from nnsb.vpr_systems import VPRSystem


class SaladRknn(VPRSystem):
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
        self.aggregator = torch.hub.load("serizba/salad", "dinov2_salad").aggregator
        self.aggregator = self.aggregator.eval().to(self.device)

    def __del__(self):
        self.rknn.release()

    def get_image_descriptor(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resize, self.resize))
        image = np.transpose(image, (2, 0, 1))[None, :].astype(np.float32)
        output = self.rknn.inference([image])
        x = torch.tensor(output[0])
        t = torch.tensor(output[1])
        with torch.no_grad():
            descriptor = self.aggregator((x, t))
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor
