from typing import Optional

import torch

from nnsb.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.feature_detectors.feature_detector import FeatureDetector
from nnsb.feature_detectors.superpoint.model import SuperPoint as SuperPointModule


class SuperPointTorchBackend(TorchBackend):
    def __init__(self):
        super().__init__(SuperPointModule())


class SuperPoint(FeatureDetector):
    def __init__(self, resize, backend: Optional[Backend] = None):
        super().__init__(resize)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resize = resize
        self.backend = backend or self.get_torch_backend()

    def preprocess(self, x):
        x = super().preprocess(x)
        return {"image": x}

    def postprocess(self, x):
        x["image_size"] = torch.tensor((self.resize, self.resize)).to(self.device)
        return x

    @staticmethod
    def get_torch_backend():
        return SuperPointTorchBackend()
