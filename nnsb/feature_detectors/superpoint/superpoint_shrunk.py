from typing import Optional

import torch

from nnsb.backend.torch import TorchBackend
from nnsb.feature_detectors.feature_detector import FeatureDetector
from nnsb.feature_detectors.superpoint.model_shrunk import SuperPoint as SuperPointModule, SuperPointFrontend


class SuperPointShrunkTorchBackend(TorchBackend):
    def __init__(self):
        super().__init__(SuperPointModule())


class SuperPointShrunk(FeatureDetector):
    def __init__(self, resize, backend: Optional[TorchBackend] = None):
        super().__init__(resize)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resize = resize
        self.backend = backend or self.get_torch_backend()
        self.postprocessor = SuperPointFrontend(resize, resize)

    def preprocess(self, x):
        return super().preprocess(x)

    def postprocess(self, x):
        pts, desc, heatmap = self.postprocessor.process_pts(x[0], x[1])
        return {
            "keypoints": torch.tensor(pts).to(self.device),
            "descriptors": torch.tensor(desc).to(self.device),
            "image_size": torch.tensor((self.resize, self.resize)).to(self.device),
        }

    @staticmethod
    def get_torch_backend():
        return SuperPointShrunkTorchBackend()
