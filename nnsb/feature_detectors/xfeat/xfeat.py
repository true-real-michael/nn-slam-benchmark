from typing import Optional

import torch

from nnsb.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.feature_detectors.feature_detector import FeatureDetector


class XFeatTorchBackend(TorchBackend):
    def __init__(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.detectAndCompute(x, top_k=4096)[0]

        xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        )
        super().__init__(Wrapper(xfeat))


class XFeat(FeatureDetector):
    def __init__(self, resize, backend: Optional[Backend] = None):
        super().__init__(resize)
        self.backend = backend or self.get_torch_backend()

    def postprocess(self, x):
        x["image_size"] = torch.tensor((self.resize, self.resize)).to(self.device).float()
        return x

    @staticmethod
    def get_torch_backend():
        return XFeatTorchBackend()
