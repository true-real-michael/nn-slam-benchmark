from abc import ABC

import torch

from nnsb.method import Method


class FeatureDetector(Method, ABC):
    def __init__(self, resize: int):
        self.resize = resize
        super().__init__()

    def preprocess(self, x):
        return x.to(self.device).unsqueeze(0)

    def postprocess(self, x):
        return x

    def __call__(self, x):
        x = self.preprocess(x)
        x = self.backend(x)
        return self.postprocess(x)

    def get_sample_input(self) -> torch.Tensor:
        return torch.randn(1, 1, self.resize, self.resize).cpu()
