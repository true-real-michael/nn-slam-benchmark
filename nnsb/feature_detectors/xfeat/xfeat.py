#  Copyright (c) 2025, Mikhail Kiselev, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Optional

import torch

from nnsb.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.feature_detectors.feature_detector import FeatureDetector


class XFeatTorchBackend(TorchBackend):
    """TorchBackend implementation for XFeat model.

    This backend initializes and manages an XFeat model for feature detection.
    """

    def __init__(self):
        """Initializes the XFeat TorchBackend with a model wrapper."""

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
    """XFeat feature detector implementation.

    XFeat is an accelerated feature extractor providing high-quality keypoints and descriptors.
    """

    def __init__(self, resize, backend: Optional[Backend] = None):
        """Initializes the XFeat feature detector.

        Args:
            resize: The image size to resize inputs to.
            backend: Optional backend instance. If None, creates an XFeatTorchBackend.
        """
        super().__init__(backend or XFeatTorchBackend(), resize)

    def postprocess(self, x):
        """Postprocesses XFeat model output.

        Args:
            x: Model output.

        Returns:
            Dict with added image size information.
        """
        x["image_size"] = (
            torch.tensor((self.resize, self.resize)).to(self.device).float()
        )
        return x
