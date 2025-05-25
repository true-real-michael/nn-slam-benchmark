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
from abc import ABC

import torch

from nnsb.backend import Backend
from nnsb.method import Method


class FeatureDetector(Method, ABC):
    """Base class for all feature detectors.

    This abstract class defines the interface for feature detectors, which
    extract keypoints and descriptors from images.

    Attributes:
        resize: The size to resize images to.
    """

    def __init__(self, backend: Backend, resize: int):
        """Initializes the feature detector.

        Args:
            backend: The backend to use for inference.
            resize: The size to resize images to.
        """
        self.resize = resize
        super().__init__(backend)

    def preprocess(self, x):
        """Preprocesses an input image for feature detection.

        Args:
            x: Input image tensor.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        return x.to(self.device).unsqueeze(0)

    def postprocess(self, x):
        """Postprocesses the feature detector output.

        Args:
            x: Feature detector output.

        Returns:
            Postprocessed features, usually a dictionary with keys like
            'keypoints', 'descriptors', etc.
        """
        return x

    def __call__(self, x):
        """Runs the feature detection pipeline.

        Args:
            x: Input image tensor.

        Returns:
            Detected features with keypoints and descriptors.
        """
        x = self.preprocess(x)
        x = self.backend(x)
        return self.postprocess(x)

    def get_sample_input(self) -> torch.Tensor:
        """Returns a sample input tensor for the model.

        Returns:
            torch.Tensor: A tensor of shape (1, 1, resize, resize).
        """
        return torch.randn(1, 1, self.resize, self.resize).cpu()
