#  Copyright (c) 2025, Ivan Moskalenko, Anastasiia Kornilova, Mikhail Kiselev
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
import numpy as np

from nnsb.backend import Backend
from nnsb.method import Method


class VPRSystem(Method, ABC):
    """Base class for all visual place recognition systems.

    This abstract class defines the interface for visual place recognition (VPR)
    systems. VPR systems take an image as input and produce a descriptor that can
    be used for place recognition.

    Attributes:
        resize: The size to resize images to.
    """

    def __init__(self, backend: Backend, resize: int):
        """Initializes the VPR system.

        Args:
            backend: The backend to use for inference.
            resize: The size to resize images to.
        """
        super().__init__(backend)
        self.resize = resize

    def get_sample_input(self) -> torch.Tensor:
        """Returns a sample input tensor for the model.

        Returns:
            torch.Tensor: A tensor of shape (1, 3, resize, resize).
        """
        return torch.randn((1, 3, self.resize, self.resize)).cpu()

    def preprocess(self, x) -> torch.Tensor:
        """Preprocesses an input image for inference.

        Args:
            x: Input image tensor.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        return x.to(self.device).unsqueeze(0)

    def postprocess(self, x) -> torch.Tensor:
        """Postprocesses the model output.

        Args:
            x: Model output tensor.

        Returns:
            torch.Tensor: Postprocessed output tensor.
        """
        return x.cpu().numpy()[0]

    def get_image_descriptor(self, x: np.ndarray):
        """Gets the descriptor of a given image.

        Args:
            x: Input image in the OpenCV format.

        Returns:
            Image descriptor as a numpy array.
        """
        x = self.preprocess(x)
        x = self.backend(x)
        return self.postprocess(x)
