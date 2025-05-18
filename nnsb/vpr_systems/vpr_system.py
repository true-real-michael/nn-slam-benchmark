#  Copyright (c) 2025, Ivan Moskalenko, Anastasiia Kornilova
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

from nnsb.method import Method


class VPRSystem(Method, ABC):
    def __init__(self, resize: int):
        """
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__()
        self.resize = resize

    def get_sample_input(self) -> torch.Tensor:
        return torch.randn((1, 3, self.resize, self.resize)).cpu()

    def preprocess(self, x) -> torch.Tensor:
        return x.to(self.device).unsqueeze(0)

    def postprocess(self, x) -> torch.Tensor:
        return x.cpu().numpy()[0]

    def get_image_descriptor(self, x: np.ndarray):
        """
        Gets the descriptor of the image given
        :param x: Image in the OpenCV format
        :return: Descriptor of the image
        """
        breakpoint()
        x = self.preprocess(x)
        x = self.backend(x)
        return self.postprocess(x)
