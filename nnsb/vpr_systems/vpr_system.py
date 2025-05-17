#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
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
import torch
import numpy as np

from nnsb.method import Method
from nnsb.utils import transform_image_for_vpr


class VPRSystem(Method):
    def __init__(self, resize: int):
        """
        :param gpu_index: The index of the GPU to be used
        """
        self.resize = resize
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        print('Running inference on device "{}"'.format(self.device))

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

