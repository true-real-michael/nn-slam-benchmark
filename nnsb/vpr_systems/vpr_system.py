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

from abc import ABC, abstractmethod

from nnsb.utils import transform_image_for_vpr


class VPRSystem(ABC):
    def __init__(self, resize: int):
        """
        :param gpu_index: The index of the GPU to be used
        """
        self.resize = resize
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        print('Running inference on device "{}"'.format(self.device))

    # @abstractmethod
    # def _get_torch_backend(self, args):
    #     pass

    def preprocess(self, x):
        return transform_image_for_vpr(x, self.resize).to(self.device)[None, :]
    
    def postprocess(self, x):
        return x.cpu().numpy()[0]

    @abstractmethod
    def get_image_descriptor(self, image: np.ndarray):
        """
        Gets the descriptor of the image given
        :param image: Image in the OpenCV format
        :return: Descriptor of the image
        """
        pass

