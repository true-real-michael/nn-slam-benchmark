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
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch
import numpy as np

from abc import ABC, abstractmethod

from aero_vloc.model_conversion import rk3588


class VPRSystem(ABC):
    def __init__(self, gpu_index: int = 0):
        """
        :param gpu_index: The index of the GPU to be used
        """
        self.device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        print('Running inference on device "{}"'.format(self.device))

    @abstractmethod
    def get_image_descriptor(self, image: np.ndarray):
        """
        Gets the descriptor of the image given
        :param image: Image in the OpenCV format
        :return: Descriptor of the image
        """
        pass

class RknnExportable(ABC):
    def export_rknn(self, output: Path):
        """
        Export the model to the RKNN format.
        :param output: The path to save the model
        """
        with NamedTemporaryFile(suffix='.pt') as file:
            self.export_torchscript(file.name)
            rk3588.convert(file.name, (self.resize, self.resize), output)
