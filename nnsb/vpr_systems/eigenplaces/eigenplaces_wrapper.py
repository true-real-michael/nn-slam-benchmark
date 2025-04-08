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

import numpy as np
import torch

from nnsb.model_conversion.torchscript import TorchScriptExportable
from nnsb.model_conversion.onnx import OnnxExportable
from nnsb.utils import transform_image_for_vpr
from nnsb.vpr_systems.vpr_system import VPRSystem


class EigenPlaces(VPRSystem):
    def __init__(
        self,
        model_path,
        backend_type,
    ):
        super().__init__()
        self.resize = 800
        self.backend = backend_type(model_path)

    def get_image_descriptor(self, image: np.ndarray):
        image = transform_image_for_vpr(image, self.resize)[None, :].to(self.device)
        descriptor = self.backend(image)
        return descriptor


