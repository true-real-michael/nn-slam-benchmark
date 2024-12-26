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
import numpy as np
import torch

from aero_vloc.utils import transform_image_for_vpr
from aero_vloc.vpr_systems.vpr_system import VPRSystem
from aero_vloc.model_conversion.rk3588 import RknnExportable


class EigenPlaces(VPRSystem, RknnExportable):
    """
    Implementation of [EigenPlaces](https://github.com/gmberton/EigenPlaces) global localization method.
    """

    def __init__(
        self,
        backbone: str = "ResNet101",
        fc_output_dim: int = 2048,
        resize: int = 800,
        gpu_index: int = 0,
    ):
        """
        :param backbone: Type of backbone
        :param fc_output_dim: Dimension of descriptors
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(gpu_index)
        self.resize = resize
        self.backbone = backbone
        self.fc_output_dim = fc_output_dim

        self.model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone=backbone,
            fc_output_dim=fc_output_dim,
        )
        self.model.eval().to(self.device)

    def get_image_descriptor(self, image: np.ndarray):
        image = transform_image_for_vpr(image, self.resize)[None, :].to(self.device)
        with torch.no_grad():
            descriptor = self.model(image)
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor

    def export_torchscript(self, output: Path):
        trace = torch.jit.trace(self.model, torch.Tensor(1, 3, self.resize, self.resize).to(self.device))
        trace.save(output)
    
