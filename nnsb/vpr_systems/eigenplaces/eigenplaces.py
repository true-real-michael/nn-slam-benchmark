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
from typing import Optional

import torch

from nnsb.backend.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.model_conversion.rknn import RknnExportable
from nnsb.model_conversion.tensorrt import TensorRTExportable
from nnsb.vpr_systems.vpr_system import VPRSystem


class EigenPlacesTorchBackend(TorchBackend):
    def __init__(self, backbone, fc_output_dim):
        super().__init__(
            torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone=backbone,
                fc_output_dim=fc_output_dim,
            )
        )


class EigenPlaces(VPRSystem, RknnExportable, TensorRTExportable):
    """
    Implementation of [EigenPlaces](https://github.com/gmberton/EigenPlaces) global localization method.
    """

    def __init__(
        self,
        backbone: str = "ResNet101",
        fc_output_dim: int = 2048,
        resize: int = 800,
        backend: Optional[Backend] = None,
    ):
        """
        :param backbone: Type of backbone
        :param fc_output_dim: Dimension of descriptors
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        """
        super().__init__(
            backend or EigenPlacesTorchBackend(backbone, fc_output_dim), resize
        )
