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
from typing import Optional
import numpy as np
import torch

from nnsb.backend.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.model_conversion.rknn import RknnExportable
from nnsb.model_conversion.torchscript import TorchScriptExportable
from nnsb.utils import transform_image_for_vpr
from nnsb.vpr_systems.vpr_system import VPRSystem
from nnsb.model_conversion.onnx import OnnxExportable


class SaladShrunkTorchBackend(TorchBackend):
    def __init__(self):
        super().__init__()
        model = torch.hub.load("serizba/salad", "dinov2_salad").eval().to(self.device)
        model.eval().to(self.device)
        self.model = model.backbone


class SALADShrunk(VPRSystem, RknnExportable):
    """
    Wrapper for [SALAD](https://github.com/serizba/salad) VPR method
    """

    def __init__(
        self,
        resize: int = 800,
        backend: Optional[Backend] = None,
    ):
        """
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(resize // 14 * 14)
        self.backend = backend or SaladShrunkTorchBackend()
        model = torch.hub.load("serizba/salad", "dinov2_salad").eval().to(self.device)
        self.aggregator = model.aggregator.eval().to(self.device)

    def postprocess(self, x):
        with torch.no_grad():
            x = self.aggregator(x)
        return super().postprocess(x)
    
    def export_rknn(self, output: Path, quantization_dataset: Optional[Path] = None):
        """
        Export the model to RKNN format.
        :param output: Path to save the exported model.
        :param quantization_dataset: Path to the dataset for quantization.
        """
        super().export_rknn(output, intermediate_format="onnx", quantization_dataset=quantization_dataset)