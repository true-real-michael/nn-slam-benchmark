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
from nnsb.utils import transform_image_for_vpr
from nnsb.vpr_systems.vpr_system import VPRSystem
from nnsb.model_conversion.onnx import OnnxExportable


class SALAD(VPRSystem, OnnxExportable, TorchScriptExportable):
    """
    Wrapper for [SALAD](https://github.com/serizba/salad) VPR method
    """

    def __init__(
        self,
        resize: int = 800,
        gpu_index: int = 0,
    ):
        """
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(gpu_index)
        self.resize = resize // 14 * 14
        self.model = torch.hub.load("serizba/salad", "dinov2_salad").eval().to(self.device)

    def get_image_descriptor(self, image: np.ndarray):
        image = transform_image_for_vpr(image, self.resize).to(self.device)[None, :]
        with torch.no_grad():
            descriptor = self.model(image)
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor

    def do_export_onnx(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self.model,
            (torch.ones((1, 3, self.resize // 14 * 14, self.resize // 14 * 14)),),
            str(output),
        )

    def do_export_torchscript(self, output: Path):
        trace = self.model.to_torchscript(
            method="trace", example_inputs=torch.randn(1, 3, self.resize, self.resize)
        )
        trace.save(str(output))
