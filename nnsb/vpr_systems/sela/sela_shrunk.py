#  Copyright (c) 2025, Feng Lu, Ivan Moskalenko, Anastasiia Kornilova, Mikhail Kiselev
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
from pathlib import Path

from nnsb.backend.torch import TorchBackend
from nnsb.model_conversion.rknn import RknnExportable
from nnsb.model_conversion.tensorrt import TensorRTExportable
from nnsb.vpr_systems.sela.network import GeoLocalizationNet
from nnsb.vpr_systems.vpr_system import VPRSystem


class SelaShrunkTorchBackend(TorchBackend):
    def __init__(self, path_to_state_dict, dinov2_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GeoLocalizationNet(dinov2_path)
        state_dict = torch.load(path_to_state_dict, map_location=device)[
            "model_state_dict"
        ]
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        super().__init__(model.backbone)


class SelaShrunk(VPRSystem, RknnExportable, TensorRTExportable):
    """
    Wrapper for [Sela](https://github.com/Lu-Feng/SelaVPR) VPR method
    """

    def __init__(self, path_to_state_dict=None, dinov2_path=None, backend=None):
        """
        :param path_to_state_dict: Path to the SelaVPR weights
        :param dinov2_path: Path to the DINOv2 (ViT-L/14) foundation model
        """
        super().__init__(
            backend or SelaShrunkTorchBackend(path_to_state_dict, dinov2_path), 224
        )
        model = GeoLocalizationNet(dinov2_path)
        state_dict = torch.load(path_to_state_dict, map_location=self.device)[
            "model_state_dict"
        ]
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        self.aggregator = model.aggregation.eval().to(self.device)

    def export_rknn(
        self,
        output: Path,
        intermediate_format="onnx",
        quantization_dataset: Optional[Path] = None,
    ):
        """
        Export the model to RKNN format.
        :param output: Path to save the exported model.
        :param quantization_dataset: Path to the dataset for quantization.
        """
        super().export_rknn(
            output,
            intermediate_format="onnx",
            quantization_dataset=quantization_dataset,
        )

    def postprocess(self, x):
        patch_feature = x["x_norm_patchtokens"].view(-1, 16, 16, 1024)
        x1 = patch_feature.permute(0, 3, 1, 2)
        with torch.no_grad():
            x1 = self.aggregator(x1)
        global_feature = torch.nn.functional.normalize(x1, p=2, dim=-1)
        return super().postprocess(global_feature)
