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
import torch

from nnsb.backend.torch import TorchBackend
from nnsb.model_conversion.torchscript import TorchScriptExportable
from nnsb.vpr_systems.sela.network import GeoLocalizationNet
from nnsb.vpr_systems.vpr_system import VPRSystem
from nnsb.model_conversion.onnx import OnnxExportable


class SelaTorchBackend(TorchBackend):
    def __init__(self, path_to_state_dict, dinov2_path):
        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.global_feat(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GeoLocalizationNet(dinov2_path)
        state_dict = torch.load(path_to_state_dict, map_location=device)[
            "model_state_dict"
        ]
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        self.model = Wrapper(model).eval().to(self.device)


class Sela(VPRSystem, OnnxExportable, TorchScriptExportable):
    """
    Wrapper for [Sela](https://github.com/Lu-Feng/SelaVPR) VPR method
    """

    def __init__(self, path_to_state_dict=None, dinov2_path=None, backend=None):
        """
        :param path_to_state_dict: Path to the SelaVPR weights
        :param dinov2_path: Path to the DINOv2 (ViT-L/14) foundation model
        """
        super().__init__(224)
        self.backend = backend or self.get_torch_backend(
            path_to_state_dict, dinov2_path
        )

    @staticmethod
    def get_torch_backend(*args, **kwargs) -> TorchBackend:
        return SelaTorchBackend(*args, **kwargs)
