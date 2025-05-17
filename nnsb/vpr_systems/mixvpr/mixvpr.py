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
from typing import Optional

import numpy as np
import torch

from nnsb.backend.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.model_conversion.torchscript import TorchScriptExportable
from nnsb.model_conversion.onnx import OnnxExportable
from nnsb.vpr_systems.mixvpr.model.mixvpr_model import VPRModel
from nnsb.vpr_systems.vpr_system import VPRSystem

MIXVPR_RESIZE = 320


class MixVprTorchBackend(TorchBackend):
    def __init__(self, ckpt_path: Optional[str] = None):
        model = VPRModel(
            backbone_arch="resnet50",
            layers_to_crop=[4],
            agg_arch="MixVPR",
            agg_config={
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4,
            },
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        super().__init__(model)


class MixVPR(VPRSystem, TorchScriptExportable, OnnxExportable):
    """
    Implementation of [MixVPR](https://github.com/amaralibey/MixVPR) global localization method.
    """

    def __init__(self, ckpt_path: Optional[str] = None, backend: Optional[Backend] = None):
        """
        :param ckpt_path: Path to the checkpoint file
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(MIXVPR_RESIZE)
        self.backend = backend or self.get_torch_backend(ckpt_path)

    @staticmethod
    def get_torch_backend(*args, **kwargs) -> TorchBackend:
        return MixVprTorchBackend(*args, **kwargs)
