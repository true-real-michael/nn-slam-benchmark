#  Copyright (c) 2024, Feng Lu, Ivan Moskalenko, Anastasiia Kornilova
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
import numpy as np
import torch
from pathlib import Path

from nnsb.model_conversion.torchscript import TorchScriptExportable
from nnsb.vpr_systems.sela.network import GeoLocalizationNet
from nnsb.vpr_systems.vpr_system import VPRSystem
from nnsb.model_conversion.onnx import OnnxExportable


def get_torch_module(model):
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            model = model
        def forward(self, x):
            return model.global_feat(x)
        
    return Wrapper(model)



class Sela(VPRSystem, OnnxExportable, TorchScriptExportable):
    """
    Wrapper for [Sela](https://github.com/Lu-Feng/SelaVPR) VPR method
    """

    def __init__(
        self,
        path_to_state_dict = None,
        dinov2_path = None,
        backend = None
    ):
        """
        :param path_to_state_dict: Path to the SelaVPR weights
        :param dinov2_path: Path to the DINOv2 (ViT-L/14) foundation model
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(224)


        if backend is None:
            model = GeoLocalizationNet(dinov2_path)
            model = model.eval().to(self.device)

            state_dict = torch.load(path_to_state_dict, map_location=self.device)["model_state_dict"]
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            self.model = get_torch_module(model)
        else:
            self.model = backend

    def get_image_descriptor(self, x: np.ndarray):
        x = self.preprocess(x)
        with torch.no_grad():
            x = self.model(x)
        # return self.postprocess(x)

    def do_export_onnx(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        
        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model.global_feat(x)

        wrapper = Wrapper(self.model)
        torch.onnx.export(
            wrapper,
            (torch.ones((1, 3, self.resize // 14 * 14, self.resize // 14 * 14)),),
            str(output),
        )

    def do_export_torchscript(self, output: Path):
        model = self.get_torch_module()
        trace = torch.jit.trace(
            model, torch.Tensor(1, 3, self.resize, self.resize).to(self.device)
        )
        trace.save(str(output))
