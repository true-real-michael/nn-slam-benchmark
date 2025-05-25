#  Copyright (c) 2025, Mikhail Kiselev, Anastasiia Kornilova
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

import torch

from nnsb.method import Method


class TorchScriptExportable(Method):
    """Mixin class for models that can be exported to TorchScript format.
    
    This class provides functionality to export models to the TorchScript format,
    which enables deployment without Python dependencies.
    """
    
    def do_export_torchscript(self, output: Path):
        """Performs the actual TorchScript export operation.
        
        Args:
            output: Path where the TorchScript model will be saved.
        """
        model = self.get_torch_backend().get_torch_module().cpu()
        trace = torch.jit.trace(
            model, self.get_sample_input()
        )
        trace.save(str(output))

    def export_torchscript(self, output: Path):
        """Exports the model to the TorchScript format.
        
        Args:
            output: Path where the TorchScript model will be saved.
        """
        output.parent.mkdir(exist_ok=True, parents=True)
        self.do_export_torchscript(output)
