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
    def do_export_torchscript(self, output: Path):
        model = self.get_torch_backend().get_torch_module().cpu()
        trace = torch.jit.trace(
            model, self.get_sample_input()
        )
        trace.save(str(output))

    def export_torchscript(self, output: Path):
        """
        Export the model to the TorchScript format.
        :param output: The path to save the model
        """
        output.parent.mkdir(exist_ok=True, parents=True)
        self.do_export_torchscript(output)
