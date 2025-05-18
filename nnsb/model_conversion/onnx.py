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


class OnnxExportable(Method):
    def do_export_onnx(self, output: Path):
        torch.onnx.export(
            self.get_torch_backend().get_torch_module().cpu(),
            (self.get_sample_input(),),
            str(output),
        )

    def export_onnx(self, output: Path):
        """
        Export the model to the ONNX format.
        :param output: The path to save the model
        """
        output.parent.mkdir(exist_ok=True, parents=True)
        self.do_export_onnx(output)
