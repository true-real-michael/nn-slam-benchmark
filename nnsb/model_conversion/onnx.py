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
