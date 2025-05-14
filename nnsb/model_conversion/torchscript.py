from pathlib import Path

import torch

from nnsb.method import Method


class TorchScriptExportable(Method):
    def do_export_torchscript(self, output: Path):
        model = self.backend.get_torch_module().cpu()
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
