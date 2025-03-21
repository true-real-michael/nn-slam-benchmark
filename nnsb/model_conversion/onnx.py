from abc import ABC, abstractmethod
from pathlib import Path


class OnnxExportable(ABC):
    @abstractmethod
    def do_export_onnx(self, output: Path):
        """
        Export the model to the ONNX format.
        :param output: The path to save the model
        """
        pass

    def export_onnx(self, output: Path):
        """
        Export the model to the ONNX format.
        :param output: The path to save the model
        """
        output.parent.mkdir(exist_ok=True, parents=True)
        self.do_export_onnx(output)
