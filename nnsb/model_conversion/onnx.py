from abc import ABC, abstractmethod
from pathlib import Path


class OnnxExportable(ABC):
    @abstractmethod
    def export_onnx(self, output: Path):
        """
        Export the model to the ONNX format.
        :param output: The path to save the model
        """
        pass
