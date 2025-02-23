from abc import ABC, abstractmethod
from pathlib import Path


class TorchScriptExportable(ABC):
    @abstractmethod
    def export_torchscript(self, output: Path):
        """
        Export the model to the TorchScript format.
        :param output: The path to save the model
        """
        pass
