import torch

from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

from rknn.api import RKNN

from aero_vloc.vpr_systems import VPRSystem


class RknnExportable(VPRSystem, ABC):
    @abstractmethod
    def export_torchscript(self, output: Path):
        pass
    
    def export_rknn(self, output: Path, quantization_dataset: Path | None = None):
        """
        Export the model to the RKNN format.
        :param output: The path to save the model
        :param quantization_dataset: Dataset to quantize the model
        """
        with NamedTemporaryFile(suffix='.pt') as file:
            self.export_torchscript(Path(file.name))
            
            input_shape = [1, 3, self.resize, self.resize]
            rknn = RKNN(verbose=True)
            rknn.config(
                mean_values=[124.16, 116.736, 103.936],
                std_values=[58.624, 57.344, 57.6],
                target_platform="rk3588",
            )

            ret = rknn.load_pytorch(model=file.name, input_size_list=[input_shape])

            if ret != 0:
                raise RuntimeError("Failed to load model")

            if quantization_dataset is not None:
                ret = rknn.build(do_quantization=True, dataset=quantization_dataset)
            else:
                ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError("Failed to build model")

            ret = rknn.export_rknn(str(output))
            if ret != 0:
                raise RuntimeError("Export rknn model failed!")

