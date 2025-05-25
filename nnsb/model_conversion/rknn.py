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
from tempfile import NamedTemporaryFile
from typing import Literal, Optional

from nnsb.model_conversion.onnx import OnnxExportable
from nnsb.model_conversion.torchscript import TorchScriptExportable


class RknnExportable(TorchScriptExportable, OnnxExportable):
    """Mixin class for models that can be exported to RKNN format.
    
    This class provides functionality to export models to the RKNN format,
    which is optimized for Rockchip Neural Network processors.
    """
    
    def export_rknn(
        self,
        output: Path,
        intermediate_format: Literal["torchscript", "onnx"] = "torchscript",
        quantization_dataset: Optional[Path] = None,
    ):
        """Exports the model to the RKNN format.
        
        Args:
            output: Path where the RKNN model will be saved.
            intermediate_format: The format used as an intermediate step in the conversion process.
                Can be either "torchscript" or "onnx". Default is "torchscript".
            quantization_dataset: Optional path to a dataset for quantization.
                
        Raises:
            ImportError: If the RKNN API is not installed.
            RuntimeError: If model loading, building, or export fails.
        """
        try:
            from rknn.api import RKNN
        except ImportError:
            raise ImportError(
                "RKNN API is not installed. Please install it to use RKNN export."
            )

        suffix = ".onnx" if intermediate_format == "onnx" else ".pt"

        with NamedTemporaryFile(suffix=suffix) as file:
            if intermediate_format == "onnx":
                self.export_onnx(Path(file.name))
            else:
                self.export_torchscript(Path(file.name))

            input_shape = [1, 3, self.resize, self.resize]
            rknn = RKNN(verbose=True)
            rknn.config(
                mean_values=[124.16, 116.736, 103.936],
                std_values=[58.624, 57.344, 57.6],
                target_platform="rk3588",
            )

            if intermediate_format == "onnx":
                ret = rknn.load_onnx(model=file.name)
            else:
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
