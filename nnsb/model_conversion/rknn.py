from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Optional

from nnsb.model_conversion.onnx import OnnxExportable
from nnsb.model_conversion.torchscript import TorchScriptExportable


class RknnExportable(TorchScriptExportable, OnnxExportable):
    def export_rknn(self, output: Path, intermediate_format: Literal["torchscript", "onnx"] = "torchscript",
                    quantization_dataset: Optional[Path] = None):
        """
        Export the model to the RKNN format.
        :param output: The path to save the model
        :param intermediate_format: The intermediate format to use for export, either "torchscript" or "onnx"
        :param quantization_dataset: The dataset to use for quantization, if any
        """
        try:
            from rknn.api import RKNN
        except ImportError:
            raise ImportError("RKNN API is not installed. Please install it to use RKNN export.")

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
