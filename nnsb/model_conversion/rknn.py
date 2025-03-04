from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from rknn.api import RKNN

from nnsb.model_conversion.torchscript import TorchScriptExportable


def export_rknn(
    model: TorchScriptExportable,
    output: Path,
    quantization_dataset: Optional[Path] = None,
):
    """
    Export the model to the RKNN format.
    :param output: The path to save the model
    """
    with NamedTemporaryFile(suffix=".pt") as file:
        model.do_export_torchscript(file.name)

        input_shape = [1, 3, model.resize, model.resize]
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
