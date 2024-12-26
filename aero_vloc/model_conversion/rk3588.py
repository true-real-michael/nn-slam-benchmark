from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import torch
from rknn.api import RKNN


def convert(
    model: Path,
    shape: Tuple[int, int],
    output: Path,
    quantization_dataset: Path | None = None,
):
    input_shape = [1, 3, shape[0], shape[1]]

    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[124.16, 116.736, 103.936],
        std_values=[58.624, 57.344, 57.6],
        target_platform="rk3588",
    )

    ret = rknn.load_pytorch(model=model, input_size_list=[input_shape])

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

