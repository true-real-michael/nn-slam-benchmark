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
from onnxruntime import InferenceSession

from nnsb.backend.backend import Backend


class OnnxBackend(Backend):
    """Backend that uses ONNX Runtime to run the model.

    This backend executes models converted to ONNX format using ONNX Runtime.
    It supports both CPU and CUDA execution providers.

    Attributes:
        session: The ONNX Runtime inference session.
    """

    def __init__(self, model_path: Path, *args, **kwargs):
        """Initializes the ONNX backend.

        Args:
            model_path: Path to the ONNX model file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.session = InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def __call__(self, x):
        """Runs inference using the ONNX model.

        Args:
            x: Input tensor for inference.

        Returns:
            The model's output after inference.
        """
        x = x.cpu().numpy()
        x = self.session.run(None, {"input.1": x})
        return x
