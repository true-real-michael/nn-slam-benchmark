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

from rknnlite.api import RKNNLite
import torch

from nnsb.backend import Backend


class RknnBackend(Backend):
    """Backend for RKNN models.

    This backend executes models optimized for Rockchip Neural Network
    processors using the RKNN Lite runtime.

    Attributes:
        rknn: The RKNNLite runtime instance.
    """

    def __init__(self, model_path: Path):
        """Initializes the RKNN backend.

        Args:
            model_path: Path to the RKNN model file.

        Raises:
            ValueError: If model loading or runtime initialization fails.
        """
        super().__init__()
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path.as_posix())
        if ret != 0:
            raise ValueError(f"Failed to load model, model path: {model_path}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise ValueError("Failed to init runtime")

    def __del__(self):
        """Releases RKNN resources when the object is destroyed."""
        self.rknn.release()

    def __call__(self, x):
        """Runs inference using the RKNN model.

        Args:
            x: Input tensor for inference.

        Returns:
            torch.Tensor: The model's output after inference.
        """
        x = x.numpy()
        x = self.rknn.inference([x])
        x = x[0][0, :]
        x = torch.tensor(x)
        return x
