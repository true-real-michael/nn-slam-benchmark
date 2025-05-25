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

import torch

from nnsb.backend.backend import Backend


class TorchscriptBackend(Backend):
    """Backend for TorchScript models.

    This backend executes models serialized with TorchScript, which offers
    a way to run PyTorch models in environments without Python dependencies.

    Attributes:
        device: The device to run inference on (CPU or CUDA).
        model: The loaded TorchScript model.
    """

    def __init__(self, model_path: Path, *args, **kwargs):
        """Initializes the TorchScript backend.

        Args:
            model_path: Path to the TorchScript model file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path).to(self.device).eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Runs inference using the TorchScript model.

        Args:
            x: Input tensor for inference.

        Returns:
            torch.Tensor: The model's output after inference.
        """
        with torch.no_grad():
            return self.model(x.to(self.device))
