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
from abc import ABC

import torch
from nnsb.backend.backend import Backend


class TorchBackend(Backend, ABC):
    """Backend for PyTorch models.

    This backend executes models using the PyTorch framework.

    Attributes:
        device: The device to run inference on (CPU or CUDA).
        model: The PyTorch model to execute.
    """

    def __init__(self, model: torch.nn.Module):
        """Initializes the PyTorch backend.

        Args:
            model: The PyTorch model to use for inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)

    def __call__(self, x):
        """Runs the model on the input data.

        Args:
            x: Input data for model inference.

        Returns:
            The model's output after inference.
        """
        with torch.no_grad():
            return self.model(x)

    def get_torch_module(self) -> torch.nn.Module:
        """Returns the torch module of the backend.

        Returns:
            torch.nn.Module: The PyTorch model.
        """
        return self.model
