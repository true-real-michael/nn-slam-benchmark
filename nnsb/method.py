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
from abc import ABC, abstractmethod

import torch

from nnsb.backend import Backend
from nnsb.backend.torch import TorchBackend


__all__ = ["Method"]


class Method(ABC):
    """
    Base class for all methods.
    """

    def __init__(self, backend: Backend):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend

    def get_torch_backend(self) -> TorchBackend:
        """
        Return the torch backend for the method.
        To be implemented by specific methods.
        """
        if isinstance(self.backend, TorchBackend):
            return self.backend
        else:
            raise NotImplementedError

    @abstractmethod
    def get_sample_input(self) -> torch.Tensor:
        """
        Return a sample input for the model.
        To be implemented by specific methods.
        """
        pass
