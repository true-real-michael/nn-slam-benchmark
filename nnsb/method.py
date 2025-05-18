from abc import ABC, abstractmethod

import torch

from nnsb.backend.torch import TorchBackend


class Method(ABC):
    """
    Base class for all methods.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    @abstractmethod
    def get_torch_backend(*args, **kwargs) -> TorchBackend:
        """
        Return the torch backend for the method.
        To be implemented by specific methods.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_input(self) -> torch.Tensor:
        """
        Return a sample input for the model.
        To be implemented by specific methods.
        """
        pass
