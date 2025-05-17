from abc import ABC, abstractmethod

from torch import nn
import torch

from nnsb.backend.torch import TorchBackend


class Method(ABC):
    """
    Base class for all methods.
    """

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