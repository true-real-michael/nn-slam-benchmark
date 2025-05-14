from abc import ABC, abstractmethod

from torch import nn

from nnsb.backend.torch import TorchBackend


class Method(ABC):
    """
    Base class for all methods.
    """

    def _get_model_for_quantization(self, *args, **kwargs):
        """
        Return a model ready for quantization.
        To be implemented by specific methods.
        """
        return self.backend.model

    @abstractmethod
    def get_sample_input(self):
        """
        Return a sample input for the model.
        To be implemented by specific methods.
        """
        pass
    
    def get_quantization_args(self):
        """
        Return a dictionary of additional arguments for quantization.
        Can be overridden by specific methods if needed.
        """
        return {}