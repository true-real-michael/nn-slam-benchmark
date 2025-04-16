import numpy as np
from abc import ABC, abstractmethod
from nnsb.backend.backend import Backend


class TorchBackend(Backend, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x: np.ndarray):
        return self.model(x)
