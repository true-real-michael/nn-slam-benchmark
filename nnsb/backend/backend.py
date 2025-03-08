from abc import ABC, abstractmethod
import numpy as np


class Backend(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray):
        pass
