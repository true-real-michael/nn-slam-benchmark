from abc import ABC, abstractmethod
import numpy as np


class Backend(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def __call__(self, x):
        pass
