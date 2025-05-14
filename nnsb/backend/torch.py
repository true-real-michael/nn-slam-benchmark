import numpy as np
from abc import ABC, abstractmethod

import torch
from nnsb.backend.backend import Backend


class TorchBackend(Backend, ABC):
    def __init__(self, *args, **kwargs):
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, x):
        with torch.no_grad():
            x = self.model(x)
        return x.cpu()
    
    def get_torch_module(self) -> torch.nn.Module:
        """
        Returns the torch module of the backend.
        This method should be implemented by subclasses.
        """
        return self.model
