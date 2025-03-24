import numpy as np
from nnsb.backend.backend import Backend


class DefaultBackend(Backend):
    def __init__(self, model, *args, **kwargs):
        self.model = model

    def __call__(self, x: np.ndarray):
        return self.model(x)
