from pathlib import Path
from nnsb.backend.backend import Backend
import torch


class TorchscriptBackend(Backend):
    def __init__(self, model_path: Path):
        self.model = torch.jit.load(model_path)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
