from pathlib import Path

import torch

from nnsb.backend.backend import Backend


class TorchscriptBackend(Backend):
    def __init__(self, model_path: Path, *args, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path).to(self.device).eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x.to(self.device))
