from pathlib import Path
from onnxruntime import InferenceSession

from nnsb.backend.backend import Backend


class OnnxBackend(Backend):
    def __init__(self, model_path: Path):
        self.session = InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def __call__(self, x):
        x = x.numpy()
        x = self.session.run(None, {"input.1": x})
        return x
