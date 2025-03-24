from pathlib import Path
from onnxruntime import InferenceSession

from nnsb.backend.backend import Backend


class OnnxBackend(Backend):
    def __init__(self, model_path: Path, *args, **kwargs):
        self.session = InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def __call__(self, x):
        x = x.cpu().numpy()
        x = self.session.run(None, {"input.1": x})
        # x = self.session.run(None, x)
        x[0]
        return x
