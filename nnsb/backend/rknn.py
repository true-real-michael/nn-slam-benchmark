from pathlib import Path

from rknnlite.api import RKNNLite

from nnsb.backend import Backend


class RknnBackend(Backend):
    def __init__(self, model_path: Path):
        super().__init__()
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path.as_posix())
        if ret != 0:
            raise ValueError(f"Failed to load model, model path: {model_path}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise ValueError("Failed to init runtime")

    def __del__(self):
        self.rknn.release()

    def __call__(self, x):
        return self.rknn.inference([x])[0][0, :]
