from pathlib import Path
from typing import Optional

import torch

from nnsb.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.feature_detectors.feature_detector import FeatureDetector
from nnsb.feature_detectors.superpoint.model_shrunk import (
    SuperPoint as SuperPointModule,
    SuperPointFrontend,
)
from nnsb.model_conversion.rknn import RknnExportable


class SuperPointShrunkTorchBackend(TorchBackend):
    def __init__(self, ckpt: Path):
        model = SuperPointModule()
        model.load_state_dict(
            torch.load(
                ckpt,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )
        super().__init__(model)


class SuperPointShrunk(FeatureDetector, RknnExportable):
    def __init__(
        self, resize, backend: Optional[Backend] = None, ckpt: Optional[Path] = None
    ):
        if not backend and not ckpt:
            raise RuntimeError("Please provide backend or ckpt")
        super().__init__(backend or SuperPointShrunkTorchBackend(ckpt), resize)
        self.postprocessor = SuperPointFrontend(resize, resize)

    def postprocess(self, x):
        pts, desc, heatmap = self.postprocessor.process_pts(
            x[0].cpu().numpy(), x[1].cpu().numpy()
        )
        return {
            "keypoints": torch.tensor(pts).to(self.device),
            "descriptors": torch.tensor(desc).to(self.device),
            "image_size": torch.tensor((self.resize, self.resize)).to(self.device),
        }
