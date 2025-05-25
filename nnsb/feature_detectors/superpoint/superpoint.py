from typing import Optional

import torch

from nnsb.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.feature_detectors.feature_detector import FeatureDetector
from nnsb.feature_detectors.superpoint.model import SuperPoint as SuperPointModule


class SuperPointTorchBackend(TorchBackend):
    """TorchBackend implementation for SuperPoint model.

    This backend initializes and manages a SuperPoint model for feature detection.
    """

    def __init__(self):
        """Initializes the SuperPoint TorchBackend with a SuperPoint model."""
        super().__init__(SuperPointModule())


class SuperPoint(FeatureDetector):
    """SuperPoint feature detector implementation.

    SuperPoint is a self-supervised interest point detector and descriptor.
    """

    def __init__(self, resize, backend: Optional[Backend] = None):
        """Initializes the SuperPoint feature detector.

        Args:
            resize: The image size to resize inputs to.
            backend: Optional backend instance. If None, creates a SuperPointTorchBackend.
        """
        super().__init__(backend or SuperPointTorchBackend(), resize)

    def preprocess(self, x):
        """Preprocesses input for the SuperPoint model.

        Args:
            x: Input tensor.

        Returns:
            Dict containing the preprocessed image.
        """
        x = super().preprocess(x)
        return {"image": x}

    def postprocess(self, x):
        """Postprocesses SuperPoint model output.

        Args:
            x: Model output.

        Returns:
            Dict with keypoints, descriptors, and image size information.
        """
        x["image_size"] = torch.tensor((self.resize, self.resize)).to(self.device)
        return x
