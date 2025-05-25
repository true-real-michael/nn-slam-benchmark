#  Copyright (c) 2025, Ivan Moskalenko, Anastasiia Kornilova, Mikhail Kiselev
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Optional

import torch

from nnsb.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.feature_matchers import FeatureMatcher
from nnsb.feature_matchers.lightglue.model.lightglue_matcher import LightGlueMatcher


class LightGlueTorchBackend(TorchBackend):
    """TorchBackend implementation for LightGlue model.

    This backend initializes and manages a LightGlue model for feature matching.
    """

    def __init__(self):
        """Initializes the LightGlue TorchBackend with a model wrapper."""

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = LightGlueMatcher(features="superpoint")

            def forward(self, x):
                return self.model({"image0": x[0], "image1": x[1]})["matches"][0]

        super().__init__(Wrapper())


class LightGlue(FeatureMatcher):
    """
    Implementation of [LightGlue](https://github.com/cvg/LightGlue)
    matcher with SuperPoint extractor.
    """

    def __init__(self, backend: Optional[Backend] = None):
        """Initializes the LightGlue feature matcher.

        Args:
            backend: Optional backend instance. If None, creates a LightGlueTorchBackend.
        """
        super().__init__(backend or LightGlueTorchBackend())

    def preprocess(self, feat):
        """Preprocesses features for the LightGlue model.

        Args:
            feat: Feature dictionary containing keypoints, scores, descriptors.

        Returns:
            Dict with preprocessed features.
        """
        keys = ["keypoints", "scores", "descriptors"]
        feat = {
            k: (torch.tensor(v).to(self.device) if k in keys else v)
            for k, v in feat.items()
        }
        feat["descriptors"] = feat["descriptors"].transpose(-1, -2).contiguous()
        return feat

    def postprocess(self, query_feat, db_feat, matches):
        """Postprocesses LightGlue model outputs.

        Args:
            query_feat: Query features.
            db_feat: Database features.
            matches: Matched indices from the model.

        Returns:
            Tuple containing number of matches, matched query points, and matched database points.
        """
        points_query = query_feat["keypoints"][0][matches[..., 0]].cpu().numpy()
        points_db = db_feat["keypoints"][0][matches[..., 1]].cpu().numpy()
        return len(points_query), points_query, points_db

    def get_sample_input(self):
        """Provides a sample input for the model.

        Returns:
            Dict with sample features.
        """
        return {
            "keypoints": torch.randint(0, 200, (1, 100, 2)),
            "descriptors": torch.rand((1, 256, 100)),
            "scores": torch.rand((1, 100)),
            "image_size": torch.tensor((200, 200)),
        }
