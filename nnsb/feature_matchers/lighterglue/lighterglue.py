#  Copyright (c) 2025, Mikhail Kiselev, Anastasiia Kornilova
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
from nnsb.feature_matchers.feature_matcher import FeatureMatcher


class LighterGlueTorchBackend(TorchBackend):
    """TorchBackend implementation for LighterGlue model.

    This backend initializes and manages a LighterGlue model for feature matching.
    """

    def __init__(self):
        """Initializes the LighterGlue TorchBackend with a model wrapper."""

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.match_lighterglue(x[0], x[1])[2]

        xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        )
        super().__init__(Wrapper(xfeat))


class LighterGlue(FeatureMatcher):
    """LighterGlue feature matcher implementation.

    A lightweight feature matcher optimized for speed.
    """

    def __init__(self, backend: Optional[Backend] = None):
        """Initializes the LighterGlue feature matcher.

        Args:
            backend: Optional backend instance. If None, creates a LighterGlueTorchBackend.
        """
        super().__init__(backend or LighterGlueTorchBackend())

    def preprocess(self, features):
        """Preprocesses features for the LighterGlue model.

        Args:
            features: Feature dictionary containing keypoints, scores, descriptors.

        Returns:
            Dict with preprocessed features.
        """
        keys = ["keypoints", "scores", "descriptors", "image_size"]
        features = {
            k: (torch.tensor(v).to(self.device) if k in keys else v)
            for k, v in features.items()
        }
        return features

    def postprocess(self, query_feat, db_feat, matches):
        """Postprocesses LighterGlue model outputs.

        Args:
            query_feat: Query features.
            db_feat: Database features.
            matches: Matched indices from the model.

        Returns:
            Tuple containing number of matches, matched query points, and matched database points.
        """
        points_query = query_feat["keypoints"][matches[..., 0]].cpu().numpy()
        points_db = db_feat["keypoints"][matches[..., 1]].cpu().numpy()
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
