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
    def __init__(self):
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
    def __init__(self, backend: Optional[Backend] = None):
        super().__init__(backend or LighterGlueTorchBackend())

    def preprocess(self, features):
        keys = ["keypoints", "scores", "descriptors", "image_size"]
        features = {
            k: (torch.tensor(v).to(self.device) if k in keys else v)
            for k, v in features.items()
        }
        return features

    def postprocess(self, query_feat, db_feat, matches):
        points_query = query_feat["keypoints"][matches[..., 0]].cpu().numpy()
        points_db = db_feat["keypoints"][matches[..., 1]].cpu().numpy()
        return len(points_query), points_query, points_db

    def get_sample_input(self):
        return {
            "keypoints": torch.randint(0, 200, (1, 100, 2)),
            "descriptors": torch.rand((1, 256, 100)),
            "scores": torch.rand((1, 100)),
            "image_size": torch.tensor((200, 200)),
        }
