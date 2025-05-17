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
        super().__init__()
        self.backend = backend or self.get_torch_backend()

    def preprocess(self, features):
        keys = ["keypoints", "scores", "descriptors"]
        return {
            k: (torch.tensor(v).to(self.device) if k in keys else v)
            for k, v in features.items()
        }

    def postprocess(self, query_feat, db_feat, matches):
        points_query = query_feat["keypoints"][matches[..., 0]].cpu().numpy()
        points_db = db_feat["keypoints"][matches[..., 1]].cpu().numpy()
        return len(points_query), points_query, points_db

    @staticmethod
    def get_torch_backend(*args, **kwargs) -> TorchBackend:
        return LighterGlueTorchBackend()
