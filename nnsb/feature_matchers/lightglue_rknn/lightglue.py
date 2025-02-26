from pathlib import Path

import torch
from rknnlite.api import RKNNLite
from torch.nn import functional as F


def normalize_keypoints(
    kpts: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    size = torch.tensor([w, h], dtype=torch.float32, device=kpts.device)
    shift = size / 2
    scale = size.max() / 2
    kpts = (kpts - shift) / scale
    return kpts


class LightGlueRknn:
    def __init__(
        self, model_path: Path, n_feat: int, filter_matches_threshold: float = 0.1
    ):
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Could not load model from {model_path}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError("Could not load init runtime")
        self.n_feat = n_feat
        self.filter_matches_threshold = filter_matches_threshold

    def filter_matches(self, scores: torch.Tensor):
        """obtain matches from a log assignment matrix [BxNxN]"""
        max0 = torch.topk(scores, k=1, dim=2, sorted=False)  # scores.max(2)
        max1 = torch.topk(scores, k=1, dim=1, sorted=False)  # scores.max(1)
        m0, m1 = max0.indices[:, :, 0], max1.indices[:, 0, :]

        indices = torch.arange(m0.shape[1], device=m0.device).expand_as(m0)
        mutual = indices == m1.gather(1, m0)
        mscores = max0.values[:, :, 0].exp()
        valid = mscores > self.filter_matches_threshold

        b_idx, m0_idx = torch.where(valid & mutual)
        m1_idx = m0[b_idx, m0_idx]
        matches = torch.concat([b_idx[:, None], m0_idx[:, None], m1_idx[:, None]], 1)
        mscores = mscores[b_idx, m0_idx]
        return matches, mscores

    def __del__(self):
        self.rknn.release()

    def trim(self, k, d):
        indices = torch.randperm(k.shape[1])[: self.n_feat]
        if k.shape[1] > self.n_feat:
            k = k[:, indices, :]
            d = d[:, indices, :]
        elif k.shape[1] < self.n_feat:
            k = F.pad(k, (0, 0, 0, self.n_feat - k.shape[1]), mode="constant", value=0)
            d = F.pad(d, (0, 0, 0, self.n_feat - d.shape[1]), mode="constant", value=0)
        return k, d, indices

    def match_feature(self, query_features, db_features, k_best):
        kpts1, desc1, _ = self.trim(
            query_features["keypoints"], query_features["descriptors"]
        )
        kpts2, desc2, _ = self.trim(
            db_features["keypoints"], db_features["descriptors"]
        )
        kpts1 = normalize_keypoints(kpts1, 600, 800)
        kpts2 = normalize_keypoints(kpts2, 600, 800)
        desc_union = torch.concatenate([desc1, desc2], axis=0)
        kpts_union = torch.concatenate([kpts1, kpts2], axis=0)

        scores = self.rknn.inference(
            inputs=[kpts_union.numpy(), desc_union.numpy()],
        )

        matches, scores = self.filter_matches(torch.tensor(scores[0]))
