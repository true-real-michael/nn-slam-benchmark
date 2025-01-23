import torch
import numpy as np
from rknnlite.api import RKNNLite


class LightGlueRknn:
    def __init__(self, model_path: Path, n_feat: int):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    
    def postprocess(self, m0, mask, mscores):
        b_idx, m0_idx = torch.where(mask)
        m1_idx = m0[b_idx, m0_idx]
        matches = torch.concat([b_idx[:, None], m0_idx[:, None], m1_idx[:, None]], 1)
        mscores = mscores[b_idx, m0_idx]
        return matches, mscores
    
    def __del__(self):
        self.rknn.release()
    
    def trim(self, k, d):
        indices = torch.randperm(k.shape[1])[:self.n_feat]
        if k.shape[1] > self.n_feat:
            k = k[:, indices, :]
            d = d[:, indices, :]
        elif k.shape[1] < self.n:
            k = F.pad(k, (0, 0, 0, self.n_feat - k.shape[1]), mode='constant', value=0)
            d = F.pad(d, (0, 0, 0, self.n_feat - d.shape[1]), mode='constant', value=0)
        return k, d, indices

    def match_feature(self, query_features, db_features, k_best):
        kpts1, desc1, indices1 = self.trim(query_features['keypoints'], query_features['descriptors'])
        kpts2, desc2, indices2 = self.trim(db_features['keypoints'], db_features['descriptors'])
        kpts1 = normalize_keypoints(kpts1, 600, 800)
        kpts2 = normalize_keypoints(kpts2, 600, 800)
        desc_union = torch.concatenate([desc1, desc2], axis=0)[None, :, :, :].numpy()
        kpts_union = torch.concatenate([kpts1, kpts2], axis=0)[None, :, :, :].numpy()

        m0_rknn, mask_rknn, mscores_rknn = rknn.inference(inputs=[kpts_union, desc_union], data_format=['nchw', 'nchw'])
