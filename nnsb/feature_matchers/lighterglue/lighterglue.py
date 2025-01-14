import numpy as np
import torch

from nnsb.feature_matchers.feature_matcher import FeatureMatcher


class LighterGlue(FeatureMatcher):
    def __init__(self, resize: int = 800, gpu_index: int = 0):
        super().__init__(resize, gpu_index)
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)

    def match_feature(self, query_features, db_features, k_best):
        num_matches = []
        matched_kpts_query = []
        matched_kpts_reference = []

        for db_index, db_feature in enumerate(db_features):
            keys = ["keypoints", "scores", "descriptors"]
            query_features = {
                k: (v.to(self.device) if k in keys else v)
                for k, v in query_features.items()
            }
            db_feature = {
                k: (v.to(self.device) if k in keys else v)
                for k, v in db_feature.items()
            }
            matches = self.xfeat.match_lighterglue(
                query_features, db_feature
            )[2]
            points_query = query_features["keypoints"][matches[..., 0]].cpu().numpy()
            points_db = db_feature["keypoints"][matches[..., 1]].cpu().numpy()
            num_matches.append(len(points_query))
            matched_kpts_query.append(points_query)
            matched_kpts_reference.append(points_db)

        num_matches = np.array(num_matches)
        res_indices = (-num_matches).argsort()[:k_best]

        matched_kpts_query = [matched_kpts_query[i] for i in res_indices]
        matched_kpts_reference = [matched_kpts_reference[i] for i in res_indices]
        return (
            res_indices,
            matched_kpts_query,
            matched_kpts_reference,
        )

    def get_feature(self, image: np.ndarray):
        output = self.xfeat.detectAndCompute(image, top_k=4096)[0]
        output.update({'image_size': (image.shape[1], image.shape[0])})
        return output
