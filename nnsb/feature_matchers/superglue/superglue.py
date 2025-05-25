# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#                       Ivan Moskalenko
#                       Anastasiia Kornilova
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from nnsb.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.feature_matchers.feature_matcher import FeatureMatcher
from nnsb.feature_matchers.superglue.model.superglue_matcher import SuperGlueMatcher


class SuperGlueTorchBackend(TorchBackend):
    """TorchBackend implementation for SuperGlue model.

    This backend initializes and manages a SuperGlue model for feature matching.

    Attributes:
        model: The SuperGlue model.
    """

    def __init__(self, path_to_sg_weights):
        """Initializes the SuperGlue TorchBackend.

        Args:
            path_to_sg_weights: Path to the SuperGlue weights.
        """
        super().__init__(SuperGlueMatcher(path_to_sg_weights))


class SuperGlue(FeatureMatcher):
    """SuperGlue feature matcher implementation.

    Implementation of [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
    matcher with SuperPoint extractor.
    """

    def __init__(
        self,
        path_to_sg_weights: Optional[Path] = None,
        backend: Optional[Backend] = None,
    ):
        """Initializes the SuperGlue feature matcher.

        Args:
            path_to_sg_weights: Path to SuperGlue weights.
            backend: Optional backend instance. If None, creates a SuperGlueTorchBackend.
        """
        super().__init__(backend or SuperGlueTorchBackend(path_to_sg_weights))

    def __call__(self, query_feat, db_feat):
        """Process features with the SuperGlue matcher.

        Args:
            query_feat: Dictionary containing query features.
            db_feat: Dictionary containing database features.

        Returns:
            Tuple containing number of matches, matched query points, and matched database points.
        """
        keys = ["keypoints", "scores", "descriptors"]
        pred = {"shape0": query_feat["image_size"], "shape1": db_feat["image_size"]}
        pred |= {k + "0": query_feat[k].to(self.device) for k in keys}
        pred |= {k + "1": db_feat[k].to(self.device) for k in keys}
        pred = self.backend(pred)
        return self.postprocess(query_feat, db_feat, pred)

    def postprocess(self, query_feat, db_feat, matches):
        """Postprocesses SuperGlue model outputs.

        Args:
            query_feat: Query features.
            db_feat: Database features.
            matches: Matched indices from the model.

        Returns:
            Tuple containing number of matches, matched query points, and matched database points.
        """
        matches = matches["matches0"][0].cpu().numpy()
        valid = matches > -1
        matched_kpts_query = query_feat["keypoints"][0][valid]
        matched_kpts_reference = db_feat["keypoints"][0][matches[valid]]
        num_matches = np.sum(valid)

        return (
            num_matches,
            matched_kpts_query,
            matched_kpts_reference,
        )

    def get_sample_input(self):
        """Provides a sample input for the model.

        Returns:
            Dict with sample features.
        """
        features = {
            "keypoints0": torch.randint(0, 200, (1, 74, 2)),
            "keypoints1": torch.randint(0, 200, (1, 81, 2)),
            "scores0": torch.rand(1, 74),
            "scores1": torch.rand(1, 81),
            "descriptors0": torch.rand(1, 256, 74),
            "descriptors1": torch.rand(1, 256, 81),
            "image_size0": torch.tensor((200, 200)),
        }
        return {k: v.to(self.device) for k, v in features.items()}
