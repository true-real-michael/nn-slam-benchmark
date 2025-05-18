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
import numpy as np
from tqdm import tqdm

from nnsb import Data
from nnsb.feature_matchers import FeatureMatcher
from nnsb.index_searchers import IndexSearcher
from nnsb.vpr_systems import VPRSystem


def create_index(
    dataset: Data,
    vpr_system: VPRSystem,
    index_searcher: IndexSearcher,
) -> None:
    """
    Create index for given dataset

    :param dataset: Data object
    :param vpr_system: VPR system
    :param index_searcher: Index searcher
    """
    global_descs = []
    for image in tqdm(dataset, desc="DB descriptors"):
        global_descs.append(vpr_system.get_image_descriptor(image))
    index_searcher.create(np.asarray(global_descs))


def create_local_features(
    dataset: Data,
    feature_matcher: FeatureMatcher,
) -> np.ndarray:
    """
    Create local features for given dataset

    :param dataset: Data object
    :param feature_matcher: Feature matcher

    :return: Local features
    """
    local_features = []
    for image in tqdm(dataset, desc="DB features"):
        features = feature_matcher.get_feature(image)
        # breakpoint()
        local_features.append(features)
    return local_features
