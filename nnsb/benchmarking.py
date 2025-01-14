from timeit import default_timer as timer
from typing import Dict, List
from random import sample

import numpy as np
from tqdm import tqdm
from nnsb.dataset import Data, Queries
from nnsb.feature_matchers.feature_matcher import FeatureMatcher
from nnsb.index_searchers.index_searcher import IndexSearcher
from nnsb.vpr_systems.vpr_system import VPRSystem

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
    for image in tqdm(
        dataset, desc="DB descriptors"
    ):
        global_descs.append(vpr_system.get_image_descriptor(image))
    index_searcher.create(np.asarray(global_descs))

def benchmark_vpr_system(
        queries: Queries,
        vpr_system: VPRSystem,
        index: IndexSearcher,
        k_closest: int,
) -> Dict[str, float]:
    """
    Benchmark VPR systems with given dataset and queries

    :param queries: Queries object
    :param vpr_systems: List of VPR systems
    :param index: Index searcher
    :param k_closest: Number of closest images to return

    :return: Dictionary with time measurements
    """
    time_measurements = {}

    start = timer()
    for image in tqdm(queries, desc=" Q descriptors"):
        vpr_system.get_image_descriptor(image)
    time_measurements["global_descs"] = timer() - start

    return time_measurements


def create_local_features(
        dataset: Data,
        feature_matcher,
) -> np.ndarray:
    """
    Create local features for given dataset

    :param dataset: Data object
    :param feature_matcher: Feature matcher

    :return: Local features
    """
    local_features = []
    for i, image in enumerate(
        tqdm(dataset, desc="DB features")
    ):
        local_features.append(feature_matcher.get_feature(image))
    return np.asarray(local_features)


def benchmark_feature_detector(queries: Queries, feature_detector) -> Dict[str, float]:
    time_measurements = {}

    start = timer()
    for query in tqdm(queries, desc=" Q features"):
        feature_detector.get_feature(query)
    time_measurements["feature_extraction"] = timer() - start

    return time_measurements

def benchmark_feature_matcher(
        queries: Queries,
        feature_matcher: FeatureMatcher,
        local_features: np.ndarray,
        k_closest: int | None,
) -> Dict[str, float]:
    """
    Benchmark feature matchers with given dataset and queries

    :param queries: Queries object
    :param feature_matchers: List of feature matchers
    :param k_closest: Number of closest images to return

    :return: Dictionary with time measurements
    """
    time_measurements = {}

    start = timer()
    query_local_features = [feature_matcher.get_feature(query) for query in tqdm(queries, desc=" Q features")]
    time_measurements["feature_extraction"] = timer() - start

    start = timer()
    for query_features in tqdm(query_local_features, desc="Matching"):
        local_features = sample(list(local_features), k_closest)
        feature_matcher.match_feature(query_features, local_features, k_closest)
    time_measurements["feature_matching"] = timer() - start

    return time_measurements

