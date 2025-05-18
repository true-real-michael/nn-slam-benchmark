from random import sample
from timeit import default_timer as timer
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from nnsb.dataset import Queries
from nnsb.feature_matchers.feature_matcher import FeatureMatcher
from nnsb.vpr_systems.vpr_system import VPRSystem


def benchmark_vpr_system(
    queries: Queries,
    vpr_system: VPRSystem,
) -> Dict[str, float]:
    """
    Benchmark VPR systems with given dataset and queries

    :param queries: Queries object
    :param vpr_systems: VPR system

    :return: Dictionary with time measurements
    """
    time_measurements = {}

    start = timer()
    for image, _ in tqdm(queries, desc=" Q descriptors"):
        vpr_system.get_image_descriptor(image)
    time_measurements["global_descs"] = len(queries) / (timer() - start)

    return time_measurements


def benchmark_feature_detector(
    queries: Queries, feature_detector
) -> Tuple[Dict[str, float], List[Any]]:
    time_measurements = {}
    features = []

    start = timer()
    for query, _ in tqdm(queries, desc=" Q features"):
        features.append(feature_detector(query))
    time_measurements["feature_extraction"] = len(queries) / (timer() - start)

    return time_measurements, features


def benchmark_feature_matcher(
    data_local_features,
    query_local_features,
    feature_matcher: FeatureMatcher,
    k_closest: int = 10,
) -> Dict[str, float]:
    """
    Benchmark feature matchers with given dataset and queries

    :param queries: Queries object
    :param feature_matchers: List of feature matchers
    :param k_closest: Number of closest images to return

    :return: Dictionary with time measurements
    """
    time_measurements = {}

    data_local_features = sample(list(data_local_features), k_closest)

    start = timer()
    for query_features in tqdm(query_local_features, desc="Matching"):
        for db_features in data_local_features:
            feature_matcher.match_feature(query_features, [db_features], k_closest)
    time_measurements["feature_matching"] = len(query_local_features) / (
        timer() - start
    )

    return time_measurements
