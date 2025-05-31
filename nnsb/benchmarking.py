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
from itertools import repeat
from random import sample
from timeit import default_timer as timer
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from nnsb.dataset import Queries
from nnsb.feature_matchers.feature_matcher import FeatureMatcher
from nnsb.vpr_systems.vpr_system import VPRSystem

__all__ = [
    "benchmark_vpr_system",
    "benchmark_feature_matcher",
    "benchmark_feature_detector",
]


def benchmark_vpr_system(
    queries: Queries,
    vpr_system: VPRSystem,
) -> Dict[str, float]:
    """Benchmarks a VPR system.

    This function measures the throughput of a VPR system on a set of queries.

    Args:
        queries: Queries object containing the test images.
        vpr_system: VPR system to benchmark.

    Returns:
        Dict mapping metric names to values (currently just "throughput").
    """
    time_measurements = {}

    for image in [queries[0]] * 30:
        vpr_system.get_image_descriptor(image)
    start = timer()
    for image, _ in tqdm(queries, desc=" Q descriptors"):
        vpr_system.get_image_descriptor(image)
    time_measurements["throughput"] = len(queries) / (timer() - start)

    return time_measurements


def benchmark_feature_detector(
    queries: Queries, feature_detector
) -> Tuple[Dict[str, float], List[Any]]:
    """Benchmarks a feature detector.

    This function measures the throughput of a feature detector on a set of queries
    and returns the extracted features.

    Args:
        queries: Queries object containing the test images.
        feature_detector: Feature detector to benchmark.

    Returns:
        Tuple containing:
        - Dict mapping metric names to values (currently just "throughput").
        - List of extracted features for each query.
    """
    time_measurements = {}
    features = []

    start = timer()
    for query in [queries[0]] * 30:
        feature_detector(query)
    for query, _ in tqdm(queries, desc="Q features"):
        features.append(feature_detector(query))
    time_measurements["throughput"] = len(queries) / (timer() - start)

    return time_measurements, features


def benchmark_feature_matcher(
    data_local_features,
    query_local_features,
    feature_matcher: FeatureMatcher,
    k_closest: int = 10,
) -> Dict[str, float]:
    """Benchmarks a feature matcher.

    This function measures the throughput of a feature matcher on pairs of query
    and database features.

    Args:
        data_local_features: List of database features.
        query_local_features: List of query features.
        feature_matcher: Feature matcher to benchmark.
        k_closest: Number of database features to sample for each query (default: 10).

    Returns:
        Dict mapping metric names to values (currently just "throughput").
    """
    time_measurements = {}

    data_local_features = sample(list(data_local_features), k_closest)

    for query_features, db_features in [
        query_local_features[0],
        data_local_features[0],
    ] * 30:
        feature_matcher(query_features, db_features)
    start = timer()
    for query_features in tqdm(query_local_features, desc="Matching"):
        for db_features in data_local_features:
            feature_matcher(query_features, db_features)
    time_measurements["throughput"] = len(query_local_features) / (timer() - start)

    return time_measurements
