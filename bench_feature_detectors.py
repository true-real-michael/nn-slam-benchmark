#  Copyright (c) 2024, Mikhail Kiselyov
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

import pickle
from pathlib import Path

from nnsb.benchmarking import benchmark_feature_detector
from nnsb.dataset import Data, Queries
from nnsb.feature_detectors.superpoint.superpoint import SuperPoint
from nnsb.feature_detectors.xfeat.xfeat import XFeat

LIMIT = 10
DATASET = "st_lucia"
BOARD = "nano"

data = Data(Path("datasets"), DATASET, limit=LIMIT, gt=False)
queries = Queries(Path("datasets"), DATASET, knn=None, limit=LIMIT)

feature_matchers = {
    f"superpoint_{resize}": [SuperPoint, [resize]]
    for resize in [800, 600, 400, 300, 200]
}

feature_matchers.update(
    {f"xfeat_{resize}": [XFeat, [resize]] for resize in [800, 600, 400, 300, 200]}
)

feature_matcher_measurements = {}

for feature_matcher_name, (method, args) in feature_matchers.items():
    print("Processing", feature_matcher_name)
    feature_matcher = method(*args)
    data_file_path = f"cache/data_features_{DATASET}_{feature_matcher_name}.pkl"
    queries_file_path = f"cache/queries_features_{DATASET}_{feature_matcher_name}.pkl"
    feature_matcher_measurements[feature_matcher_name], queries_features = (
        benchmark_feature_detector(queries, feature_matcher)
    )
    with open(queries_file_path, "wb") as f:
        pickle.dump(queries_features, f)
    del queries_features
    _, data_features = benchmark_feature_detector(data, feature_matcher)
    with open(data_file_path, "wb") as f:
        pickle.dump(data_features, f)
    del feature_matcher
    del data_features

output_file = Path(f"measurements/{BOARD}/{DATASET}_feature.pkl")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "wb") as f:
    pickle.dump(feature_matcher_measurements, f)
