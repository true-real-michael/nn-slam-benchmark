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

import json
import pickle
from itertools import repeat
from pathlib import Path

from nnsb.benchmarking import benchmark_feature_matcher
from nnsb.feature_matchers.lighterglue.lighterglue import LighterGlue
from nnsb.feature_matchers.lightglue.lightglue import LightGlue
from nnsb.feature_matchers.superglue.superglue import SuperGlue

LIMIT = 1000
# LIMIT = 4
DATASET = "st_lucia"
BOARD = "orin25"

superpoint_feature_matchers = {
    "lightglue": [LightGlue, []],
    "superglue": [SuperGlue, ["weights/superglue_outdoor.pth"]],
}
xfeat_feature_matchers = {
    "lighterglue": [LighterGlue, []],
}

feature_matcher_measurements = {}

for resize in [200, 300, 400, 600, 800]:
    superpoint_data_file_path = f"cache/data_features_{DATASET}_superpoint_{resize}.pkl"
    superpoint_queries_file_path = f"cache/queries_features_{DATASET}_superpoint_{resize}.pkl"
    xfeat_data_file_path = f"cache/data_features_{DATASET}_xfeat_{resize}.pkl"
    xfeat_queries_file_path = f"cache/queries_features_{DATASET}_xfeat_{resize}.pkl"

    for feature_matcher_name, (method, args) in superpoint_feature_matchers.items():
        print("Processing", feature_matcher_name)
        feature_matcher = method(*args)
        with open(superpoint_queries_file_path, "rb") as f:
            queries_local_features = list(repeat(pickle.load(f)[0], LIMIT))
        with open(superpoint_data_file_path, "rb") as f:
            data_local_features = list(repeat(pickle.load(f)[0], LIMIT))
        feature_matcher_measurements[f"{feature_matcher_name}_{resize}"] = (
            benchmark_feature_matcher(
                data_local_features, queries_local_features, feature_matcher, 1
            )
        )
        del feature_matcher

    for feature_matcher_name, (method, args) in xfeat_feature_matchers.items():
        print("Processing", feature_matcher_name)
        feature_matcher = method(*args)
        with open(xfeat_queries_file_path, "rb") as f:
            queries_local_features = list(repeat(pickle.load(f)[0], LIMIT))
        with open(xfeat_data_file_path, "rb") as f:
            data_local_features = list(repeat(pickle.load(f)[0], LIMIT))
        feature_matcher_measurements[f"{feature_matcher_name}_{resize}"] = (
            benchmark_feature_matcher(
                data_local_features, queries_local_features, feature_matcher, 1
            )
        )
        del feature_matcher

output_file = Path(f"measurements/{BOARD}/{DATASET}_feature_matcher.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(feature_matcher_measurements, f)

