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
from itertools import repeat
from pathlib import Path

from nnsb.benchmarking import benchmark_feature_matcher
from nnsb.feature_matchers.lighterglue.lighterglue import LighterGlue
from nnsb.feature_matchers.lightglue.lightglue import LightGlue
from nnsb.feature_matchers.superglue.superglue import SuperGlue

LIMIT = 100
DATASET = "st_lucia"
BOARD = "nano"
RESIZE = 800

feature_matchers = {
    # "lightglue": [LightGlue, []],
    "lighterglue": [LighterGlue, []],
    # "superglue": [SuperGlue, ["weights/superglue_outdoor.pth"]],
}


feature_matcher_measurements = {}

for feature_matcher_name, (method, args) in feature_matchers.items():
    print("Processing", feature_matcher_name)
    feature_matcher = method(*args)
    data_file_path = f"cache/data_features_{DATASET}_xfeat_{RESIZE}.pkl"
    queries_file_path = f"cache/queries_features_{DATASET}_xfeat_{RESIZE}.pkl"
    with open(queries_file_path, "rb") as f:
        queries_local_features = repeat(pickle.load(f)[0], LIMIT)
    with open(data_file_path, "rb") as f:
        data_local_features = repeat(pickle.load(f)[0], LIMIT)
    feature_matcher_measurements[(feature_matcher_name, RESIZE)] = (
        benchmark_feature_matcher(
            data_local_features, queries_local_features, feature_matcher, 1
        )
    )
    del feature_matcher

output_file = Path(f"measurements/{BOARD}/{DATASET}_lighterglue_feature_{RESIZE}.pkl")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "wb") as f:
    pickle.dump(feature_matcher_measurements, f)
