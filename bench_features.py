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

import numpy as np
import nnsb
import pickle

from pathlib import Path
from nnsb.benchmarking import benchmark_feature_matcher, benchmark_vpr_system, create_index, create_local_features
from nnsb.dataset import Data, Queries
from nnsb.retrieval_system import RetrievalSystem

LIMIT = None
DATASET = "satellite"
BOARD = "orin3"

test_ds = Data(Path("datasets"), DATASET, limit=LIMIT, gt=False)
queries = Queries(
    Path("datasets"),
    DATASET,
    knn=None,
    limit=LIMIT
)

feature_matchers = {
    # 'lightglue': [nnsb.LightGlue, [800]],
    # 'superglue': [nnsb.SuperGlue, ['weights/superglue_outdoor.pth']],
    # 'lighterglue': [nnsb.LighterGlue, [800]],
    'sela': [nnsb.SelaLocal, ['weights/SelaVPR_msls.pth', 'weights/dinov2_vitl14_pretrain.pth']],
}

feature_matcher_measurements = {}

for feature_matcher_name, (method, args) in feature_matchers.items():
    print('Processing', feature_matcher_name)
    feature_matcher = method(*args)
    file_path = f'cache/local_features_{DATASET}_{feature_matcher_name}.pkl'
    if Path(file_path).exists():
        with open(file_path, 'rb') as f:
            local_features = pickle.load(f)
    else:
        local_features = create_local_features(test_ds, feature_matcher)
        local_features = np.asarray(local_features)
        with open(file_path, 'wb') as f:
            pickle.dump(local_features, f)
    feature_matcher_measurements[feature_matcher_name] = benchmark_feature_matcher(queries, feature_matcher, local_features, 10)
    del feature_matcher

output_file = Path(f'measurements/{BOARD}/{DATASET}_feature.pkl')
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'wb') as f:
    pickle.dump(feature_matcher_measurements, f)
