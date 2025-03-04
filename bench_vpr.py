#  Copyright (c) 2023, Mikhail Kiselyov
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
from collections import defaultdict
from pathlib import Path

from nnsb.benchmarking import benchmark_vpr_system
from nnsb.dataset import Queries
from nnsb.backend.torchscript import TorchscriptBackend
from nnsb.vpr_systems.cosplace.cosplace import CosPlace
from nnsb.vpr_systems.eigenplaces.eigenplaces import EigenPlaces
from nnsb.vpr_systems.netvlad.netvlad import NetVLAD
from nnsb.vpr_systems.mixvpr.mixvpr_wrapper import MixVPR

LIMIT = None
DATASET = "st_lucia"
BOARD = "test"
RESIZE = 800

queries = Queries(Path("datasets"), DATASET, knn=None, limit=LIMIT)

vpr_systems = {}

# for resize in [800, 600, 400, 300, 200]:
#     vpr_systems.update(
#         {
#             f"netvlad_{resize}": [
#                 NetVLAD,
#                 ["weights/mapillary_WPCA4096.pth.tar"],
#                 {"resize": resize, "use_faiss": True},
#             ],
#         }
#     )
# for resize in [800, 600, 400, 300, 200]:
#     vpr_systems.update(
#         {
#             f"cosplace_{resize}": [CosPlace, [], {"resize": resize}],
#         }
#     )
# for resize in [800, 600, 400, 300, 200]:
#     vpr_systems.update(
#         {
#             f"eigenplaces_{resize}": [EigenPlaces, [], {"resize": resize}],
#         }
#     )

vpr_systems.update(
    {
        "mixvpr": [
            MixVPR,
            ["weights/torchscript/300/mixvpr.pt", TorchscriptBackend],
            {},
        ],
    }
)


measurements = defaultdict(dict)

for vpr_system_name, (method, args, kwargs) in vpr_systems.items():
    print("Processing", vpr_system_name)
    vpr_system = method(*args, **kwargs)
    measurements[vpr_system_name] = benchmark_vpr_system(queries, vpr_system)
    del vpr_system

output_file = Path(f"measurements/{BOARD}/{DATASET}_vpr.pkl")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "wb") as f:
    pickle.dump(measurements, f)
