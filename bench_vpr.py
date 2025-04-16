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

import json
from collections import defaultdict
from pathlib import Path

from nnsb.benchmarking import benchmark_vpr_system
from nnsb.dataset import Queries
from nnsb.backend.torchscript import TorchscriptBackend
from nnsb.backend.tensorrt import TensorRTBackend
from nnsb.vpr_systems.cosplace.cosplace import CosPlace
from nnsb.vpr_systems.eigenplaces.eigenplaces import EigenPlaces
from nnsb.vpr_systems.netvlad.netvlad import NetVLAD
from nnsb.vpr_systems.salad.salad import SALAD
from nnsb.vpr_systems.mixvpr.mixvpr import MixVPR
from nnsb.vpr_systems.sela.sela import Sela

LIMIT = None
DATASET = "st_lucia"
BOARD = "mixvpr"
RESIZE = 800

queries = Queries(Path("datasets"), DATASET, knn=None, limit=LIMIT)

measurements = defaultdict(dict)


# for resize in [800, 600, 400, 300, 200]:
#     trt_path = Path(f"weights/quant/{resize}/netvlad.trt")
#     if trt_path.exists():
#         measurements[f'netvlad_{resize}_q'] = benchmark_vpr_system(
#             queries,
#             NetVLAD(backend=TensorRTBackend(trt_path), resize=resize)
#         )
#     measurements[f'netvlad_{resize}'] = benchmark_vpr_system(
#         queries,
#         NetVLAD(
#             weights="weights/mapillary_WPCA4096.pth.tar",
#             resize=resize,
#         )
#     )

# for resize in [800, 600, 400, 300, 200]:
#     trt_path = Path(f"weights/quant/{resize}/eigenplaces.trt")
#     if trt_path.exists():
#         measurements[f'eigenplaces_{resize}_q'] = benchmark_vpr_system(
#             queries,
#             EigenPlaces(backend=TensorRTBackend(trt_path), resize=resize)
#         )
#     measurements[f'eigenplaces_{resize}'] = benchmark_vpr_system(
#         queries,
#         EigenPlaces(
#             resize=resize,
#         )
#     )

# for resize in [800, 600, 400, 300, 200]:
for resize in [200]:
    trt_path = Path(f"weights/quant/{resize}/cosplace.trt")
    if trt_path.exists():
        measurements[f'cosplace_{resize}_q'] = benchmark_vpr_system(
            queries,
            CosPlace(backend=TensorRTBackend(trt_path), resize=resize)
        )
    measurements[f'cosplace_{resize}'] = benchmark_vpr_system(
        queries,
        CosPlace(
            resize=resize,
        )
    )

# for resize in [800, 600, 400, 300, 200]:
#     trt_path = Path(f"weights/quant/{resize}/salad.trt")
#     if trt_path.exists():
#         measurements[f'salad_{resize}_q'] = benchmark_vpr_system(
#             queries,
#             SALAD(backend=TensorRTBackend(trt_path), resize=resize // 14 * 14)
#         )
#     measurements[f'salad_{resize}'] = benchmark_vpr_system(
#         queries,
#         SALAD(
#             resize=resize // 14 * 14,
#         )
#     )

# trt_path = Path("weights/quant/300/mixvpr.trt")
# if trt_path.exists():
#     measurements['mixvpr_300_q'] = benchmark_vpr_system(
#         queries,
#         MixVPR(backend=TensorRTBackend(trt_path))
#     )
# measurements[f'mixvpr_300'] = benchmark_vpr_system(
#     queries,
#     MixVPR(
#         ckpt_path="weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"
#     )
# )

# measurements[f'sela_200'] = benchmark_vpr_system(
#     queries,
#     Sela(
#         dinov2_path="weights/dinov2_vitl14_pretrain.pth",
#         path_to_state_dict="weights/SelaVPR_msls.pth",
#     )
# )
# trt_path = Path("weights/quant/200/sela.trt")
# if trt_path.exists():
#     measurements['sela_200_q'] = benchmark_vpr_system(
#         queries,
#         Sela(backend=TensorRTBackend(trt_path))
#     )


# weights/SelaVPR_msls.pth
# weights/dinov2_vitl14_pretrain.pth

output_file = Path(f"measurements/orin_quant/{DATASET}_vpr_{BOARD}.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(measurements, f)

print(json.dumps(measurements, indent=2))
