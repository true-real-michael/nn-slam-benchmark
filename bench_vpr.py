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
import sys
import subprocess
import argparse
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

LIMIT = 10

def run_single_benchmark(dataset, resize, system, quantized=False):
    """Run a single benchmark in a subprocess and return the results"""
    cmd = [
        sys.executable, 
        __file__, 
        "--dataset", dataset, 
        "--resize", str(resize), 
        "--system", system
    ]
    if quantized:
        cmd.append("--quantized")
    
    print("calling", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running benchmark for {system}_{resize}{'' if not quantized else '_q'}")
        print(result.stderr)
        return None
    
    try:
        last_line = result.stdout.splitlines()[-1]
        return json.loads(last_line)
    except json.JSONDecodeError:
        breakpoint()
        print(last_line)
        print(f"Error parsing benchmark results for {system}_{resize}{'' if not quantized else '_q'}")
        return None

def run_single_system(dataset, resize, system, quantized):
    """Function to run a single benchmark directly"""
    queries = Queries(Path("datasets"), dataset, knn=None, limit=LIMIT)
    result = {}
    
    if system == "netvlad":
        if quantized:
            trt_path = Path(f"weights/quant/{resize}/netvlad.trt")
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries,
                    NetVLAD(backend=TensorRTBackend(trt_path), resize=resize)
                )
        else:
            result = benchmark_vpr_system(
                queries,
                NetVLAD(
                    weights="weights/mapillary_WPCA4096.pth.tar",
                    resize=resize,
                )
            )
    elif system == "eigenplaces":
        if quantized:
            trt_path = Path(f"weights/quant/{resize}/eigenplaces.trt")
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries,
                    EigenPlaces(backend=TensorRTBackend(trt_path), resize=resize)
                )
        else:
            result = benchmark_vpr_system(
                queries,
                EigenPlaces(resize=resize)
            )
    elif system == "cosplace":
        if quantized:
            trt_path = Path(f"weights/quant/{resize}/cosplace.trt")
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries,
                    CosPlace(backend=TensorRTBackend(trt_path), resize=resize)
                )
        else:
            result = benchmark_vpr_system(
                queries,
                CosPlace(resize=resize)
            )
    elif system == "salad":
        adjusted_resize = resize // 14 * 14  # Ensure divisible by 14
        if quantized:
            trt_path = Path(f"weights/quant/{resize}/salad.trt")
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries,
                    SALAD(backend=TensorRTBackend(trt_path), resize=adjusted_resize)
                )
        else:
            result = benchmark_vpr_system(
                queries,
                SALAD(resize=adjusted_resize)
            )
    elif system == "mixvpr":
        if quantized:
            trt_path = Path(f"weights/quant/{resize}/mixvpr.trt")
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries,
                    MixVPR(backend=TensorRTBackend(trt_path))
                )
        else:
            result = benchmark_vpr_system(
                queries,
                MixVPR(ckpt_path="weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt")
            )
    elif system == "sela":
        if quantized:
            trt_path = Path(f"weights/quant/{resize}/sela.trt")
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries,
                    Sela(backend=TensorRTBackend(trt_path))
                )
        else:
            result = benchmark_vpr_system(
                queries,
                Sela(
                    dinov2_path="weights/dinov2_vitl14_pretrain.pth",
                    path_to_state_dict="weights/SelaVPR_msls.pth",
                )
            )

    print(json.dumps(result))
    return result

def main():
    parser = argparse.ArgumentParser(description="Run VPR benchmarks")
    parser.add_argument("--dataset", type=str, default="st_lucia", help="Dataset name")
    parser.add_argument("--board", type=str, default="orin25", help="Board name")
    parser.add_argument("--resize", type=int, help="Single resize value")
    parser.add_argument("--system", type=str, help="Single system name")
    parser.add_argument("--quantized", action="store_true", help="Run quantized model")
    args = parser.parse_args()

    # If both resize and system are specified, run a single benchmark directly
    if args.resize is not None and args.system is not None:
        run_single_system(args.dataset, args.resize, args.system, args.quantized)
        return

    # Otherwise, run all benchmarks in subprocesses
    measurements = defaultdict(dict)
    systems = ["netvlad", "eigenplaces", "cosplace", "salad", "mixvpr", "sela"]
    resize_values = [800, 600, 400, 300, 200]
    
    # Special case for MixVPR (only resize 300)
    mixvpr_resize = [300]
    # Special case for Sela (only resize 200)
    sela_resize = [200]

    for system in systems:
        if system == "mixvpr":
            resizes_to_use = mixvpr_resize
        elif system == "sela":
            resizes_to_use = sela_resize
        else:
            resizes_to_use = resize_values
            
        for resize in resizes_to_use:
            # Run non-quantized version
            result = run_single_benchmark(args.dataset, resize, system)
            if result:
                measurements[f'{system}_{resize}'] = result
            
            # Run quantized version
            result = run_single_benchmark(args.dataset, resize, system, quantized=True)
            if result:
                measurements[f'{system}_{resize}_q'] = result

    output_file = Path(f"measurements/orin_quant/{args.dataset}_vpr_{args.board}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(measurements, f)

    print(json.dumps(measurements, indent=2))

if __name__ == "__main__":
    main()
