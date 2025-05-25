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

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from nnsb.backend.rknn import RknnBackend
from nnsb.benchmarking import benchmark_vpr_system
from nnsb.dataset import Queries

try:
    from nnsb.backend.tensorrt import TensorRTBackend
except ImportError:
    pass
from nnsb.vpr_systems.cosplace.cosplace import CosPlace
from nnsb.vpr_systems.eigenplaces.eigenplaces import EigenPlaces
from nnsb.vpr_systems.mixvpr.mixvpr import MixVPR
from nnsb.vpr_systems.netvlad.netvlad import NetVLAD
from nnsb.vpr_systems.salad.salad import SALAD
from nnsb.vpr_systems.sela.sela import Sela

LIMIT = 1000


def run_single_benchmark(
    dataset,
    resize,
    system,
    quantized=False,
    backend=None,
    output_csv=None,
    board="orin25",
):
    """Run a single benchmark in a subprocess and return the results"""
    cmd = [
        sys.executable,
        __file__,
        "--dataset",
        dataset,
        "--resize",
        str(resize),
        "--system",
        system,
        "--board",
        board,
        "--output-csv",
        output_csv,
    ]
    if quantized:
        cmd.append("--quantized")
    if backend:
        cmd.append(f"--backend={backend}")

    print("calling", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"Error running benchmark for {system}_{resize}{'' if not quantized else '_q'}{f'_{backend}' if backend else ''}"
        )
        print(result.stderr)
        return None

    return {"status": "complete"}


def append_results_to_csv(
    csv_path,
    model_name,
    resize,
    metrics,
    quantized=False,
    backend=None,
    dataset="st_lucia",
    board="orin25",
):
    """Append benchmark results to a CSV file"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    model_id = f"{model_name}_{resize}"
    if quantized:
        model_id += "_q"
    if backend:
        model_id += f"_{backend}"

    row_data = {
        "model": model_name,
        "resize": resize,
        "quantized": str(quantized).lower(),
        "backend": backend if backend else "torch",
        "dataset": dataset,
        "board": board,
        "model_id": model_id,
    }

    if metrics:
        for key, value in metrics.items():
            row_data[key] = value

    with open(csv_path, "a", newline="") as f:
        fieldnames = [
            "model",
            "resize",
            "quantized",
            "backend",
            "dataset",
            "board",
            "model_id",
            "throughput",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

    print(f"Results for {model_id} appended to {csv_path}")


def run_single_system(dataset, resize, system, quantized, backend, output_csv, board):
    """Function to run a single benchmark directly"""
    queries = Queries(Path("datasets"), dataset, knn=None, limit=LIMIT, resize=resize)
    result = {}

    if backend == "rknn":
        rknn_dir = "weights/rknn/rknn_q" if quantized else "weights/rknn/rknn"
        rknn_path = Path(f"{rknn_dir}/{system}_{resize}.rknn")
        if not rknn_path.exists():
            print(f"Error: RKNN model not found at {rknn_path}")
            return {}

        if system == "netvlad":
            result = benchmark_vpr_system(
                queries, NetVLAD(backend=RknnBackend(rknn_path), resize=resize)
            )
        elif system == "eigenplaces":
            result = benchmark_vpr_system(
                queries, EigenPlaces(backend=RknnBackend(rknn_path), resize=resize)
            )
        elif system == "cosplace":
            result = benchmark_vpr_system(
                queries, CosPlace(backend=RknnBackend(rknn_path), resize=resize)
            )
        elif system == "salad":
            adjusted_resize = resize // 14 * 14
            result = benchmark_vpr_system(
                queries, SALAD(backend=RknnBackend(rknn_path), resize=adjusted_resize)
            )
        elif system == "mixvpr":
            result = benchmark_vpr_system(
                queries, MixVPR(backend=RknnBackend(rknn_path))
            )
        elif system == "sela":
            result = benchmark_vpr_system(queries, Sela(backend=RknnBackend(rknn_path)))
    elif quantized:
        trt_path = Path(f"weights/quant/{resize}/{system}.trt")
        if system == "netvlad":
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries, NetVLAD(backend=TensorRTBackend(trt_path), resize=resize)
                )
        elif system == "eigenplaces":
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries,
                    EigenPlaces(backend=TensorRTBackend(trt_path), resize=resize),
                )
        elif system == "cosplace":
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries, CosPlace(backend=TensorRTBackend(trt_path), resize=resize)
                )
        elif system == "salad":
            adjusted_resize = resize // 14 * 14
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries,
                    SALAD(backend=TensorRTBackend(trt_path), resize=adjusted_resize),
                )
        elif system == "mixvpr":
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries, MixVPR(backend=TensorRTBackend(trt_path))
                )
        elif system == "sela":
            if trt_path.exists():
                result = benchmark_vpr_system(
                    queries, Sela(backend=TensorRTBackend(trt_path))
                )
    else:
        if system == "netvlad":
            result = benchmark_vpr_system(
                queries,
                NetVLAD(
                    weights="weights/mapillary_WPCA4096.pth.tar",
                    resize=resize,
                ),
            )
        elif system == "eigenplaces":
            result = benchmark_vpr_system(queries, EigenPlaces(resize=resize))
        elif system == "cosplace":
            result = benchmark_vpr_system(queries, CosPlace(resize=resize))
        elif system == "salad":
            adjusted_resize = resize // 14 * 14
            result = benchmark_vpr_system(queries, SALAD(resize=adjusted_resize))
        elif system == "mixvpr":
            result = benchmark_vpr_system(
                queries,
                MixVPR(
                    ckpt_path="weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"
                ),
            )
        elif system == "sela":
            result = benchmark_vpr_system(
                queries,
                Sela(
                    dinov2_path="weights/dinov2_vitl14_pretrain.pth",
                    path_to_state_dict="weights/SelaVPR_msls.pth",
                ),
            )

    print(json.dumps(result))

    if output_csv and result:
        append_results_to_csv(
            output_csv,
            system,
            resize,
            result,
            quantized=quantized,
            backend=backend,
            dataset=dataset,
            board=board,
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run VPR benchmarks")
    parser.add_argument("--dataset", type=str, default="st_lucia", help="Dataset name")
    parser.add_argument("--board", type=str, default="orin25", help="Board name")
    parser.add_argument("--resize", type=int, help="Single resize value")
    parser.add_argument("--system", type=str, help="Single system name")
    parser.add_argument("--quantized", action="store_true", help="Run quantized model")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["tensorrt", "rknn"],
        help="Backend to use (tensorrt or rknn)",
    )
    parser.add_argument("--output-csv", type=str, help="CSV file to append results to")
    args = parser.parse_args()

    if not args.output_csv:
        backend_suffix = f"_{args.backend}" if args.backend else ""
        args.output_csv = f"measurements/{args.board}/{args.dataset}_vpr_{args.board}{backend_suffix}.csv"

    if args.resize is not None and args.system is not None:
        run_single_system(
            args.dataset,
            args.resize,
            args.system,
            args.quantized,
            args.backend,
            args.output_csv,
            args.board,
        )
        return

    systems = ["netvlad", "eigenplaces", "cosplace", "salad", "mixvpr", "sela"]
    resize_values = [800, 600, 400, 300, 200]
    mixvpr_resize = [320]
    sela_resize = [224]

    for system in systems:
        if system == "mixvpr":
            resizes_to_use = mixvpr_resize
        elif system == "sela":
            resizes_to_use = sela_resize
        else:
            resizes_to_use = resize_values

        for resize in resizes_to_use:
            run_single_benchmark(
                args.dataset,
                resize,
                system,
                output_csv=args.output_csv,
                board=args.board,
            )

            run_single_benchmark(
                args.dataset,
                resize,
                system,
                quantized=True,
                output_csv=args.output_csv,
                board=args.board,
            )

            if args.backend == "rknn" or args.backend is None:
                run_single_benchmark(
                    args.dataset,
                    resize,
                    system,
                    backend="rknn",
                    output_csv=args.output_csv,
                    board=args.board,
                )

                run_single_benchmark(
                    args.dataset,
                    resize,
                    system,
                    quantized=True,
                    backend="rknn",
                    output_csv=args.output_csv,
                    board=args.board,
                )

    print(f"All benchmark results written to {args.output_csv}")


if __name__ == "__main__":
    main()
