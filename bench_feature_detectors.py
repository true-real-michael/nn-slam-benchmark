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

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

from nnsb.benchmarking import benchmark_feature_detector
from nnsb.dataset import Data, Queries
from nnsb.feature_detectors.superpoint.superpoint import SuperPoint
from nnsb.feature_detectors.xfeat.xfeat import XFeat

LIMIT = 1000


def run_single_benchmark(dataset, resize, detector, output_csv=None, board="orin25"):
    """Run a single benchmark in a subprocess and return the results"""
    cmd = [
        sys.executable,
        __file__,
        "--dataset",
        dataset,
        "--resize",
        str(resize),
        "--detector",
        detector,
        "--board",
        board,
    ]

    if output_csv:
        cmd.extend(["--output-csv", output_csv])

    print("calling", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running benchmark for {detector}_{resize}")
        print(result.stderr)
        return None

    return {"status": "complete"}


def append_results_to_csv(
    csv_path, detector_name, resize, metrics, dataset="st_lucia", board="orin25"
):
    """Append benchmark results to a CSV file"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    detector_id = f"{detector_name}_{resize}"

    row_data = {
        "detector": detector_name,
        "resize": resize,
        "dataset": dataset,
        "board": board,
        "detector_id": detector_id,
    }

    if metrics:
        for key, value in metrics.items():
            row_data[key] = value

    with open(csv_path, "a", newline="") as f:
        fieldnames = [
            "detector",
            "throughput" "resize",
            "dataset",
            "board",
            "detector_id",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

    print(f"Results for {detector_id} appended to {csv_path}")


def run_single_detector(
    dataset, resize, detector, output_csv, board, save_features=True
):
    """Function to run a single benchmark directly"""
    data = Data(Path("datasets"), dataset, limit=LIMIT, resize=resize, superpoint=True)
    queries = Queries(
        Path("datasets"), dataset, knn=None, limit=LIMIT, resize=resize, superpoint=True
    )

    feature_matcher = None
    if detector == "superpoint":
        feature_matcher = SuperPoint(resize)
    elif detector == "xfeat":
        feature_matcher = XFeat(resize)
    else:
        print(f"Unknown feature detector: {detector}")
        return {}

    print(f"Processing {detector}_{resize}")

    data_file_path = f"cache/data_features_{dataset}_{detector}_{resize}.pkl"
    queries_file_path = f"cache/queries_features_{dataset}_{detector}_{resize}.pkl"

    # Benchmark query features
    measurements, queries_features = benchmark_feature_detector(
        queries, feature_matcher
    )

    if save_features:
        Path("cache").mkdir(exist_ok=True)
        with open(queries_file_path, "wb") as f:
            pickle.dump(queries_features, f)

    # Benchmark data features
    _, data_features = benchmark_feature_detector(data, feature_matcher)

    if save_features:
        with open(data_file_path, "wb") as f:
            pickle.dump(data_features, f)

    print(json.dumps(measurements))

    if output_csv:
        append_results_to_csv(
            output_csv,
            detector,
            resize,
            measurements,
            dataset=dataset,
            board=board,
        )

    # Save to JSON file for compatibility with the original script
    output_file = Path(f"measurements/{board}/{dataset}_feature.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    feature_matcher_measurements = {}
    feature_matcher_measurements[f"{detector}_{resize}"] = measurements

    with open(output_file, "w") as f:
        json.dump(feature_matcher_measurements, f)

    return measurements


def main():
    parser = argparse.ArgumentParser(description="Run Feature Detector benchmarks")
    parser.add_argument("--dataset", type=str, default="st_lucia", help="Dataset name")
    parser.add_argument("--board", type=str, default="orin25", help="Board name")
    parser.add_argument("--resize", type=int, help="Single resize value")
    parser.add_argument(
        "--detector", type=str, help="Feature detector name (superpoint, xfeat)"
    )
    parser.add_argument("--output-csv", type=str, help="CSV file to append results to")
    parser.add_argument(
        "--no-save-features", action="store_true", help="Don't save feature cache files"
    )
    args = parser.parse_args()

    if not args.output_csv:
        args.output_csv = (
            f"measurements/{args.board}/{args.dataset}_feature_{args.board}.csv"
        )

    # If specific detector and resize are provided, run a single benchmark
    if args.resize is not None and args.detector is not None:
        run_single_detector(
            args.dataset,
            args.resize,
            args.detector,
            args.output_csv,
            args.board,
            not args.no_save_features,
        )
        return

    # Otherwise, run all benchmarks
    detectors = ["superpoint", "xfeat"]
    resize_values = [800, 600, 400, 300, 200]

    for detector in detectors:
        for resize in resize_values:
            run_single_benchmark(
                args.dataset,
                resize,
                detector,
                output_csv=args.output_csv,
                board=args.board,
            )

    print(f"All benchmark results written to {args.output_csv}")


if __name__ == "__main__":
    main()
