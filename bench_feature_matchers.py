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
from itertools import repeat
from pathlib import Path

from nnsb.benchmarking import benchmark_feature_matcher
from nnsb.feature_matchers.lighterglue.lighterglue import LighterGlue
from nnsb.feature_matchers.lightglue.lightglue import LightGlue
from nnsb.feature_matchers.superglue.superglue import SuperGlue

LIMIT = 10


def run_single_benchmark(
    dataset, resize, matcher, detector="superpoint", output_csv=None, board="orin25"
):
    """Run a single benchmark in a subprocess and return the results"""
    cmd = [
        sys.executable,
        __file__,
        "--dataset",
        dataset,
        "--resize",
        str(resize),
        "--matcher",
        matcher,
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
        print(f"Error running benchmark for {matcher}_{resize} with {detector}")
        print(result.stderr)
        return None

    return {"status": "complete"}


def append_results_to_csv(
    csv_path,
    matcher_name,
    resize,
    metrics,
    detector="superpoint",
    dataset="st_lucia",
    board="orin25",
):
    """Append benchmark results to a CSV file"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    matcher_id = f"{matcher_name}_{resize}"

    row_data = {
        "matcher": matcher_name,
        "resize": resize,
        "detector": detector,
        "dataset": dataset,
        "board": board,
        "matcher_id": matcher_id,
    }

    if metrics:
        for key, value in metrics.items():
            row_data[key] = value

    with open(csv_path, "a", newline="") as f:
        fieldnames = [
            "matcher",
            "resize",
            "detector",
            "dataset",
            "board",
            "matcher_id",
            "throughput",
            "num_matches",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

    print(f"Results for {matcher_id} with {detector} appended to {csv_path}")


def run_single_matcher(dataset, resize, matcher, detector, output_csv, board):
    """Function to run a single benchmark directly"""
    feature_matcher = None

    # Initialize the correct matcher
    if matcher == "lightglue":
        feature_matcher = LightGlue()
    elif matcher == "superglue":
        feature_matcher = SuperGlue("weights/superglue_outdoor.pth")
    elif matcher == "lighterglue":
        feature_matcher = LighterGlue()
    else:
        print(f"Unknown feature matcher: {matcher}")
        return {}

    print(f"Processing {matcher}_{resize} with {detector}")

    # Load the local features
    data_file_path = f"cache/data_features_{dataset}_{detector}_{resize}.pkl"
    queries_file_path = f"cache/queries_features_{dataset}_{detector}_{resize}.pkl"

    try:
        with open(queries_file_path, "rb") as f:
            queries_local_features = list(repeat(pickle.load(f)[0], LIMIT))
        with open(data_file_path, "rb") as f:
            data_local_features = list(repeat(pickle.load(f)[0], LIMIT))
    except FileNotFoundError as e:
        print(f"Error: Could not find feature file: {e}")
        return {}

    # Run the benchmark
    measurements = benchmark_feature_matcher(
        data_local_features, queries_local_features, feature_matcher, 1
    )

    print(json.dumps(measurements))

    if output_csv:
        append_results_to_csv(
            output_csv,
            matcher,
            resize,
            measurements,
            detector=detector,
            dataset=dataset,
            board=board,
        )

    # Save to JSON file for compatibility with the original script
    output_file = Path(f"measurements/{board}/{dataset}_feature_matcher.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    feature_matcher_measurements = {}
    feature_matcher_measurements[f"{matcher}_{resize}"] = measurements

    with open(output_file, "w") as f:
        json.dump(feature_matcher_measurements, f)

    return measurements


def main():
    parser = argparse.ArgumentParser(description="Run Feature Matcher benchmarks")
    parser.add_argument("--dataset", type=str, default="st_lucia", help="Dataset name")
    parser.add_argument("--board", type=str, default="orin25", help="Board name")
    parser.add_argument("--resize", type=int, help="Single resize value")
    parser.add_argument(
        "--matcher",
        type=str,
        help="Feature matcher name (lightglue, superglue, lighterglue)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="superpoint",
        choices=["superpoint", "xfeat"],
        help="Feature detector used",
    )
    parser.add_argument("--output-csv", type=str, help="CSV file to append results to")
    args = parser.parse_args()

    if not args.output_csv:
        args.output_csv = (
            f"measurements/{args.board}/{args.dataset}_feature_matcher_{args.board}.csv"
        )

    # If specific matcher and resize are provided, run a single benchmark
    if args.resize is not None and args.matcher is not None:
        run_single_matcher(
            args.dataset,
            args.resize,
            args.matcher,
            args.detector,
            args.output_csv,
            args.board,
        )
        return

    # Otherwise, run all benchmarks
    resize_values = [800, 600, 400, 300, 200]

    # Define which matchers work with which detectors
    superpoint_matchers = ["lightglue", "superglue"]
    xfeat_matchers = ["lighterglue"]

    # Run all superpoint matchers
    for matcher in superpoint_matchers:
        for resize in resize_values:
            run_single_benchmark(
                args.dataset,
                resize,
                matcher,
                detector="superpoint",
                output_csv=args.output_csv,
                board=args.board,
            )

    # Run all xfeat matchers
    for matcher in xfeat_matchers:
        for resize in resize_values:
            run_single_benchmark(
                args.dataset,
                resize,
                matcher,
                detector="xfeat",
                output_csv=args.output_csv,
                board=args.board,
            )

    print(f"All benchmark results written to {args.output_csv}")


if __name__ == "__main__":
    main()
