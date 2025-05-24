#!/usr/bin/env python3
"""
Script to quantize all VPR models to TensorRT format.
This script runs each model quantization in a separate subprocess to handle memory properly.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from nnsb.vpr_systems.cosplace.cosplace import CosPlace
from nnsb.vpr_systems.eigenplaces.eigenplaces import EigenPlaces
from nnsb.vpr_systems.mixvpr.mixvpr_shrunk import MixVPRShrunk
from nnsb.vpr_systems.netvlad.netvlad import NetVLAD
from nnsb.vpr_systems.salad.salad_shrunk import SALADShrunk
from nnsb.vpr_systems.sela.sela_shrunk import SelaShrunk

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "netvlad": {
        "class": NetVLAD,
        "args": {
            "weights": "weights/mapillary_WPCA4096.pth.tar",
        },
        "resize_values": [800, 600, 400, 300, 200],
    },
    "eigenplaces": {
        "class": EigenPlaces,
        "args": {
            "backbone": "ResNet101",
            "fc_output_dim": 2048,
        },
        "resize_values": [800, 600, 400, 300, 200],
    },
    "cosplace": {
        "class": CosPlace,
        "args": {
            "backbone": "ResNet101",
            "fc_output_dim": 2048,
        },
        "resize_values": [800, 600, 400, 300, 200],
    },
    "salad": {
        "class": SALADShrunk,
        "args": {},
        "resize_values": [800, 600, 400, 300, 200],
    },
    "sela": {
        "class": SelaShrunk,
        "args": {
            "path_to_state_dict": "weights/SelaVPR_msls.pth",
            "dinov2_path": "weights/dinov2_vitl14_pretrain.pth",
        },
        "resize_values": [224],  # Sela uses fixed 224x224 input
    },
    "mixvpr": {
        "class": MixVPRShrunk,
        "args": {
            "ckpt_path": "weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"
        },
        "resize_values": [320],  # MixVPR uses fixed 320x320 input
    },
}


def run_single_model_quantization(
    model_name: str,
    resize: int,
    calibration_data_dir: Path,
    calibration_data_name: str,
    output_path: Path,
) -> bool:
    """
    Run a single model quantization in a subprocess by calling this script with specific parameters.

    Args:
        model_name: Name of the model to quantize
        resize: Resize value for the model
        calibration_dataset: Path to calibration dataset
        output_path: Path to save the TensorRT model

    Returns:
        True if quantization was successful, False otherwise
    """
    # Build command to call this script with --single-model parameter
    cmd = [
        sys.executable,
        __file__,
        "--single-model",
        "--model",
        model_name,
        "--resize",
        str(resize),
        "--calibration-data-dir",
        str(calibration_data_dir),
        "--calibration-data-name",
        str(calibration_data_name),
        "--output-path",
        str(output_path),
    ]

    # Add any model-specific arguments
    for key, value in MODEL_CONFIGS[model_name]["args"].items():
        cmd.extend([f"--{key}", str(value)])

    logger.info(f"Starting quantization subprocess for {model_name}_{resize}")
    logger.debug(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error(f"Quantization failed for {model_name}_{resize} in {elapsed:.2f}s")
        logger.error(f"Error: {result.stderr}")
        return False

    logger.info(f"Successfully quantized {model_name}_{resize} in {elapsed:.2f}s")
    return True


def quantize_single_model(
    model_name: str,
    resize: int,
    calibration_data_dir: Path,
    calibration_data_name,
    output_path: Path,
    **model_args,
) -> bool:
    """
    Quantize a single model. This is run in a dedicated process.

    Args:
        model_name: Name of the model to quantize
        resize: Resize value for the model
        calibration_dataset: Path to calibration dataset
        output_path: Path to save the TensorRT model
        model_args: Additional model-specific arguments
    """
    try:
        # Initialize the model with the provided arguments
        model_class = MODEL_CONFIGS[model_name]["class"]

        # Adjust resize for specific models
        if model_name == "salad":
            model_args["resize"] = resize // 14 * 14
        elif model_name not in ("sela", "mixvpr"):
            model_args["resize"] = resize

        logger.info(f"Initializing {model_name} with resize={resize}")
        model = model_class(**model_args)

        # Create a temporary directory for the ONNX file
        temp_dir = Path("/tmp/quantization_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = temp_dir / f"{model_name}_{resize}.onnx"

        # Step 1: Quantize and export to ONNX
        logger.info(f"Exporting {model_name}_{resize} to ONNX")
        onnx_success = model.quantize_to_onnx(
            calibration_data_dir, calibration_data_name, onnx_path, resize
        )

        if not onnx_success or not onnx_path.exists():
            logger.error(f"Failed to export {model_name}_{resize} to ONNX")
            return False

        # Step 2: Convert ONNX to TensorRT using trtexec
        logger.info(f"Converting {model_name}_{resize} ONNX to TensorRT")

        # Make sure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run the TensorRT conversion using trtexec
        trt_cmd = [
            "/usr/src/tensorrt/bin/trtexec",
            f"--onnx={onnx_path}",
            "--int8",
            f"--saveEngine={output_path}",
        ]

        logger.info(f"Running TensorRT conversion: {' '.join(trt_cmd)}")
        trt_result = subprocess.run(trt_cmd, capture_output=True, text=True)

        # Clean up temporary ONNX file
        try:
            if onnx_path.exists():
                os.remove(onnx_path)
                logger.debug(f"Removed temporary ONNX file: {onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary ONNX file: {e}")

        if trt_result.returncode != 0:
            logger.error(f"TensorRT conversion failed: {trt_result.stderr}")
            return False

        logger.info(f"TensorRT engine successfully created at {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error quantizing {model_name}_{resize}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize VPR models to TensorRT format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/tensorrt",
        help="Directory to save the TensorRT models",
    )
    parser.add_argument(
        "--calibration-data-dir",
        type=str,
        required=False,
        help="Path to calibration dataset",
    )
    parser.add_argument(
        "--calibration-data-name",
        type=str,
        required=False,
        help="Name of calibration dataset",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to quantize",
    )
    parser.add_argument(
        "--resize", type=int, nargs="+", help="Specific resize values to use"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Arguments for single model quantization mode
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Run quantization for a single model",
    )
    parser.add_argument("--model", type=str, help="Model name for single model mode")
    parser.add_argument(
        "--output-path", type=str, help="Output path for single model mode"
    )

    # Parse all other arguments that might be model-specific
    args, unknown = parser.parse_known_args()

    # Process additional model-specific arguments
    extra_args = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                extra_args[key] = unknown[i + 1]
                i += 2
            else:
                extra_args[key] = True
                i += 1
        else:
            i += 1

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Single model quantization mode (called as subprocess)
    if args.single_model:
        if (
            not args.model
            or not args.resize
            or not args.calibration_data_dir
            or not args.calibration_data_name
            or not args.output_path
        ):
            logger.error("Missing required arguments for single model quantization")
            return 1

        success = quantize_single_model(
            model_name=args.model,
            resize=int(args.resize[0]),
            calibration_data_dir=Path(args.calibration_data_dir),
            calibration_data_name=Path(args.calibration_data_name),
            output_path=Path(args.output_path),
            **extra_args,
        )

        return 0 if success else 1

    # Main mode - process all models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.calibration_data_dir:
        logger.error("Calibration dataset is required for quantization")
        return 1

    calibration_data_dir = Path(args.calibration_data_dir)
    if not calibration_data_dir.exists():
        logger.error(f"Calibration dataset not found: {calibration_data_dir}")
        return 1

    results = {}

    for model_name in args.models:
        results[model_name] = {}

        resize_values = (
            args.resize if args.resize else MODEL_CONFIGS[model_name]["resize_values"]
        )

        for resize in resize_values:
            if model_name == "sela" and resize != 224:
                logger.warning(
                    f"Sela only supports 224x224 input. Skipping resize={resize}"
                )
                continue

            if model_name == "mixvpr" and resize != 320:
                logger.warning(
                    f"MixVPR only supports 320x320 input. Skipping resize={resize}"
                )
                continue

            output_path = output_dir / f"{model_name}_{resize}.trt"

            success = run_single_model_quantization(
                model_name=model_name,
                resize=resize,
                calibration_data_dir=calibration_data_dir,
                calibration_data_name=args.calibration_data_name,
                output_path=output_path,
            )

            results[model_name][resize] = "Success" if success else "Failed"

    # Write results to file
    result_file = output_dir / "quantization_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    # Output summary
    logger.info("\n===== Quantization Results =====")
    for model_name, model_results in results.items():
        logger.info(f"{model_name}:")
        for resize, status in model_results.items():
            logger.info(f"  - {resize}: {status}")
    logger.info(f"Results saved to {result_file}")

    # Exit with appropriate status code
    failed = any(
        status == "Failed"
        for model_results in results.values()
        for status in model_results.values()
    )

    if failed:
        logger.warning("Some quantizations failed. Check the logs for details.")
        return 1
    else:
        logger.info("All requested models successfully quantized to TensorRT format.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
