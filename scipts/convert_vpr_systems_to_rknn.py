#!/usr/bin/env python3
"""
Script to convert all VPR models to RKNN format.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

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


def convert_model(
    model_name: str, resize: int, output_dir: Path, dataset_path: Optional[Path] = None
) -> bool:
    """
    Convert a specific model to RKNN format

    Args:
        model_name: Name of the model to convert
        resize: Resize value for the model
        output_dir: Directory to save the RKNN model
        dataset_path: Path to quantization dataset
    """
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        return False

    config = MODEL_CONFIGS[model_name]
    model_class = config["class"]
    model_args = config["args"].copy()

    if resize not in config["resize_values"]:
        logger.warning(
            f"Resize {resize} not in supported values for {model_name}: {config['resize_values']}"
        )
        return False

    logger.info(f"Converting {model_name} with resize={resize}")

    output_path = output_dir / f"{model_name}_{resize}.rknn"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if model_name == "salad":
            model_args["resize"] = resize // 14 * 14
        elif model_name not in ("sela", "mixvpr"):
            model_args["resize"] = resize

        model = model_class(**model_args)

        logger.info(f"Exporting {model_name}_{resize} to {output_path}")
        model.export_rknn(output_path, quantization_dataset=dataset_path)
        logger.info(f"Successfully exported {model_name}_{resize} to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error converting {model_name}_{resize}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert VPR models to RKNN format")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/rknn",
        help="Directory to save the RKNN models",
    )
    parser.add_argument("--dataset", type=str, help="Path to quantization dataset")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to convert",
    )
    parser.add_argument(
        "--resize", type=int, nargs="+", help="Specific resize values to use"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir)
    dataset_path = Path(args.dataset) if args.dataset else None

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

            success = convert_model(
                model_name,
                resize,
                output_dir,
                dataset_path,
            )
            results[model_name][resize] = "Success" if success else "Failed"

    logger.info("\n===== Conversion Results =====")
    for model_name, model_results in results.items():
        logger.info(f"{model_name}:")
        for resize, status in model_results.items():
            logger.info(f"  - {resize}: {status}")

    failed = any(
        status == "Failed"
        for model_results in results.values()
        for status in model_results.values()
    )

    if failed:
        logger.warning("Some conversions failed. Check the logs for details.")
        return 1
    else:
        logger.info("All requested models successfully converted to RKNN format.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
