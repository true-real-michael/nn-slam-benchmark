"""
Worker script to run TensorRT quantization in a subprocess.
This ensures memory gets properly freed after the quantization process completes.
"""

import argparse
import importlib
from pathlib import Path
import sys

from nnsb.vpr_systems.cosplace.cosplace import CosPlace
from nnsb.vpr_systems.eigenplaces.eigenplaces import EigenPlaces
from nnsb.vpr_systems.netvlad.netvlad import NetVLAD
from nnsb.vpr_systems.salad.salad_shrunk import SALADShrunk
from nnsb.vpr_systems.sela.sela_shrunk import SelaShrunk
from nnsb.vpr_systems.mixvpr.mixvpr_shrunk import MixVPRShrunk


def main():
    """Main entry point for the quantization worker subprocess"""
    parser = argparse.ArgumentParser(description="TensorRT Quantization Worker")
    parser.add_argument("--model_class", type=str, required=True, 
                        help="Full class name (module.class) of the model to quantize")
    parser.add_argument("--calibration_path", type=str, required=True,
                        help="Path to calibration data")
    parser.add_argument("--onnx_path", type=str, required=True,
                        help="Output path for the ONNX model")
    parser.add_argument("--resize", type=int, required=True,
                        help="Image resize value")
    
    # Parse any additional model-specific arguments
    args, unknown = parser.parse_known_args()
    
    # Create a dict of additional args (remove '--' prefix)
    extra_args = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                extra_args[key] = unknown[i + 1]
                i += 2
            else:
                extra_args[key] = True
                i += 1
        else:
            i += 1
    
    try:
        # Import the model class dynamically
        module_name, class_name = args.model_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        
        # Create an instance of the model
        model_instance = model_class(**extra_args)
        
        # Run the quantization
        result = model_instance._quantize_to_onnx(
            Path(args.calibration_path), 
            args.onnx_path, 
            args.resize
        )
        
        # Exit with appropriate status code
        sys.exit(0 if result else 1)
        
    except Exception as e:
        print(f"Error during quantization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
