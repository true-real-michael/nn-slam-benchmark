#!/bin/bash

# Base directory containing the weight subdirectories
BASE_DIR="../weights/quant"

# Find all subdirectories (200, 300, etc.) in the weights directory
for subdir in "$BASE_DIR"/*/; do
    # Remove trailing slash to get the directory name
    subdir="${subdir%/}"
    
    # Get just the number part (200, 300, etc.) from the path
    number=$(basename "$subdir")
    
    # Find all .onnx files in the subdirectory
    for onnx_file in "$subdir"/*.onnx; do
        # Skip if no ONNX files found
        [ -e "$onnx_file" ] || continue
        
        # Get the filename without path
        onnx_filename=$(basename "$onnx_file")
        
        # Create the output engine filename
        engine_file="$subdir/${onnx_filename%.onnx}.trt"
        
        # Run trtexec command
        echo "Processing $onnx_file -> $engine_file"
        /usr/src/tensorrt/bin/trtexec \
            --onnx="$onnx_file" \
            --int8 \
            --saveEngine="$engine_file"
        
        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "Successfully created $engine_file"
        else
            echo "Failed to process $onnx_file"
        fi
    done
done

echo "All processing complete."
