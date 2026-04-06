#!/bin/bash

# 1. Define the path to trtexec
# Usually found in /usr/src/tensorrt/bin/ or in your PATH
TRTEXEC_PATH="/usr/src/tensorrt/bin/trtexec"

# 2. Check if trtexec exists
if [ ! -f "$TRTEXEC_PATH" ]; then
    echo "Error: TensorRT (trtexec) not found at $TRTEXEC_PATH"
    echo "Please install TensorRT or update the TRTEXEC_PATH in this script."
    echo "Common install command: sudo apt-get install tensorrt"
    exit 1
fi

# 3. Check if correct number of arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_onnx_file> <output_engine_file>"
    echo "Example: $0 yolo_model_100.onnx model_rtx3060.engine"
    exit 1
fi

INPUT_ONNX=$1
OUTPUT_ENGINE=$2

# 4. Check if input file exists
if [ ! -f "$INPUT_ONNX" ]; then
    echo "Error: Input file $INPUT_ONNX does not exist."
    exit 1
fi

echo "Starting conversion: $INPUT_ONNX -> $OUTPUT_ENGINE"
echo "Targeting FP16 precision for RTX 3060..."

# 5. Run the conversion
# --fp16: Greatly increases speed on RTX 3060
# --avgRuns=10: Benchmarks the engine after building
$TRTEXEC_PATH \
    --onnx="$INPUT_ONNX" \
    --saveEngine="$OUTPUT_ENGINE" \
    --fp16 \
    --avgRuns=10

if [ $? -eq 0 ]; then
    echo "----------------------------------------------------"
    echo "Success! Engine saved to $OUTPUT_ENGINE"
    echo "----------------------------------------------------"
else
    echo "Error: TensorRT conversion failed."
    exit 1
fi
