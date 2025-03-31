#!/bin/bash

# Check operating system and adjust commands accordingly
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows system (Git Bash or similar)
    DATE_CMD="date +%Y%m%d_%H%M%S"
    MKDIR_CMD="mkdir -p"
    TEE_CMD="tee -a"
    OS_TYPE="Windows"
else
    # Unix-like system
    DATE_CMD="date +%Y%m%d_%H%M%S"
    MKDIR_CMD="mkdir -p"
    TEE_CMD="tee -a"
    OS_TYPE="Unix"
fi

# Configuration
TIMESTAMP=$($DATE_CMD)
OUTPUT_DIR="./fine_tuned_typhoon_$TIMESTAMP"
CACHE_DIR="./dataset_cache"
LOG_FILE="training_$TIMESTAMP.log"

# Create directories
$MKDIR_CMD "$OUTPUT_DIR"
$MKDIR_CMD "$CACHE_DIR"

echo "Operating System: $OS_TYPE"
echo "Starting Typhoon Model training at $(date)" | $TEE_CMD "$LOG_FILE"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH" | $TEE_CMD "$LOG_FILE"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Using Python command: $PYTHON_CMD" | $TEE_CMD "$LOG_FILE"

# Check if required packages are installed
echo "Checking required packages..." | $TEE_CMD "$LOG_FILE"
$PYTHON_CMD -c "import torch; import transformers; import datasets; import peft; import trl" || {
    echo "Error: Missing required packages. Please install requirements first:" | $TEE_CMD "$LOG_FILE"
    echo "pip install -r requirements.txt" | $TEE_CMD "$LOG_FILE"
    exit 1
}

# Run the training script
echo "Starting training process..." | $TEE_CMD "$LOG_FILE"
$PYTHON_CMD typhoon_model_training.py \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  2>&1 | $TEE_CMD "$LOG_FILE"

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training failed" | $TEE_CMD "$LOG_FILE"
    exit 1
fi

# Log completion
echo "Training completed at $(date)" | $TEE_CMD "$LOG_FILE"
echo "Model saved to $OUTPUT_DIR" | $TEE_CMD "$LOG_FILE"

# Run a quick evaluation on the trained model
echo "Running evaluation on trained model..." | $TEE_CMD "$LOG_FILE"
$PYTHON_CMD typhoon_model_training.py \
  --model_path "$OUTPUT_DIR" \
  --eval_only \
  2>&1 | $TEE_CMD "$LOG_FILE"

echo "Process completed successfully!" | $TEE_CMD "$LOG_FILE"

# Create a Windows batch file equivalent
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Creating Windows batch file version..."
    cat > run_training.bat << EOF
@echo off
set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set OUTPUT_DIR=./fine_tuned_typhoon_%TIMESTAMP%
set CACHE_DIR=./dataset_cache
set LOG_FILE=training_%TIMESTAMP%.log

mkdir "%OUTPUT_DIR%" 2>nul
mkdir "%CACHE_DIR%" 2>nul

echo Starting Typhoon Model training at %date% %time% > %LOG_FILE%

python typhoon_model_training.py --output_dir "%OUTPUT_DIR%" --cache_dir "%CACHE_DIR%" 2>&1 >> %LOG_FILE%

echo Training completed at %date% %time% >> %LOG_FILE%
echo Model saved to %OUTPUT_DIR% >> %LOG_FILE%

echo Running evaluation on trained model... >> %LOG_FILE%
python typhoon_model_training.py --model_path "%OUTPUT_DIR%" --eval_only 2>&1 >> %LOG_FILE%

echo Process completed successfully! >> %LOG_FILE%
EOF
    echo "Windows batch file created: run_training.bat"
fi
