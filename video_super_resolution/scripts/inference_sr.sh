#!/bin/bash

# Set strict error handling
set -euo pipefail
trap 'echo "Error occurred. Exiting..." >&2; exit 1' ERR

# PyTorch memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Base directory setup
BASE_DIR="/app/video_super_resolution"
INPUT_DIR="${BASE_DIR}/input"
VIDEO_DIR="${INPUT_DIR}/video"
TEXT_DIR="${INPUT_DIR}/text"
MODEL_DIR="${BASE_DIR}/pretrained_weight"
RESULTS_DIR="${BASE_DIR}/results"

# Input files
PROMPT_FILE="${TEXT_DIR}/prompt.txt"
MODEL_FILE="${MODEL_DIR}/model.pt"

# Processing parameters
FRAME_LENGTH=8  # Number of video frames processed simultaneously
UPSCALE=2

# Function to check if required directories and files exist
check_prerequisites() {
    echo "Checking prerequisites..."
    local missing=0

    # Check directories
    for dir in "$VIDEO_DIR" "$TEXT_DIR" "$MODEL_DIR"; do
        if [ ! -d "$dir" ]; then
            echo "Error: Required directory not found: $dir"
            missing=1
        fi
    done

    # Check model file
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Error: Model file not found: $MODEL_FILE"
        missing=1
    fi

    # Check prompt file
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "Error: Prompt file not found: $PROMPT_FILE"
        missing=1
    fi

    # Create results directory if it doesn't exist
    mkdir -p "$RESULTS_DIR"

    return $missing
}

# Main processing function
process_videos() {
    # Get all .mp4 files in the folder using find to handle special characters
    mapfile -t mp4_files < <(find "$VIDEO_DIR" -type f -name "*.mp4")

    # Print the list of MP4 files
    echo "MP4 files to be processed:"
    for mp4_file in "${mp4_files[@]}"; do
        echo "$mp4_file"
    done

    # Read lines from the text file, skipping empty lines
    mapfile -t lines < <(grep -v '^\s*$' "$PROMPT_FILE")

    # Debugging output
    echo "Number of MP4 files: ${#mp4_files[@]}"
    echo "Number of lines in the text file: ${#lines[@]}"

    # Ensure we have at least one video to process
    if [ ${#mp4_files[@]} -eq 0 ]; then
        echo "Error: No MP4 files found in $VIDEO_DIR"
        exit 1
    fi

    # Ensure the number of video files matches the number of lines
    if [ ${#mp4_files[@]} -ne ${#lines[@]} ]; then
        echo "Error: Number of MP4 files (${#mp4_files[@]}) and prompts (${#lines[@]}) do not match."
        exit 1
    fi

    # Loop through video files and corresponding lines
    for i in "${!mp4_files[@]}"; do
        mp4_file="${mp4_files[$i]}"
        line="${lines[$i]}"
        
        # Extract the filename without the extension
        file_name=$(basename "$mp4_file" .mp4)
        
        echo "Processing video file: $mp4_file with prompt: $line"
            
        # Run Python script with parameters
        python3 \
            "${BASE_DIR}/scripts/inference_sr.py" \
            --solver_mode 'fast' \
            --steps 15 \
            --input_path "${mp4_file}" \
            --model_path "${MODEL_FILE}" \
            --prompt "${line}" \
            --upscale ${UPSCALE} \
            --max_chunk_len ${FRAME_LENGTH} \
            --file_name "${file_name}.mp4" \
            --save_dir "${RESULTS_DIR}"

        echo "Completed processing: $file_name"
    done
}

# Main execution
echo "Starting video super-resolution processing..."
if ! check_prerequisites; then
    echo "Prerequisites check failed. Exiting."
    exit 1
fi
process_videos
echo "All videos processed successfully."