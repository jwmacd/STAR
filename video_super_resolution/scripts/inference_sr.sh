#!/bin/bash

# Set strict error handling
set -euo pipefail
trap 'echo "Error occurred. Exiting..." >&2; exit 1' ERR

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
FRAME_LENGTH=4  # Number of video frames processed simultaneously
UPSCALE=1

# Function to check if required directories and files exist
check_prerequisites() {
    echo "Checking prerequisites..."
    local missing=0

    # Check directories
    for dir in "$VIDEO_DIR" "$TEXT_DIR" "$MODEL_DIR" "$RESULTS_DIR"; do
        if [ ! -d "$dir" ]; then
            echo "Creating directory: $dir"
            mkdir -p "$dir" || {
                echo "Error: Failed to create directory: $dir" >&2
                return 1
            }
        fi
    done

    # Check required files with more descriptive messages
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Error: Required model file not found at: $MODEL_FILE" >&2
        missing=1
    fi

    if [ ! -f "$PROMPT_FILE" ]; then
        echo "Error: Required prompt file not found at: $PROMPT_FILE" >&2
        missing=1
    fi

    return $missing
}

# Main processing function
process_videos() {
    local exit_code=0

    # Get all .mp4 files in the folder using find to handle special characters
    if ! mapfile -t mp4_files < <(find "$VIDEO_DIR" -type f -name "*.mp4" -print0 | sort -z | tr '\0' '\n'); then
        echo "Error: Failed to read MP4 files" >&2
        return 1
    fi

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

        echo "Processing video file: $mp4_file"
        echo "Using prompt: $line"

        if ! torchrun \
            --nnodes=1 \
            --nproc_per_node=4 \
            "${BASE_DIR}/scripts/inference_sr.py" \
            --input_path "${mp4_file}" \
            --model_path "${MODEL_FILE}" \
            --prompt "${line}" \
            --upscale ${UPSCALE} \
            --max_chunk_len ${FRAME_LENGTH} \
            --file_name "${file_name}.mp4" \
            --save_dir "${RESULTS_DIR}"; then
            
            echo "Error: Failed to process video: $file_name" >&2
            exit_code=1
            continue  # Continue with next video instead of stopping completely
        fi

        echo "Successfully processed: $file_name"
    done

    return $exit_code
}

# Main execution
echo "Starting video super-resolution processing..."
if ! check_prerequisites; then
    echo "Prerequisites check failed. Exiting."
    exit 1
fi
process_videos
echo "All videos processed successfully."