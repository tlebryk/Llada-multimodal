#!/usr/bin/env bash
set -euo pipefail

# URL of the parquet file (use the 'resolve/main' path for raw download)
URL="https://huggingface.co/datasets/HuggingFaceM4/WebSight/resolve/main/v0.2/train-00000-of-00738-80a58552f2fb3344.parquet"

# Directories
RAW_DIR="datasets/websight/raw"
EXTRACTED_DIR="datasets/websight/all_data"
FINAL_DIR="datasets/websight"

# Make sure all target dirs exist
mkdir -p "${RAW_DIR}" "${EXTRACTED_DIR}" "${FINAL_DIR}"

# Download the Parquet file
echo "Downloading Parquet to ${RAW_DIR}..."
curl -L --progress-bar -o "${RAW_DIR}/train-00000-of-00738-80a58552f2fb3344.parquet" "${URL}"

# Run extraction
echo "Running extract_websight.py..."
python extract_websight.py \
    --input_dir "${RAW_DIR}" \
    --output_dir "${EXTRACTED_DIR}"

# Build final dataset
echo "Running create_ds.py..."
python create_ds.py \
    --input_dir "${EXTRACTED_DIR}" \
    --output_dir "${FINAL_DIR}" \
    --filetype "html" \
    -m 2000

echo "All done! Dataset is in ${FINAL_DIR}/"
