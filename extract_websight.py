#!/usr/bin/env python3

import os
import argparse
import glob
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm
import uuid
import base64
from pathlib import Path


def extract_parquet_data(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.parquet",
    max_files: int = None,
    max_samples: int = None,
):
    """
    Extract HTML and images from WebSight parquet files and save them to disk.

    Args:
        input_dir: Directory containing parquet files
        output_dir: Directory to save extracted HTML and images
        pattern: Glob pattern to match parquet files
        max_files: Maximum number of parquet files to process
        max_samples: Maximum number of samples to extract per file
    """
    # Find all parquet files
    parquet_files = glob.glob(os.path.join(input_dir, pattern))
    if max_files:
        parquet_files = parquet_files[:max_files]

    print(f"Found {len(parquet_files)} parquet files matching pattern '{pattern}'")

    total_extracted = 0

    for file_path in tqdm(parquet_files, desc="Processing parquet files"):
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)

            if max_samples:
                df = df.head(max_samples)

            # Process each row
            for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc=f"Extracting from {os.path.basename(file_path)}",
            ):
                # Generate a unique ID for the pair
                pair_id = str(uuid.uuid4())

                # Extract HTML content
                html_content = None
                if "html" in row:
                    html_content = row["html"]
                elif "text" in row:
                    html_content = row["text"]

                # Extract image data
                image_data = None
                for img_col in ["image", "img", "screenshot"]:
                    if img_col in row:
                        image_data = row[img_col]
                        break

                # Skip if either HTML or image is missing
                if html_content is None or image_data is None:
                    continue

                # Save HTML content
                html_file_path = os.path.join(output_dir, f"{pair_id}.html")
                with open(html_file_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

                # Process and save image
                try:
                    # Handle different image data formats
                    pil_image = None
                    if isinstance(image_data, Image.Image):
                        pil_image = image_data
                    elif isinstance(image_data, dict) and "bytes" in image_data:
                        img_bytes = image_data["bytes"]
                        pil_image = Image.open(io.BytesIO(img_bytes))
                    elif isinstance(image_data, bytes):
                        pil_image = Image.open(io.BytesIO(image_data))
                    elif isinstance(image_data, str) and image_data.startswith(
                        "data:image"
                    ):
                        # Handle base64 encoded images
                        image_data = image_data.split(",")[1]
                        img_bytes = base64.b64decode(image_data)
                        pil_image = Image.open(io.BytesIO(img_bytes))

                    if pil_image:
                        img_file_path = os.path.join(output_dir, f"{pair_id}.png")
                        pil_image.save(img_file_path)
                        total_extracted += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                    # Remove the HTML file if image processing failed
                    if os.path.exists(html_file_path):
                        os.remove(html_file_path)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"Total extracted pairs: {total_extracted}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract HTML and images from WebSight parquet files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted HTML and images",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.parquet",
        help="Glob pattern to match parquet files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of parquet files to process",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to extract per file",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract data from parquet files
    extract_parquet_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        max_files=args.max_files,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
