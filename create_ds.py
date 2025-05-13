import os
import json
import glob
import random
import argparse
from math import floor


def create_train_val_test_split(
    input_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    image_prefix="",
    seed=42,
    caption_as_string=True,
    filetype="gui",
    max_length=None,
    length_stats=False,
):
    """
    Create train, validation, and test splits from paired .png and .gui files.

    Args:
        input_dir: Directory containing the .png and .gui files
        output_dir: Directory to save the output JSON files
        train_ratio: Proportion of data for training set (default: 0.8)
        val_ratio: Proportion of data for validation set (default: 0.1)
        test_ratio: Proportion of data for test set (default: 0.1)
        image_prefix: Optional prefix for image paths in the JSON
        seed: Random seed for reproducibility
        caption_as_string: If True, caption will be a string; if False, caption will be a list
        filetype: File extension for the text files (default: gui)
        max_length: Maximum allowed length for the text files (default: None = no limit)
        length_stats: Whether to output length statistics (default: False)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(seed)

    # Find all paired files
    paired_files = []
    filtered_out = []
    lengths = []

    gui_files = glob.glob(os.path.join(input_dir, f"*.{filetype}"))

    for gui_file in gui_files:
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(gui_file))[0]

        # Construct the path to the corresponding PNG file
        png_file = os.path.join(input_dir, f"{base_name}.png")

        # Check if the PNG file exists
        if os.path.exists(png_file):
            # Read the content of the text file to check its length
            with open(gui_file, "r") as f:
                content = f.read().strip()
                content_length = len(content)

                # Track lengths for statistics
                if length_stats:
                    lengths.append((base_name, content_length))

                # Apply length filtering if specified
                if max_length is not None and content_length > max_length:
                    filtered_out.append((base_name, content_length))
                else:
                    paired_files.append((png_file, gui_file, base_name, content))

    # Print filtering statistics if max_length was specified
    if max_length is not None:
        print(
            f"Filtered out {len(filtered_out)} files exceeding maximum length of {max_length} characters"
        )
        print(f"Retained {len(paired_files)} files within length limit")

        # Show some examples of filtered files if any were filtered out
        if filtered_out and len(filtered_out) > 0:
            print("\nExamples of filtered files:")
            for i, (filename, length) in enumerate(
                sorted(filtered_out, key=lambda x: x[1], reverse=True)[:5]
            ):
                print(f"  {filename}: {length} characters")

    # Print length statistics if requested
    if length_stats and lengths:
        lengths_only = [l for _, l in lengths]
        print("\nLength statistics for all files:")
        print(f"  Min length: {min(lengths_only)} characters")
        print(f"  Max length: {max(lengths_only)} characters")
        print(
            f"  Average length: {sum(lengths_only) / len(lengths_only):.2f} characters"
        )

        # Calculate percentiles
        lengths_only.sort()
        p90_idx = int(0.9 * len(lengths_only))
        p95_idx = int(0.95 * len(lengths_only))
        p99_idx = int(0.99 * len(lengths_only))

        print(f"  90th percentile: {lengths_only[p90_idx]} characters")
        print(f"  95th percentile: {lengths_only[p95_idx]} characters")
        print(f"  99th percentile: {lengths_only[p99_idx]} characters")

    # Create a mapping of filenames to integer IDs
    # Sort the files first to ensure consistent IDs across runs
    sorted_file_list = sorted([base_name for _, _, base_name, _ in paired_files])
    id_mapping = {filename: i + 1 for i, filename in enumerate(sorted_file_list)}

    # Shuffle the data
    random.shuffle(paired_files)

    # Calculate split indices
    total_samples = len(paired_files)
    train_end = floor(total_samples * train_ratio)
    val_end = train_end + floor(total_samples * val_ratio)

    # Split the data
    train_files = paired_files[:train_end]
    val_files = paired_files[train_end:val_end]
    test_files = paired_files[val_end:]

    # Create and save splits with different caption formats
    create_json_file(
        train_files,
        os.path.join(output_dir, "train.json"),
        image_prefix,
        id_mapping,
        caption_as_string=caption_as_string,
    )
    create_json_file(
        val_files,
        os.path.join(output_dir, "val.json"),
        image_prefix,
        id_mapping,
        caption_as_string=caption_as_string,
    )
    create_json_file(
        test_files,
        os.path.join(output_dir, "test.json"),
        image_prefix,
        id_mapping,
        caption_as_string=caption_as_string,
    )

    # Print summary
    print(
        f"\nCreated splits with {len(train_files)} training samples, {len(val_files)} validation samples, and {len(test_files)} test samples"
    )
    print(f"Generated {len(id_mapping)} unique integer IDs for images")


def create_json_file(
    file_pairs, output_path, image_prefix="", id_mapping=None, caption_as_string=False
):
    """
    Create a JSON file from a list of file pairs.

    Args:
        file_pairs: List of tuples (png_file, gui_file, base_name, content)
        output_path: Path to save the JSON file
        image_prefix: Optional prefix for image paths in the JSON
        id_mapping: Dictionary mapping filenames to integer IDs
        caption_as_string: If True, caption will be a string; if False, caption will be a list
    """
    data = []

    for _, _, base_name, gui_content in file_pairs:
        # Get the integer ID for this image
        image_id = id_mapping.get(base_name, 0) if id_mapping else 0

        # Create an entry for this pair
        entry = {
            "image": f"{image_prefix}{base_name}.png",
            "image_id": image_id,  # Now using an integer ID
        }

        # Format caption based on parameter
        if caption_as_string:
            entry["caption"] = gui_content  # As a single string
        else:
            entry["caption"] = [gui_content]  # As a list with one item

        data.append(entry)

    # Write the data to the output JSON file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Created {output_path} with {len(data)} entries")


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Create train, validation, and test splits from paired .png and .gui files.",
        epilog="""
Example usage:
  python script.py --input-dir datasets/COCO/web/all_data --output-dir datasets/COCO
  python script.py --input-dir input_folder --output-dir output_folder --max-length 5000
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=False,
        default="datasets/websight/all_data",
        help="Directory containing the .png and .gui files",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=False,
        default="datasets/websight",
        help="Directory to save the output JSON files",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training set (default: 0.8)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation set (default: 0.1)",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for test set (default: 0.1)",
    )

    parser.add_argument(
        "--image-prefix",
        type=str,
        default="",
        help="Optional prefix for image paths in the JSON",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--caption-as-list",
        action="store_true",
        help="If set, caption will be stored as a list; otherwise as a string (default: string)",
    )

    parser.add_argument(
        "--filetype",
        "-f",
        type=str,
        default="html",
        help="Filetype for the text files (default: gui)",
    )

    parser.add_argument(
        "--max-length",
        "-m",
        type=int,
        default=2000,
        help="Maximum allowed length for text files in characters (default: None = no limit)",
    )

    parser.add_argument(
        "--length-stats",
        "-l",
        action="store_true",
        help="Output statistics about text file lengths",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate that the ratios sum to 1
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:  # Allow a small floating-point error
        parser.error(
            f"The sum of train, validation, and test ratios should be 1.0, but got {ratio_sum}"
        )

    return args


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Call the function with parsed arguments
    create_train_val_test_split(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        filetype=args.filetype,
        image_prefix=args.image_prefix,
        seed=args.seed,
        caption_as_string=not args.caption_as_list,  # Invert the flag to match the original behavior
        max_length=args.max_length,
        length_stats=args.length_stats,
    )
