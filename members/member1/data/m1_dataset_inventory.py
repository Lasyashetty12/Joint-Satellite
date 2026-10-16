"""Inventory image files in a small satellite dataset folder."""

import argparse
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".npy"
}


def inventory_dataset(data_dir: Path):
    """Return supported files and their sizes beneath data_dir."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a folder, got: {data_dir}")

    results = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            results.append((path, path.stat().st_size))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="List image files in a satellite dataset folder."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset folder to inspect",
    )
    args = parser.parse_args()

    try:
        results = inventory_dataset(args.data_dir)
    except (FileNotFoundError, NotADirectoryError) as error:
        parser.error(str(error))

    print(f"Dataset folder: {args.data_dir.resolve()}")
    print(f"Supported image files found: {len(results)}")
    if not results:
        print("No supported images found. Check the folder path and file formats.")
        return

    print("\nFiles:")
    for path, size in results:
        print(f"- {path} | {size:,} bytes")


if __name__ == "__main__":
    main()
