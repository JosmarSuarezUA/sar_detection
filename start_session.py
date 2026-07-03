import argparse
from pathlib import Path
import json

import fiftyone as fo

from tools.annotation_converter import FiftyOneDatasetManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a FiftyOne session for SyntheticBodiesAtSea data")
    parser.add_argument(
        "--format",
        choices=["coco", "yolov5"],
        default="coco",
        help="Annotation format to import",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "SyntheticBodiesAtSea",
        help="Path to the dataset root directory",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional custom FiftyOne dataset name",
    )
    parser.add_argument(
        "--yaml-path",
        default=None,
        help="Path to the YOLOv5 dataset YAML file",
    )
    parser.add_argument(
        "--splits-json",
        default=None,
        help="Path to the splits JSON file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()

    if args.format == "coco":
        #Read json splits
        splits_json_path = args.splits_json
        with open(splits_json_path, "r") as f:
            splits_config = json.load(f)
        dataset_name = args.dataset_name or None
        manager = FiftyOneDatasetManager(dataset_name=dataset_name)
        manager.import_coco_splits(splits_config, format_type="coco")
    else:
        yaml_path = str(args.yaml_path)
        dataset_name = args.dataset_name or None
        manager = FiftyOneDatasetManager(dataset_name=dataset_name)
        manager.import_yolov5_yaml(yaml_path=yaml_path, splits=["train", "val"])

    dataset = manager.dataset
    print(f"Loaded {len(dataset)} samples into FiftyOne.")

    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()