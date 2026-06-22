import fiftyone as fo
import os

# Input COCO dataset paths
splits = {
    "train": {
        "images": "../datasets/SyntheticBodiesAtSea/images/training",
        "annotations": "../datasets/SyntheticBodiesAtSea/annotations/coco/instances_train.json",
    },
    "val": {
        "images": "../datasets/SyntheticBodiesAtSea/images/validation",
        "annotations": "../datasets/SyntheticBodiesAtSea/annotations/coco/instances_val.json",
    },
    "test": {
        "images": "../datasets/SyntheticBodiesAtSea/images/test",
        "annotations": "../datasets/SyntheticBodiesAtSea/annotations/coco/instances_test.json",
    },
}

output_dir = "../datasets/SyntheticBodiesAtSea/annotations/yolo51"
os.makedirs(output_dir, exist_ok=True)

# Clear old cached sessions to prevent schema corruption
dataset_name = "synthetic_bodies_at_sea"
if dataset_name in fo.list_datasets():
    print(f"Purging existing dataset '{dataset_name}' from backend...")
    fo.delete_dataset(dataset_name)

master_dataset = fo.Dataset(name=dataset_name, persistent=False)
detected_classes = set()

for split, paths in splits.items():
    print(f"Loading {split} split...")
    
    split_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=paths["images"],
        labels_path=paths["annotations"],
        name=f"temp_{split}",
    )
    
    # Extract the true original class names from the COCO metadata directly
    # This completely bypasses the need for the dataset schema queries to find 'ground_truth'
    if "classes" in split_dataset.info:
        detected_classes.update(split_dataset.info["classes"])
    
    split_dataset.tag_samples(split)
    master_dataset.merge_samples(split_dataset)
    split_dataset.delete()

# Format the absolute class tracking mapping for the exporter
classes = sorted(list(detected_classes))

# CRITICAL FALLBACK: If classes metadata is still empty, manually grab from the default field
if not classes:
    # Look for any valid string values in the default detection path
    classes = sorted(list(master_dataset.distinct("detections.detections.label")))

# Hard stop protection if the dataset contains no annotations whatsoever
if not classes:
    raise ValueError(
        "No classes could be extracted. Please verify that your JSON annotation files "
        "contain valid 'categories' and 'annotations' lists."
    )

print(f"\nFinal Class Mapping Verified ({len(classes)}):")
for idx, cls in enumerate(classes):
    print(f"  ID {idx} -> {cls}")

print(master_dataset.default_classes)


# Export cleanly
print(f"\nExporting single unified YOLO dataset to: {output_dir}")

# Export the splits
print(list(splits.keys()))
for split in list(splits.keys()):
    split_view = master_dataset.match_tags(split)
    print(split_view)
    split_view.export(
        export_dir=output_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="detections",
        split=split,
        export_media="symlink",
        classes=master_dataset.default_classes,
    )

print("Finished successfully!")