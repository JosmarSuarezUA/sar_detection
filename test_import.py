from tools.annotation_converter import FiftyOneDatasetManager
import fiftyone as fo

splits = {
    "train": {
        # "images": "./datasets/synbase/SyntheticBodiesAtSea/images/training",
        "images": "./datasets/synbase/SyntheticBodiesAtSea/images/train",
        "annotations": "./datasets/synbase/SyntheticBodiesAtSea/annotations/coco/instances_train.json",
    },
    "val": {
        # "images": "./datasets/synbase/SyntheticBodiesAtSea/images/validation",
        "images": "./datasets/synbase/SyntheticBodiesAtSea/images/val",
        "annotations": "./datasets/synbase/SyntheticBodiesAtSea/annotations/coco/instances_val.json",
    }
}

# For yolo import
yaml_path = "./datasets/synbase/SyntheticBodiesAtSea/annotations/yolo51/dataset.yaml"
splits_yolov5 = ['train', 'val']

# ## COCO Import Example

# # Pipeline execution via method chaining
manager = FiftyOneDatasetManager(dataset_name="synthetic_bodies_at_sea")
manager.import_coco_splits(splits, format_type="coco")

# # Inspect clean properties anywhere in your pipeline
print(f"Total images processed: {manager.sample_count}")
print(f"Target classes: {manager.classes}")


print(manager.dataset.distinct("tags"))
print(manager.dataset.tags)
# Export to  COCO, YOLOv5, and VOC formats
manager.export_splits(output_dir="datasets/synbase/SyntheticBodiesAtSea/annotations/coco51", format_type="coco", label_field="detections", export_media=False)
manager.export_splits(output_dir="datasets/synbase/SyntheticBodiesAtSea", format_type="yolov5", label_field="detections", export_media=False)
manager.export_splits(output_dir="datasets/synbase/SyntheticBodiesAtSea/annotations/voc51", format_type="voc", label_field="detections", export_media=False)

# session = fo.launch_app(dataset=manager.dataset)
# session.wait()