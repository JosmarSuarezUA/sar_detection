import fiftyone as fo
import os
import textwrap
from typing import Dict, List, Optional, Any

class FiftyOneDatasetManager:
    """
    A unified manager to handle FiftyOne dataset imports, merges, 
    and multi-format exports seamlessly across projects.
    """
    
    # Supported format mapping for easier user strings
    FORMAT_MAPPING = {
        "coco": fo.types.COCODetectionDataset,
        "yolov4": fo.types.YOLOv4Dataset,
        "yolov5": fo.types.YOLOv5Dataset,
        "cvat": fo.types.CVATImageDataset,
        "voc": fo.types.VOCDetectionDataset,
    }

    def __init__(self, dataset_name: str = None, persistent: bool = False):
        self.dataset_name = dataset_name
        self.persistent = persistent
        self.classes: List[str] = []
        self._dataset: Optional[fo.Dataset] = None
        
        self._purge_existing_backend()

    @property
    def dataset(self) -> fo.Dataset:
        """Access the underlying FiftyOne dataset instance."""
        if self._dataset is None:
            self._dataset = fo.Dataset(name=self.dataset_name, persistent=self.persistent)
        return self._dataset
    
    def set_dataset(self, dataset: fo.Dataset) -> None:
        """Sets the underlying FiftyOne dataset instance."""
        self._dataset = dataset
        self.dataset_name = dataset.name
        self.recalculate_classes()

    def recalculate_classes(self) -> List[str]:
        """Returns the list of classes in the dataset."""
        try:
            if self._dataset and self._dataset.has_field("detections"):
                self.classes = self.dataset.distinct("detections.detections.label")
            elif self._dataset and self._dataset.has_field("ground_truth"):
                self.classes = self.dataset.distinct("ground_truth.detections.label")
            else:
                self.classes = []
        except Exception as e:
            print(f"Error recalculating classes: {e}")
            self.classes = []
        return self.classes
    @property
    def sample_count(self) -> int:
        """Returns total number of samples currently in the dataset."""
        return len(self.dataset) if self._dataset else 0

    def _purge_existing_backend(self):
        """Clears old cached sessions to prevent schema corruption."""
        if self.dataset_name in fo.list_datasets():
            print(f"Purging existing dataset '{self.dataset_name}' from backend...")
            fo.delete_dataset(self.dataset_name)

    def _reset_dataset(self):
        """Ensure the underlying FiftyOne dataset is empty by deleting
        and recreating it. This is used to guarantee imports start
        from a clean dataset state.
        """
        if self.dataset_name in fo.list_datasets():
            print(f"Resetting existing dataset '{self.dataset_name}'...")
            fo.delete_dataset(self.dataset_name)
        # Recreate a fresh dataset instance
        self._dataset = fo.Dataset(name=self.dataset_name, persistent=self.persistent)

    def delete_dataset(self):
        """Deletes the underlying FiftyOne dataset instance."""
        if self._dataset and fo.dataset_exists(self._dataset.name):
            print(f"Deleting dataset '{self._dataset.name}'...")
            fo.delete_dataset(self._dataset.name)
            self._dataset = None
            self.classes = []
    def import_coco_splits(self, splits_config: Dict[str, Dict[str, str]]) -> 'FiftyOneDatasetManager':
        """
        Imports and merges multiple dataset splits (e.g., train, val, test).
        Format types: 'coco', 'yolov5', etc.
        """
        # Ensure we start from an empty dataset
        self._reset_dataset()

        detected_classes = set()

        for split, paths in splits_config.items():
            print(f"Loading {split} split in {self.FORMAT_MAPPING.get('coco')} format...")
            
            # Dynamically pull the correct arguments based on format requirements
            import_kwargs = {
                "dataset_type": self.FORMAT_MAPPING.get('coco'),
                "data_path": paths.get("images") or paths.get("data_path"),
                "name": f"temp_{self.dataset_name}_{split}",
            }
            if "annotations" in paths:
                import_kwargs["labels_path"] = paths["annotations"]

            split_dataset = fo.Dataset.from_dir(**import_kwargs)
            
            # Extract true original class names if present in metadata
            if "classes" in split_dataset.info:
                detected_classes.update(split_dataset.info["classes"])
            
            split_dataset.tag_samples(split)
            self.dataset.merge_samples(split_dataset)
            split_dataset.delete()

        # Resolve final unified classes
        self.classes = sorted(list(detected_classes))
        if not self.classes:
            self.classes = sorted(list(self.dataset.distinct("detections.detections.label")))
        if not self.classes:
            self.classes = sorted(list(self.dataset.distinct("ground_truth.detections.label")))

        if not self.classes:
            raise ValueError("No classes could be extracted. Please check your annotation files.")

        print(f"\nFinal Class Mapping Verified ({len(self.classes)}):")
        for idx, cls in enumerate(self.classes):
            print(f"  ID {idx} -> {cls}")

        # Update core dataset properties for downstream usage
        self.dataset.default_classes = self.classes
        return self
    
    def import_yolov4_splits(self, splits_config: Dict[str, Dict[str, str]]) -> 'FiftyOneDatasetManager':
        """
        Imports and merges multiple YOLOv4 dataset splits (e.g., train, val, test).
        """
        # Ensure we start from an empty dataset
        self._reset_dataset()
        for split, paths in splits_config.items():
            print(f"Loading {split} split in {self.FORMAT_MAPPING.get('yolov4')} format...")
            
            # Dynamically pull the correct arguments based on format requirements
            import_kwargs = {
                "dataset_type": self.FORMAT_MAPPING.get('yolov4'),
                "images_path": paths.get("images_path"),
                "labels_path": paths.get("labels_path"),
                "objects_path": paths.get("objects_path"),
                "name": f"temp_{self.dataset_name}_{split}",
            }
            split_dataset = fo.Dataset.from_dir(**import_kwargs)
            
            # Extract true original class names if present in metadata
            # if "classes" in split_dataset.info:
            #     detected_classes.update(split_dataset.info["classes"])
            
            split_dataset.tag_samples(split)
            self.dataset.merge_samples(split_dataset)
            split_dataset.delete()

        # Resolve final unified classes
        # self.classes = sorted(list(detected_classes))
        # if not self.classes:
        #     self.classes = sorted(list(self.dataset.distinct("detections.detections.label")))

        # if not self.classes:
        #     raise ValueError("No classes could be extracted. Please check your annotation files.")

        self.classes = list(self.dataset.default_classes) if self.dataset.default_classes else list(self.dataset.distinct("ground_truth.detections.label"))
        print(f"\nFinal Class Mapping Verified ({len(self.classes)}):")
        for idx, cls in enumerate(self.classes):
            print(f"  ID {idx} -> {cls}")

        # Update core dataset properties for downstream usage
        self.dataset.default_classes = self.classes
        return self
    def import_yolov5_yaml(self, yaml_path: str, splits: List[str] = ['train', 'val']) -> 'FiftyOneDatasetManager':
        """
        Imports a YOLOv5 dataset defined by a YAML configuration file.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YOLOv5 YAML file not found: {yaml_path}")

        print(f"Importing YOLOv5 dataset from YAML: {yaml_path}")

        # Start from an empty dataset to avoid mixing previous samples
        self._reset_dataset()

        print(f"Yaml exists: {os.path.exists(yaml_path)}")

        for split in splits:
            self.dataset.add_dir(
                dataset_type=fo.types.YOLOv5Dataset,
                yaml_path=yaml_path,
                split=split,
                tags=split,
            )

        self.recalculate_classes()
        return self
    
    def generate_yolov5_yaml(self, dataset, output_dir: str) -> None:
        classes = dataset.distinct("detections.detections.label")

        yaml_content = textwrap.dedent("""
            path: .
            train: ./images/train/
            val: ./images/val/

            names:
        """)
        for i, cls in enumerate(classes):
            yaml_content += f"  {i}: {cls}\n"

        yaml_path = os.path.join(output_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

    def export_splits(self, output_dir: str, format_type: str = "yolov5", label_field: str = "detections", export_media: str = False):
        """
        Exports tagged dataset splits into the designated format.
        """
        dataset_type = self.FORMAT_MAPPING.get(format_type.lower())
        if not dataset_type:
            raise ValueError(f"Unsupported export format: {format_type}")

        if not self.dataset.distinct("tags"):
            raise ValueError("Dataset has no splits tagged. Please run 'import_splits' first.")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nExporting dataset to: {output_dir} using format {format_type.upper()}")

        for split in self.dataset.distinct("tags"):
            split_view = self.dataset.match_tags(split)
            print(f"Exporting split '{split}' ({len(split_view)} samples)...")
            
            split_view.export(
                export_dir=output_dir if format_type.lower() != "voc" else os.path.join(output_dir, split),
                dataset_type=dataset_type,
                label_field=label_field,
                split=split,
                export_media=export_media,
                classes=self.classes,
                labels_path= f"instances_{split}.json" if format_type.lower() == "coco" else None,
            )
        if(format_type.lower() == "yolov5"):
            self.generate_yolov5_yaml(self.dataset, output_dir)
            # pass
        print("Export finished successfully!")

    # def map_classes(self, class_mapping: Dict[str, str]) -> 'FiftyOneDatasetManager':
    #     """
    #     Maps existing class labels to new labels based on a provided mapping.
    #     """
    #     if not class_mapping:
    #         raise ValueError("Class mapping dictionary is empty.")

    #     print(f"Mapping classes using provided mapping: {class_mapping}")
    #     for sample in self.dataset:
    #         detections = sample['detections']
    #         for detection in detections.detections:
    #             if detection.label in class_mapping:
    #                 detection.label = class_mapping[detection.label]
    #         sample.save()

    #     # Update the internal classes list
    #     self.classes = sorted(list(set(class_mapping.values())))
    #     self.dataset.default_classes = self.classes
    #     return self

 