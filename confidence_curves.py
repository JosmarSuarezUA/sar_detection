import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
import yaml
from ultralytics import YOLO
from tools.wandb_helper import WandbHelper

# =========================
# CONFIGURACIÓN
# =========================


SPLIT = "val"

OUTPUT_ROOT = Path("results_threshold")

LOW_CONF = 0.001
MATCH_IOU = 0.20

THRESHOLDS = [
    0.001, 0.005, 0.010, 0.020, 0.030,
    0.050, 0.075, 0.100, 0.150, 0.200,
    0.300, 0.400, 0.500, 0.600, 0.700,
    0.800, 0.900
]


# =========================
# CLI / DATASET RESOLUTION
# =========================


def build_parser() -> argparse.ArgumentParser:
    # Generate a new parser that matches the inputs for the run_eval function
    parser = argparse.ArgumentParser(
        description="Run confidence-threshold sweep for a W&B model artifact and log results to W&B."
    )
    parser.add_argument(
            "--project",
            required=False,
            default="sar_detection",
            help="W&B project to log the run to"
        )
    parser.add_argument(
        "--train_run_name",
        required=False,
        default=None,
        help="Name of the W&B run/group (e.g. 'seadronesseejp_500_epochs')"
    )
    parser.add_argument(
        "--model-path",
        required=False,
        default=None,
        help="Path to the W&B model artifact"
    )
    parser.add_argument(
            "--wandb-model-path",
            required=False,
            help="Path to the W&B model artifact"
        )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use W&B artifact download and logging; disable to run locally with --model-path"
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="Path to the YOLO dataset YAML file"
    )
    parser.add_argument(
        "--split",
        default=SPLIT,
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where evaluation results are written"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=THRESHOLDS,
        help="List of confidence thresholds to sweep over"
    )
    parser.add_argument(
        "--low-conf",
        type=float,
        default=LOW_CONF,
        help="Low confidence threshold for initial predictions"
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=MATCH_IOU,
        help="IoU threshold for matching predictions to ground truth"
    )
    return parser


def load_dataset_config(data_yaml_path: Path) -> dict:
    with open(data_yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_split_paths(data_yaml_path: Path, split: str) -> tuple[Path, Path]:
    dataset_config = load_dataset_config(data_yaml_path)

    if split not in dataset_config:
        available_splits = ", ".join(
            key for key in ("train", "val", "test") if key in dataset_config
        )
        raise KeyError(
            f"Split '{split}' not found in {data_yaml_path}. "
            f"Available splits: {available_splits or 'none'}"
        )

    dataset_root_value = dataset_config.get("path")
    if dataset_root_value is None:
        dataset_root = data_yaml_path.parent.resolve()
    else:
        dataset_root_path = Path(dataset_root_value)
        dataset_root = (
            dataset_root_path
            if dataset_root_path.is_absolute()
            else (data_yaml_path.parent / dataset_root_path).resolve()
        )

    split_path = Path(dataset_config[split])
    image_dir = (
        split_path
        if split_path.is_absolute()
        else (dataset_root / split_path).resolve()
    )

    if split_path.parts and split_path.parts[0] == "images":
        label_split_path = Path("labels", *split_path.parts[1:])
    else:
        label_split_path = Path("labels", *split_path.parts)

    label_dir = (dataset_root / label_split_path).resolve()

    return image_dir, label_dir


def resolve_wandb_model_path(train_run_name: str | None, project: str, wandb_model_path: str | None) -> str:
    if wandb_model_path:
        return wandb_model_path

    if not train_run_name:
        raise ValueError("wandb_model_path must be provided when use_wandb=True.")

    wbhelper = WandbHelper(project=project)
    run_id = wbhelper.get_run_id_by_name(train_run_name)
    inferred_model_path = wbhelper.get_ultralytics_best_model_path(run_id)

    if not inferred_model_path:
        raise ValueError(
            "wandb_model_path was not provided and could not be inferred from the train run."
        )

    return inferred_model_path


# =========================
# FUNCIONES AUXILIARES
# =========================


def load_yolo_labels(label_path: Path, image_width: int, image_height: int) -> list[dict]:
    """
    Lee etiquetas YOLO:
    class_id x_center y_center width height
    y devuelve cajas xyxy en píxeles.
    """
    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()

            if len(values) < 5:
                continue

            class_id = int(values[0])
            xc, yc, w, h = map(float, values[1:5])

            x1 = (xc - w / 2) * image_width
            y1 = (yc - h / 2) * image_height
            x2 = (xc + w / 2) * image_width
            y2 = (yc + h / 2) * image_height

            ground_truth.append({
                "class_id": class_id,
                "box": np.array([x1, y1, x2, y2], dtype=float)
            })

    return ground_truth


def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calcula IoU entre dos cajas xyxy."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def match_predictions(
    predictions: list[dict], ground_truth: list[dict], iou_threshold: float
) -> tuple[int, int, int]:
    """
    Emparejamiento greedy uno a uno.
    Las predicciones se procesan de mayor a menor confianza.
    La clase debe coincidir.
    """
    predictions = sorted(
        predictions,
        key=lambda prediction: prediction["confidence"],
        reverse=True
    )

    used_ground_truth = set()
    true_positives = 0
    false_positives = 0

    for prediction in predictions:
        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(ground_truth):
            if gt_index in used_ground_truth:
                continue

            if prediction["class_id"] != gt["class_id"]:
                continue

            iou = calculate_iou(prediction["box"], gt["box"])

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_iou >= iou_threshold:
            true_positives += 1
            used_ground_truth.add(best_gt_index)
        else:
            false_positives += 1

    false_negatives = len(ground_truth) - len(used_ground_truth)

    return true_positives, false_positives, false_negatives


def calculate_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    beta = 2
    f2 = (
        (1 + beta ** 2) * precision * recall /
        ((beta ** 2 * precision) + recall)
        if (beta ** 2 * precision) + recall > 0 else 0.0
    )

    return precision, recall, f1, f2


# =========================
# PREDICCIONES A BAJO UMBRAL
# =========================


def predict_low_conf(
    model: YOLO,
    image_dir: Path,
    label_dir: Path,
    low_conf: float = LOW_CONF,
) -> list[dict]:
    """Run YOLO predictions at a very low confidence threshold and pair each
    image's detections with its ground-truth boxes, so later threshold sweeps
    can simply filter by confidence without re-running inference."""
    image_paths = sorted(
        path for path in image_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Procesando {len(image_paths)} imágenes en {image_dir}...")

    all_images = []

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=low_conf,
            iou=0.7,
            max_det=300,
            verbose=False
        )

        result = results[0]
        image_height, image_width = result.orig_shape

        predictions = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confidences, classes):
                predictions.append({
                    "box": box,
                    "confidence": float(confidence),
                    "class_id": int(class_id)
                })

        relative_path = image_path.relative_to(image_dir)
        label_path = label_dir / relative_path.with_suffix(".txt")

        ground_truth = load_yolo_labels(label_path, image_width, image_height)

        all_images.append({
            "image": str(image_path),
            "predictions": predictions,
            "ground_truth": ground_truth
        })

    return all_images


# =========================
# BARRIDO DE UMBRALES
# =========================


def sweep_thresholds(
    all_images: list[dict],
    thresholds: list[float] = THRESHOLDS,
    match_iou: float = MATCH_IOU,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the cached low-confidence predictions at each threshold and
    compute precision/recall/F1/F2 plus a few detection-count diagnostics."""
    threshold_results = []
    detailed_results = []

    for threshold in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        images_without_detections = 0
        detections_per_image = []
        retained_confidences = []

        for image_data in all_images:
            filtered_predictions = [
                prediction
                for prediction in image_data["predictions"]
                if prediction["confidence"] >= threshold
            ]

            number_of_detections = len(filtered_predictions)
            detections_per_image.append(number_of_detections)

            if number_of_detections == 0:
                images_without_detections += 1

            retained_confidences.extend(
                prediction["confidence"]
                for prediction in filtered_predictions
            )

            tp, fp, fn = match_predictions(
                filtered_predictions,
                image_data["ground_truth"],
                match_iou
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            detailed_results.append({
                "threshold": threshold,
                "image": image_data["image"],
                "detections": number_of_detections,
                "ground_truth": len(image_data["ground_truth"]),
                "tp": tp,
                "fp": fp,
                "fn": fn
            })

        precision, recall, f1, f2 = calculate_metrics(total_tp, total_fp, total_fn)

        threshold_results.append({
            "threshold": threshold,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "mean_detections_per_image": np.mean(detections_per_image),
            "std_detections_per_image": np.std(detections_per_image),
            "images_without_detections": images_without_detections,
            "percentage_images_without_detections":
                100 * images_without_detections / len(all_images),
            "mean_retained_confidence":
                np.mean(retained_confidences) if retained_confidences else 0.0,
            "std_retained_confidence":
                np.std(retained_confidences) if retained_confidences else 0.0
        })

    summary_df = pd.DataFrame(threshold_results)
    details_df = pd.DataFrame(detailed_results)

    # Once a threshold exceeds the model's effective confidence ceiling, no
    # predictions survive the filter at all (tp = fp = 0), which makes
    # precision/recall/F1/F2 mathematically undefined rather than "0". The
    # loop above falls back to 0.0 for those undefined ratios, which produces
    # a misleading cliff at the tail of the curves. Instead, hold those
    # metrics at the last threshold where at least one prediction survived —
    # this mirrors how ultralytics' own P/R/F1 curves behave beyond the max
    # observed confidence (flat-lining via interpolation instead of dropping
    # to zero). Raw tp/fp/fn/detection counts are left untouched so the
    # "no detections" diagnostic columns still reflect what really happened.
    no_predictions_mask = (summary_df["tp"] + summary_df["fp"]) == 0
    metric_columns = ["precision", "recall", "f1", "f2"]
    summary_df.loc[no_predictions_mask, metric_columns] = np.nan
    summary_df[metric_columns] = summary_df[metric_columns].ffill()

    return summary_df, details_df


def select_best_threshold(summary_df: pd.DataFrame, metric: str = "f2") -> tuple[float, pd.Series]:
    best_row = summary_df.loc[summary_df[metric].idxmax()]
    return float(best_row["threshold"]), best_row


def save_results(
    output_dir: Path,
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    best_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "threshold_summary.csv", index=False)
    details_df.to_csv(output_dir / "image_details.csv", index=False)

    with open(output_dir / "selected_threshold.txt", "w", encoding="utf-8") as file:
        file.write(f"{best_threshold:.6f}\n")


# =========================
# WANDB LOGGING
# =========================


def log_threshold_curves_to_wandb(
    run: "wandb.sdk.wandb_run.Run",  
    summary_df: pd.DataFrame,
) -> None:
    """Log precision/recall/F1/F2 vs confidence curves as W&B Custom Charts,
    plus a raw summary table for inspection in the W&B UI.

    Note: we build the charts with `wandb.plot.line(..., split_table=True)`
    directly instead of going through ultralytics' `wb._plot_curve` helper.
    That helper hardcodes `split_table=False`, which dumps a
    "<Chart Title>_table" entry into the main Tables section for every curve
    (e.g. "Precision-Confidence Curve_table"). Setting `split_table=True`
    keeps the chart but tucks its backing table away under the separate
    "Custom Chart Tables" section instead of cluttering the main workspace.
    """
    x = summary_df["threshold"].to_numpy()


    for metric_name, y_title in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("f2", "F2"),
    ):
        y = summary_df[metric_name].to_numpy()

        curve_table = wandb.Table(
            data=list(zip(x.tolist(), y.tolist())),
            columns=["Confidence", y_title],
        )

        chart_title = f"{metric_name.title()}-Confidence Curve"
        chart = wandb.plot.line(
            curve_table,
            x="Confidence",
            y=y_title,
            title=chart_title,
            split_table=True,
        )
        run.log({chart_title: chart})


# =========================
# NUEVA FUNCIÓN: EVALUACIÓN + LOGGING EN WANDB
# =========================


def run_eval(
    train_run_name: str,
    project: str,
    data_yaml: str | Path,
    wandb_model_path: str | None = None,
    model_path: str | None = None,
    use_wandb: bool = True,
    split: str = SPLIT,
    output_dir: Path = OUTPUT_ROOT,
    thresholds: list[float] | None = None,
    low_conf: float = LOW_CONF,
    match_iou: float = MATCH_IOU,
) -> dict[str, Any]:
    """Run the confidence-threshold sweep for a W&B model artifact and log the
    resulting precision/recall/F1/F2-vs-confidence curves to Weights & Biases.

    Parameters
    ----------
    train_run_name: Name used for the W&B run/group (e.g. "seadronesseejp_500_epochs").
    project: W&B project to log the run to.
    data_yaml: Path to the YOLO dataset YAML file.
    wandb_model_path: W&B artifact path for the model weights
        (e.g. "entity/project/run_xxx_model:best"), downloaded and expected to
        contain a "best.pt" file.
    """
    thresholds = thresholds or THRESHOLDS
    data_yaml = Path(data_yaml).resolve()
    run_label = train_run_name or data_yaml.parent.name

    if use_wandb:
        wandb_model_path = resolve_wandb_model_path(train_run_name, project, wandb_model_path)
    else:
        if not model_path:
            raise ValueError("model_path must be provided when use_wandb=False.")

    configs = {
        "train_run_name": train_run_name,
        "data_yaml": str(data_yaml),
        "wandb_model_path": wandb_model_path,
        "model_path": model_path,
        "split": split,
        "thresholds": thresholds,
        "low_conf": low_conf,
        "match_iou": match_iou,
    }
    
    run = (
        wandb.init(
            project=project,
            job_type="eval",
            name=f"curves_{run_label}",
            tags=["conf_curves"],
            config=configs,
        )
        if use_wandb
        else None
    )

    try:

        # Load the model from W&B or from a local path, depending on the mode.
        if use_wandb:
            artifact = run.use_artifact(wandb_model_path)
            model_path = Path(artifact.download()) / "best.pt"
            model = YOLO(str(model_path))
        elif model_path:
            model = YOLO(model_path)
        else:
            raise ValueError("Either model_path or wandb_model_path must be provided.")

        
        image_dir, label_dir = resolve_split_paths(data_yaml, split)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        run_output_dir = output_dir / run_label / split

        all_images = predict_low_conf(model, image_dir, label_dir, low_conf=low_conf)
        summary_df, details_df = sweep_thresholds(
            all_images, thresholds=thresholds, match_iou=match_iou
        )
        best_threshold, best_row = select_best_threshold(summary_df, metric="f2")

        save_results(run_output_dir, summary_df, details_df, best_threshold)
        if use_wandb and run is not None:
            log_threshold_curves_to_wandb(run, summary_df)

    
        # Logging the best thresholds and their corresponding metric values to W&B summary

        if use_wandb and run is not None:
            best_f1_threshold, best_f1_row = select_best_threshold(summary_df, metric="f1")
            best_f2_threshold, best_f2_row = select_best_threshold(summary_df, metric="f2")
            best_precision_threshold, best_precision_row = select_best_threshold(summary_df, metric="precision")
            best_recall_threshold, best_recall_row = select_best_threshold(summary_df, metric="recall")

            run.summary["best_f1_confidence"] = best_f1_threshold
            run.summary["best_f1"] = float(best_f1_row["f1"])
            run.summary["best_f2_confidence"] = best_f2_threshold
            run.summary["best_f2"] = float(best_f2_row["f2"])
            run.summary["best_precision_confidence"] = best_precision_threshold
            run.summary["best_precision"] = float(best_precision_row["precision"])
            run.summary["best_recall_confidence"] = best_recall_threshold
            run.summary["best_recall"] = float(best_recall_row["recall"])

            # Logging a summary table of the best thresholds and their corresponding metric values to W&B
            summary_table = wandb.Table(
                columns=["Metric", "Best Confidence Threshold", "Best Value"],
                data=[
                    ["F1", run.summary["best_f1_confidence"], run.summary["best_f1"]],
                    ["F2", run.summary["best_f2_confidence"], run.summary["best_f2"]],
                    ["Precision", run.summary["best_precision_confidence"], run.summary["best_precision"]],
                    ["Recall", run.summary["best_recall_confidence"], run.summary["best_recall"]],
                ],
            )
            run.log({"best_thresholds_summary": summary_table})


        # Create a summary of the best threshold and metrics for logging to W&B summary and also as a table
        


        print("\nResultados:")
        print(summary_df.to_string(index=False))
        print("\nUmbral seleccionado:")
        print(f"conf = {best_threshold:.6f}")
        print(f"F2   = {best_row['f2']:.4f}")
        print(f"Precision = {best_row['precision']:.4f}")
        print(f"Recall    = {best_row['recall']:.4f}")
    finally:
        if run is not None:
            run.finish()

    return {
        "summary_df": summary_df,
        "details_df": details_df,
        "best_threshold": best_threshold,
    }

if __name__ == "__main__":

    args = build_parser().parse_args()

    train_run_name = args.train_run_name
    project = args.project
    data_yaml = args.data_yaml
    split = args.split
    output_dir = args.output_dir
    match_iou = args.match_iou
    model_path = args.model_path
    wandb_model_path = args.wandb_model_path

    run_eval(
        train_run_name=train_run_name,
        
        project=project,
        data_yaml=data_yaml,
        wandb_model_path=wandb_model_path,
        model_path=model_path,
        use_wandb=args.use_wandb,
        split=split,
        output_dir=output_dir,
        thresholds=args.thresholds,
        low_conf=args.low_conf,
        match_iou=match_iou,
    )