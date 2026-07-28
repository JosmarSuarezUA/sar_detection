"""
da_metrics.py

Utility functions to extract domain-adaptation-relevant metrics from an
Ultralytics YOLO model and log them to a single, organized W&B eval run.

Logging conventions used here (each serves a different purpose -- not
duplicates of each other):
  - run.summary["{target}/metric"]  -> final scalar per target. This is what
    populates sortable/filterable columns in the W&B *project* Runs table,
    so you can compare "B/map50" across many different training runs.
  - run.log({"results_table": wandb.Table(...)})  -> one browsable table
    inside THIS run, including array-valued columns (per-class P/R) that
    don't fit as scalars.
  - CSV artifact -> optional, only if you want results outside W&B entirely
    (e.g. feeding pandas for paper figures without hitting the W&B API).

Usage pattern:
    1. Train on Source X  -> best.pt (logged as a W&B model artifact)
    2. In the eval script: evaluate_source_against_targets(...) handles
       every target dataset in one call, logs scalars + table + validator
       plots + embeddings + domain gap + t-SNE, all organized under
       "{target}/..." keys.
    3. Add datasets by adding entries to `dataset_configs` -- nothing else
       needs to change.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from tools.iou20_detectionvalidator import IoU20DetectionValidator
from ultralytics import YOLO
from tools.wandb_helper import WandbHelper

# IoU 0.20 -> index 0, 0.50 -> 6, 0.75 -> 11, 0.95 -> 15 (0.05 steps from 0.20 to 0.95)
IOU_THRESH_INDEX = {0.20: 0, 0.50: 6, 0.75: 11, 0.95: 15}


# ---------------------------------------------------------------------------
# 1. Detection metrics: mAP, precision, recall (+ validator plot directory)
# ---------------------------------------------------------------------------

def evaluate_detection_metrics(
    model: YOLO,
    data_yaml: str,
    split: str = "test",
    plots: bool = True,
    iou_threshold: float = 0.20,
    conf_threshold: float = 0.20
) -> tuple[dict[str, Any], Path]:
    """Evaluate detection metrics over IoU thresholds from 0.20 to 0.95.

    Returns:
        (metrics_dict, save_dir) where save_dir is the folder Ultralytics
        wrote confusion_matrix.png / PR_curve.png / val_batch*.jpg etc. into
        (only populated when plots=True).
    """
    metrics = model.val(
        data=data_yaml,
        split=split,
        verbose=False,
        plots=plots,
        validator=IoU20DetectionValidator,
        iou=iou_threshold,
        conf=conf_threshold
    )

    all_ap = np.asarray(metrics.box.all_ap)  # (n_classes_with_targets, n_iou_thresholds)

    if all_ap.size == 0:
        result = {"map20_95": 0.0, "map50_95": 0.0, "map20": 0.0, "map50": 0.0, "map75": 0.0, "map95": 0.0}
    else:
        result = {
            "map20_95": float(all_ap.mean()),
            "map50_95": float(all_ap[:, IOU_THRESH_INDEX[0.50]:].mean()),
            "map20": float(all_ap[:, IOU_THRESH_INDEX[0.20]].mean()),
            "map50": float(all_ap[:, IOU_THRESH_INDEX[0.50]].mean()),
            "map75": float(all_ap[:, IOU_THRESH_INDEX[0.75]].mean()),
            "map95": float(all_ap[:, IOU_THRESH_INDEX[0.95]].mean()),
        }

    result.update({
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "precision_per_class": metrics.box.p.tolist(),
        "recall_per_class": metrics.box.r.tolist(),
    })

    return result, Path(metrics.save_dir)


# ---------------------------------------------------------------------------
# 2. Confidence mean
# ---------------------------------------------------------------------------

def compute_confidence_mean(model: YOLO, image_dir: str, iou_threshold: float = 0.20, conf_threshold: float = 0.20) -> dict:
    """Average (and std) confidence of predictions on a folder of images.

    conf_threshold is set to 0.20 by default, but you can lower it to 0.001 if you want to see the confidence 
    distribution of *all* predictions, not just the high-confidence ones.
    """
    results = model.predict(image_dir, iou=iou_threshold, conf=conf_threshold, verbose=False)
    all_confs = []
    for r in results:
        if r.boxes is not None and len(r.boxes):
            all_confs.extend(r.boxes.conf.cpu().numpy().tolist())

    if not all_confs:
        return {"confidence_mean": None, "confidence_std": None, "n_detections": 0}

    return {
        "confidence_mean": float(np.mean(all_confs)),
        "confidence_std": float(np.std(all_confs)),
        "n_detections": len(all_confs),
    }


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------

def extract_embeddings(model: YOLO, image_dir: str, extensions=(".jpg", ".jpeg", ".png")) -> tuple:
    """One embedding vector per image, using Ultralytics' built-in model.embed().

    Returns:
        embeddings: np.ndarray of shape (n_images, feature_dim)
        image_paths: list of str, same order as embeddings rows
    """
    image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in extensions)
    embeddings = [model.embed(str(p), verbose=False)[0].cpu().numpy() for p in image_paths]
    return np.stack(embeddings), [str(p) for p in image_paths]


# ---------------------------------------------------------------------------
# 4. Full pipeline for one (model, target dataset) evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    model_path: str,
    data_yaml: str,
    image_dir: str,
    split: str = "test",
    run_name: str = "",
    iou_threshold: float = 0.20,
    conf_threshold: float = 0.20
) -> tuple[dict, np.ndarray, list[str], Path]:
    """Runs detection metrics + confidence + embeddings for one eval.

    Returns (metrics_dict, embeddings_array, image_paths, validator_plots_dir).
    """
    model = YOLO(model_path)

    det_metrics, plots_dir = evaluate_detection_metrics(model, 
                                                        data_yaml, 
                                                        split=split, 
                                                        iou_threshold=iou_threshold, 
                                                        conf_threshold=conf_threshold)
    conf_metrics = compute_confidence_mean(model, 
                                           image_dir, 
                                           iou_threshold=iou_threshold, 
                                           conf_threshold=conf_threshold)

    embeddings, image_paths = extract_embeddings(model, image_dir)

    result = {
        "run_name": run_name,
        "model_path": model_path,
        "data_yaml": data_yaml,
        "split": split,
        **det_metrics,
        **conf_metrics,
    }
    return result, embeddings, image_paths, plots_dir


# ---------------------------------------------------------------------------
# 5. "Improved precision / recall" -- delta vs a baseline run
# ---------------------------------------------------------------------------

def compute_improvement(baseline_result: dict, adapted_result: dict) -> dict:
    """Compare an adapted-model run against a source-only baseline run on the
    SAME target dataset/split. This is what "improved precision/recall" means
    in most DA papers: not the raw value, but the delta over naive transfer."""
    def _delta(key):
        b, a = baseline_result.get(key), adapted_result.get(key)
        return None if b is None or a is None else a - b

    return {
        "run_name": f"{adapted_result['run_name']}_vs_{baseline_result['run_name']}",
        "delta_map50_95": _delta("map50_95"),
        "delta_map50": _delta("map50"),
        "delta_precision": _delta("precision_mean"),
        "delta_recall": _delta("recall_mean"),
        "delta_confidence_mean": _delta("confidence_mean"),
    }


# ---------------------------------------------------------------------------
# 6. Domain gap from embeddings (quantitative + visual)
# ---------------------------------------------------------------------------

def compute_domain_gap_mmd(source_embeddings: np.ndarray, target_embeddings: np.ndarray, gamma: float = 1.0) -> float:
    """RBF-kernel Maximum Mean Discrepancy. Lower = source/target feature
    distributions are closer (less domain gap)."""
    def rbf_kernel(x, y, gamma):
        x_sq = np.sum(x ** 2, axis=1, keepdims=True)
        y_sq = np.sum(y ** 2, axis=1, keepdims=True)
        dist = x_sq + y_sq.T - 2 * x @ y.T
        return np.exp(-gamma * dist)

    k_ss = rbf_kernel(source_embeddings, source_embeddings, gamma).mean()
    k_tt = rbf_kernel(target_embeddings, target_embeddings, gamma).mean()
    k_st = rbf_kernel(source_embeddings, target_embeddings, gamma).mean()
    return float(k_ss + k_tt - 2 * k_st)


def plot_tsne(source_embeddings: np.ndarray, target_embeddings: np.ndarray, out_path: str) -> str:
    """Visualize source vs target embeddings in 2D. Requires scikit-learn + matplotlib."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    combined = np.vstack([source_embeddings, target_embeddings])
    labels = np.array(["source"] * len(source_embeddings) + ["target"] * len(target_embeddings))
    proj = TSNE(n_components=2, init="pca", random_state=42).fit_transform(combined)

    plt.figure(figsize=(6, 6))
    for label, color in [("source", "tab:blue"), ("target", "tab:orange")]:
        mask = labels == label
        plt.scatter(proj[mask, 0], proj[mask, 1], label=label, alpha=0.6, s=15, c=color)
    plt.legend()
    plt.title("Source vs Target embeddings (t-SNE)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# 7. W&B logging helpers
# ---------------------------------------------------------------------------

def log_dataset_config(run, dataset_configs: dict, source_name: str) -> None:
    """Summarize every dataset's label in run.config -- gives you a single
    place (the W&B Overview tab) that spells out what A, B, C (...N) mean,
    regardless of how many datasets you're running with."""
    config_update = {
        "source_dataset_name": source_name,
        "source_dataset_label": dataset_configs[source_name]["label"],
        "target_dataset_names": [name for name in dataset_configs if name != source_name],
    }
    for name, cfg in dataset_configs.items():
        config_update[f"dataset_{name}_label"] = cfg["label"]
    run.config.update(config_update)


def log_target_scalars(run, target_name: str, result: dict) -> None:
    """Log final numeric metrics to run.summary (not run.log) -- these are
    single final values, not a time series, so summary is the semantically
    correct place. This is what makes them sortable/filterable columns when
    comparing this run against other training runs in the project."""
    for key, value in result.items():
        if isinstance(value, (int, float)):
            run.summary[f"{target_name}/{key}"] = value


def log_target_plots(run, target_name: str, plots_dir: Path) -> None:
    """Log the confusion matrix / PR / F1 / P / R curves and val batch mosaics
    that model.val(plots=True) wrote to disk, under eval/{target}/plots/*."""
    if plots_dir is None or not plots_dir.exists():
        return
    image_paths = sorted(plots_dir.glob("*.png")) + sorted(plots_dir.glob("*.jpg"))
    log_dict = {f"{target_name}/plots/{p.stem}": wandb.Image(str(p)) for p in image_paths}
    if log_dict:
        run.log(log_dict)


def log_results_table(run, results_list: list, table_name: str = "results_table") -> None:
    """One browsable table for all targets in this run (handles array-valued
    columns like per-class precision/recall that summary scalars can't)."""
    if not results_list:
        return
    keys = sorted(set().union(*[r.keys() for r in results_list]))
    table = wandb.Table(columns=keys, data=[[r.get(k) for k in keys] for r in results_list])
    run.log({table_name: table})


def save_results_csv(results_list: list, out_path: str) -> None:
    """Optional: write results to CSV for use outside W&B (e.g. pandas)."""
    keys = sorted(set().union(*[r.keys() for r in results_list]))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_list)


# ---------------------------------------------------------------------------
# 8. Orchestration: one source vs N targets, fully generalized
# ---------------------------------------------------------------------------

def evaluate_source_against_targets(
    run,
    model_path: str,
    dataset_configs: dict,
    source_name: str,
    target_names: list[str] | None = None,
    split: str = "test",
    result_path: str = "da_results",
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Runs the full metrics + embeddings + plots pipeline for source_name
    against every dataset in target_names (defaults to ALL datasets in
    dataset_configs, including source itself as the in-domain baseline).

    Works for any number of datasets -- just add entries to dataset_configs.
    """
    result_folder = Path(result_path)
    result_folder.mkdir(parents=True, exist_ok=True)
    target_names = target_names or list(dataset_configs.keys())

    log_dataset_config(run, dataset_configs, source_name)

    results = []
    embeddings_by_dataset = {}

    for target_name in target_names:
        cfg = dataset_configs[target_name]

        result, emb, paths, plots_dir = run_full_evaluation(
            model_path=model_path,
            data_yaml=cfg["yaml"],
            image_dir=cfg["images"],
            split=split,
            run_name=f"source{source_name}_to_{target_name}",
            iou_threshold=cfg.get("iou", 0.20),
            conf_threshold=cfg.get("conf", 0.20)
        )
        result["source_dataset_name"] = source_name
        result["source_dataset_label"] = dataset_configs[source_name]["label"]
        result["target_dataset_name"] = target_name
        result["target_dataset_label"] = cfg["label"]

        results.append(result)
        embeddings_by_dataset[target_name] = emb
        np.save(result_folder / f"embeddings_{source_name}_to_{target_name}.npy", emb)

        log_target_scalars(run, target_name, result)
        log_target_plots(run, target_name, plots_dir)

    log_results_table(run, results)

    csv_path = result_folder / f"source{source_name}_results.csv"
    save_results_csv(results, csv_path)
    csv_artifact = wandb.Artifact(f"source{source_name}_results", type="evaluation")
    csv_artifact.add_file(str(csv_path))
    run.log_artifact(csv_artifact)

    # Domain gap: source vs every other target (skips source-vs-itself)
    for target_name in target_names:
        if target_name == source_name:
            continue
        gap = compute_domain_gap_mmd(embeddings_by_dataset[source_name], embeddings_by_dataset[target_name])
        run.summary[f"{target_name}/domain_gap_mmd"] = gap

        tsne_path = plot_tsne(
            embeddings_by_dataset[source_name],
            embeddings_by_dataset[target_name],
            out_path=str(result_folder / f"tsne_{source_name}_vs_{target_name}.png"),
        )
        run.log({f"{target_name}/tsne": wandb.Image(tsne_path)})

    return results, embeddings_by_dataset


def run_eval(run_name:str, 
             dataset_configs: dict,
             project: str,
             source_name: str = 'A',
             wandb_model_path: str = None,
             local_model_path: str = None,
             result_folder: str = "da_results" ):
    wandb.login()
    run = wandb.init(
        project=project,
        job_type="eval",
        tags=["da_metrics"],
        name=f"eval_{run_name}",
        group=run_name,
    )
    if(wandb_model_path != None):
        artifact = run.use_artifact(wandb_model_path)
        SOURCE_MODEL = artifact.download() + "/best.pt"
    else:
        SOURCE_MODEL = local_model_path
    evaluate_source_against_targets(
        run=run,
        model_path=SOURCE_MODEL,
        dataset_configs=dataset_configs,
        source_name=source_name,
        result_path=result_folder,
    )

    run.finish()


# ---------------------------------------------------------------------------
# Example run -- adding a 4th dataset "D" only means adding one dict entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    entity = "your_wandb_entity_name"  # Change this to your W&B entity name
    project = "your_wandb_project_name"  # Change this to your W&B project name
    wbhelper = WandbHelper(entity=entity, project=project)

    dataset_configs = {
        "A": {
            "label": "SeaDronesSee_Juanpe",
            "yaml": "datasets/processed/sns_juanpe_yolov5/dataset.yaml",
            "images": "datasets/raw/SeaDronesSee_Juanpe/images/test",
            "iou": 0.20,
            "conf": 0.15,
            "train_run_name": "seadronesseejp_500_epochs",
        },
        "B": {
            "label": "synbase_yolov5",
            "yaml": "datasets/processed/synbase_yolov5/dataset.yaml",
            "images": "datasets/processed/synbase_yolov5/images/test",
            "iou": 0.20,
            "conf": 0.20,
            "train_run_name": "synbase_500_epochs", 
        },
        "C": {
            "label": "afo_humans_no_vehicle_060_yolov5",
            "yaml": "datasets/processed/afo_humans_no_vehicle_060_yolov5/dataset.yaml",
            "images": "datasets/processed/afo_humans_no_vehicle_060_yolov5/images/test",
            "iou": 0.20,
            "conf": 0.20,
            "train_run_name": "afo_humans_no_vehicle_060_500_epochs",
        },
        # Add a 4th dataset here later, e.g.:
        # "D": {"label": "...", "yaml": "...", "images": "..."},
    }

    for dataset, cfg in dataset_configs.items():
        run_name = cfg["train_run_name"]
        run_id = wbhelper.get_run_id_by_name(run_name=run_name)
        wandb_model_path = wbhelper.get_ultralytics_best_model_path(run_id=run_id)
        source_name = dataset  # Change this to whichever dataset is your source
        
        run_eval(run_name=run_name,
             dataset_configs=dataset_configs,
             project=project,
             wandb_model_path=wandb_model_path,
             source_name=source_name)