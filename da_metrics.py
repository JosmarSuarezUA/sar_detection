"""
da_metrics.py

Utility functions to extract domain-adaptation-relevant metrics from an
Ultralytics YOLO model: mAP / precision / recall, confidence mean, and
embeddings. Designed to be run once per (model, source, target) combination
in a source -> target experimental matrix.

Usage pattern:
    1. Train on Source X  -> best.pt
    2. For each target Y in {A, B, C}:
         result, embeddings, paths = run_full_evaluation(...)
    3. Collect all results into a table with save_results_csv()
    4. Compare a source-only run vs an adapted run with compute_improvement()
    5. Use embeddings + compute_domain_gap() / plot_tsne() to visualize/quantify
       how close source and target feature distributions are.
"""

import csv
from pathlib import Path

import numpy as np
from tools.iou20_detectionvalidator import IoU20DetectionValidator
from ultralytics import YOLO
import wandb


# ---------------------------------------------------------------------------
# 1. Detection metrics: mAP, precision, recall
# ---------------------------------------------------------------------------

def evaluate_detection_metrics(
    model: YOLO,
    data_yaml: str,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate detection metrics over IoU thresholds from 0.20 to 0.95.

    Args:
        model: An Ultralytics YOLO model.
        data_yaml: Path to the dataset YAML file.
        split: Dataset split to evaluate, such as 'val' or 'test'.

    Returns:
        A dictionary containing mAP and precision/recall metrics.
    """
    metrics = model.val(
        data=data_yaml,
        split=split,
        verbose=False,
        validator=IoU20DetectionValidator,
    )

    # all_ap has shape:
    # (number_of_classes_with_targets, number_of_iou_thresholds)
    all_ap = np.asarray(metrics.box.all_ap)

    if all_ap.size == 0:
        map20_95 = 0.0
        map20 = 0.0
        map50 = 0.0
        map75 = 0.0
        map95 = 0.0
    else:
        # Threshold indices for:
        # 0.20 -> 0, 0.50 -> 6, 0.75 -> 11, 0.95 -> 15
        map20_95 = float(all_ap.mean())
        map50_95 = float(all_ap[:, 6:].mean())
        map20 = float(all_ap[:, 0].mean())
        map50 = float(all_ap[:, 6].mean())
        map75 = float(all_ap[:, 11].mean())
        map95 = float(all_ap[:, 15].mean())

    return {
        "map20_95": map20_95,
        "map50_95": map50_95,
        "map20": map20,
        "map50": map50,
        "map75": map75,
        "map95": map95,
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "precision_per_class": metrics.box.p.tolist(),
        "recall_per_class": metrics.box.r.tolist(),
    }

# ---------------------------------------------------------------------------
# 2. Confidence mean
# ---------------------------------------------------------------------------

def compute_confidence_mean(model: YOLO, image_dir: str, conf_threshold: float = 0.001) -> dict:
    """Average (and std) confidence of predictions on a folder of images.

    conf_threshold is set very low so we capture the model's raw confidence
    behavior rather than only the predictions it's already sure about --
    this matters for comparing confidence *distributions* across domains.
    """
    results = model.predict(image_dir, conf=conf_threshold, verbose=False)
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
    image_paths = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix.lower() in extensions
    )
    embeddings = []
    for img_path in image_paths:
        emb = model.embed(str(img_path), verbose=False)[0]  # returns a list of tensors
        embeddings.append(emb.cpu().numpy())

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
) -> tuple:
    """Runs detection metrics + confidence + embeddings for one eval and
    returns (metrics_dict, embeddings_array, image_paths)."""
    model = YOLO(model_path)

    det_metrics = evaluate_detection_metrics(model, data_yaml, split=split)
    conf_metrics = compute_confidence_mean(model, image_dir)
    embeddings, image_paths = extract_embeddings(model, image_dir)

    result = {
        "run_name": run_name,
        "model_path": model_path,
        "data_yaml": data_yaml,
        "split": split,
        **det_metrics,
        **conf_metrics,
    }
    return result, embeddings, image_paths


# ---------------------------------------------------------------------------
# 5. "Improved precision / recall" -- delta vs a baseline run
# ---------------------------------------------------------------------------

def compute_improvement(baseline_result: dict, adapted_result: dict) -> dict:
    """Compare an adapted-model run against a source-only baseline run on the
    SAME target dataset/split. This is what "improved precision/recall" means
    in most DA papers: not the raw value, but the delta over naive transfer."""
    def _delta(key):
        b, a = baseline_result.get(key), adapted_result.get(key)
        if b is None or a is None:
            return None
        return a - b

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
    """Simple RBF-kernel Maximum Mean Discrepancy between two sets of embeddings.
    Lower = source and target feature distributions are closer (less domain gap).
    """
    def rbf_kernel(x, y, gamma):
        x_sq = np.sum(x ** 2, axis=1, keepdims=True)
        y_sq = np.sum(y ** 2, axis=1, keepdims=True)
        dist = x_sq + y_sq.T - 2 * x @ y.T
        return np.exp(-gamma * dist)

    k_ss = rbf_kernel(source_embeddings, source_embeddings, gamma).mean()
    k_tt = rbf_kernel(target_embeddings, target_embeddings, gamma).mean()
    k_st = rbf_kernel(source_embeddings, target_embeddings, gamma).mean()
    return float(k_ss + k_tt - 2 * k_st)


def plot_tsne(source_embeddings: np.ndarray, target_embeddings: np.ndarray, out_path: str = "tsne_domain_gap.png"):
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
# 7. Saving results
# ---------------------------------------------------------------------------

def save_results_csv(results_list: list, out_path: str):
    """Dump a list of metrics dicts (from run_full_evaluation or
    compute_improvement) to a single CSV -- one row per run."""
    keys = sorted(set().union(*[r.keys() for r in results_list]))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results_list:
            writer.writerow(r)


def log_results_table(run, results_list: list, table_name: str = "evaluation_results"):
    """Log results as a W&B table."""
    if not results_list:
        return

    keys = sorted(set().union(*[r.keys() for r in results_list]))
    data = [[r.get(k) for k in keys] for r in results_list]
    table = wandb.Table(columns=keys, data=data)
    run.log({table_name: table})


# ---------------------------------------------------------------------------
# Example: Source A -> Targets A, B, C  (repeat this block per model)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wandb.login()
    run = wandb.init(
        project="sar_detection",
        job_type="eval",
        name="test_eval",
        group="SeaDronesSee filtered Juanpe",
    )
    # artifact = run.use_artifact(f"{model_name}_source_{source_name}:latest")
    artifact = run.use_artifact("run_seadronesseejp_100_epochs-2_20260710_112028_model:best")
    # SOURCE_MODEL = "runs/detect/sar_detection/seadronesseejp_100_epochs-2/weights/best.pt"
    SOURCE_MODEL = artifact.download() + "/best.pt"
    dataset_configs = {
        "A": {
            "label": "SeaDronesSee_Juanpe",
            "yaml": "datasets/processed/sns_juanpe_yolov5/dataset.yaml",
            "images": "datasets/raw/SeaDronesSee_Juanpe/images/test",
        },
        "B": {
            "label": "synbase_yolov5",
            "yaml": "datasets/processed/synbase_yolov5/dataset.yaml",
            "images": "datasets/processed/synbase_yolov5/images/test",
        },
        "C": {
            "label": "afo_humans_no_vehicle_060_yolov5",
            "yaml": "datasets/processed/afo_humans_no_vehicle_060_yolov5/dataset.yaml",
            "images": "datasets/processed/afo_humans_no_vehicle_060_yolov5/images/test",
        },
    }

    results = []
    embeddings_by_dataset = {}
    result_folder = Path("da_results")
    result_folder.mkdir(exist_ok=True)

    source_name = "A"
    source_label = dataset_configs[source_name]["label"]
    run.config.update({
        "source_dataset_name": source_name,
        "source_dataset_label": source_label,
        "dataset_A_label": dataset_configs["A"]["label"],
        "dataset_B_label": dataset_configs["B"]["label"],
        "dataset_C_label": dataset_configs["C"]["label"],
    })

    for target_name, cfg in dataset_configs.items():
        yaml_path = cfg["yaml"]
        img_dir = cfg["images"]
        target_label = cfg["label"]

        result, emb, paths = run_full_evaluation(
            model_path=SOURCE_MODEL,
            data_yaml=yaml_path,
            image_dir=img_dir,
            split="test",
            run_name=f"source{source_name}_to_{target_name}",
        )
        result["source_dataset_name"] = source_name
        result["source_dataset_label"] = source_label
        result["target_dataset_name"] = target_name
        result["target_dataset_label"] = target_label

        results.append(result)
        embeddings_by_dataset[target_name] = emb
        np.save(result_folder / f"embeddings_sourceA_to_{target_name}.npy", emb)

        wandb.log({f"{target_name}/{k}": v for k, v in result.items() if isinstance(v, (int, float))})

    csv_path = result_folder / "sourceA_results.csv"
    save_results_csv(results, csv_path)
    log_results_table(run, results, table_name="sourceA_results_table")

    csv_artifact = wandb.Artifact("sourceA_results", type="evaluation")
    csv_artifact.add_file(str(csv_path))
    run.log_artifact(csv_artifact)

    domain_gap_values = {}
    for target_name in ["B", "C"]:
        gap = compute_domain_gap_mmd(embeddings_by_dataset["A"], embeddings_by_dataset[target_name])
        domain_gap_values[f"domain_gap_A_to_{target_name}"] = gap
        print(f"MMD domain gap A -> {target_name}: {gap:.4f}")

    run.log(domain_gap_values)

    tsne_b_path = plot_tsne(embeddings_by_dataset["A"], embeddings_by_dataset["B"], out_path=str(result_folder / "tsne_A_vs_B.png"))
    tsne_c_path = plot_tsne(embeddings_by_dataset["A"], embeddings_by_dataset["C"], out_path=str(result_folder / "tsne_A_vs_C.png"))
    run.log({
        "tsne_A_vs_B": wandb.Image(tsne_b_path),
        "tsne_A_vs_C": wandb.Image(tsne_c_path),
    })

    run.finish()