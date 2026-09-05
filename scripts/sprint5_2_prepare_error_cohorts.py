"""Lock the Sprint 5.2 protocol, threshold, predictions, and error cohorts.

Validation is processed first. The experiment contract is written before test
predictions are computed. This script does not create or modify notebooks/reports.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Iterable

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sprint5_explainer_feasibility as runtime
import sprint5_explainer_protocol_and_evaluation as explainer_protocol


SCHEMA_VERSION = 1
MODEL_SEED = 42
VALIDATION_SEED = 10_042
TEST_SEED = 20_042
NEIGHBORHOOD_SEED = 42
TARGET_LIMIT = 20
SCORE_BIN_COUNT = 10
FORWARD_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-6

ROOT = runtime.PROJECT_ROOT
METRICS = ROOT / "artifacts" / "metrics"
FIGURES = ROOT / "artifacts" / "figures" / "sprint5_2"
LOCK_PATH = METRICS / "sprint5_2_protocol_lock.json"
PREDICTION_PATH = METRICS / "sprint5_2_predictions.csv.gz"
CURVE_PATH = METRICS / "sprint5_2_threshold_curve.csv.gz"
TARGET_PATH = METRICS / "sprint5_2_targets.csv"
VALIDATION_BATCH_PATH = METRICS / "sprint5_2_validation_target_batches.npz"
TEST_BATCH_PATH = METRICS / "sprint5_2_test_target_batches.npz"
SUMMARY_PATH = METRICS / "sprint5_2_preparation_summary.json"
FIGURE_PATH = FIGURES / "01_threshold_and_errors.png"

EXPECTED_DATA_HASH = "95470dab2c48523f7118a92204c090de37a957bb053bd5841c7bdba09558ba85"
EXPECTED_CHECKPOINT_HASH = "558db58e8c788edb8f5862b64093ef5a1a463debadd80e7e119d230ca602b61e"
EXPECTED_FEATURE_HASH = "d5d98446df501381cd39a269273340b13f964e7f3a7e598f7286ef1b626afea1"

def pandas_module():
    """Return pandas without importing it before the notebook runtime loads torch."""
    if hasattr(runtime, "pd"):
        return runtime.pd
    import pandas

    return pandas


def now() -> str:
    return datetime.now().astimezone().isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_hash(payload: dict) -> str:
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_hash(arrays: dict[str, np.ndarray]) -> str:
    """Hash logical arrays, independent of timestamps in the NPZ container."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def frame_hash(frame, columns: Iterable[str]) -> str:
    return arrays_hash({name: frame[name].to_numpy() for name in columns})


def save_csv(frame, path: Path, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "compresslevel": 9, "mtime": 0} if compressed else None
    frame.to_csv(path, index=False, compression=compression)


def predict_split(model, data, config: dict, split: str):
    torch, pd = runtime.torch, pandas_module()
    if split == "validation":
        input_nodes, seed = data.valid_idx, VALIDATION_SEED
    elif split == "test":
        input_nodes, seed = data.test_idx, TEST_SEED
    else:
        raise ValueError(split)
    loader = runtime.NeighborLoader(
        data=data,
        input_nodes=input_nodes,
        num_neighbors=config["sampling"]["num_neighbors"],
        batch_size=config["sampling"]["batch_size"],
        num_workers=config["sampling"]["num_workers"],
        subgraph_type="directional",
        transform=runtime.attach_node_relative_edge_time,
        shuffle=False,
    )
    runtime.set_seed(seed)
    node_ids, labels, logits, scores = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(runtime.DEVICE)
            n = int(batch.batch_size)
            batch_logits = model(
                batch.x, batch.edge_index, getattr(batch, "edge_type", None), batch.edge_delta
            )[:n]
            node_ids.append(batch.n_id[:n].cpu().numpy())
            labels.append(batch.y[:n].cpu().numpy())
            logits.append(batch_logits.cpu().numpy())
            scores.append(torch.sigmoid(batch_logits).cpu().numpy())
    return pd.DataFrame(
        {
            "split": split,
            "node_id": np.concatenate(node_ids).astype(np.int64),
            "label": np.concatenate(labels).astype(np.int64),
            "fraud_logit": np.concatenate(logits).astype(np.float64),
            "fraud_score": np.concatenate(scores).astype(np.float64),
        }
    )


def build_threshold_curve(labels: np.ndarray, scores: np.ndarray):
    """Evaluate every unique validation score as a >= threshold."""
    pd = pandas_module()
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("labels and scores must be non-empty aligned vectors")
    if not np.isfinite(scores).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary and scores must be finite")
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError("validation must contain positive labels")
    order = np.argsort(-scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(scores) - 1]
    predicted_positive = ends + 1
    tp = np.cumsum(ordered_labels, dtype=np.int64)[ends]
    fp, fn = predicted_positive - tp, positive_count - tp
    precision, recall = tp / predicted_positive, tp / positive_count
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return pd.DataFrame(
        {
            "threshold": ordered_scores[ends],
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def select_threshold(curve) -> dict:
    best_f1 = float(curve["f1"].max())
    candidates = curve[np.isclose(curve["f1"], best_f1, rtol=0, atol=1e-15)]
    best_recall = float(candidates["recall"].max())
    candidates = candidates[np.isclose(candidates["recall"], best_recall, rtol=0, atol=1e-15)]
    selected = candidates.sort_values("threshold", ascending=True, kind="stable").iloc[0]
    return {
        "threshold": float(selected["threshold"]),
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
        "f1": float(selected["f1"]),
        "true_positive": int(selected["true_positive"]),
        "false_positive": int(selected["false_positive"]),
        "false_negative": int(selected["false_negative"]),
        "candidate_count": int(len(curve)),
        "final_tie_count": int(len(candidates)),
    }


def add_context(frame, threshold: float, in_degree: np.ndarray):
    result = frame.copy()
    ids = result["node_id"].to_numpy(dtype=np.int64)
    scores = result["fraud_score"].to_numpy(dtype=np.float64)
    labels = result["label"].to_numpy(dtype=np.int64)
    predicted = (scores >= threshold).astype(np.int64)
    result["predicted_class"] = predicted
    result["cohort"] = np.select(
        [(labels == 1) & (predicted == 1), (labels == 0) & (predicted == 1),
         (labels == 1) & (predicted == 0), (labels == 0) & (predicted == 0)],
        ["TP", "FP", "FN", "TN"], default="INVALID"
    )
    if (result["cohort"] == "INVALID").any():
        raise AssertionError("cohort assignment failed")
    result["distance_to_threshold"] = np.abs(scores - threshold)
    result["in_degree"] = in_degree[ids]
    result["log1p_degree"] = np.log1p(result["in_degree"].to_numpy(dtype=np.float64))
    result["sampled_event_count"] = np.minimum(result["in_degree"], 15).astype(np.int64)
    result["score_bin"] = np.minimum((scores * SCORE_BIN_COUNT).astype(np.int64), 9)
    return result


def split_metrics(frame, threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    counts = frame["cohort"].value_counts().to_dict()
    tp, fp, fn, tn = (int(counts.get(name, 0)) for name in ("TP", "FP", "FN", "TN"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": int(len(frame)),
        "positive_count": int((frame["label"] == 1).sum()),
        "threshold": float(threshold),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "predicted_positive_rate": float((tp + fp) / len(frame)),
        "average_precision": float(average_precision_score(frame["label"], frame["fraud_score"])),
        "roc_auc": float(roc_auc_score(frame["label"], frame["fraud_score"])),
    }


def select_error_rows(frame, cohort: str):
    candidates = frame[frame["cohort"] == cohort].copy()
    n = min(TARGET_LIMIT, len(candidates))
    boundary_n = (n + 1) // 2
    boundary = candidates.sort_values(
        ["distance_to_threshold", "node_id"], ascending=[True, True], kind="stable"
    ).head(boundary_n)
    remaining = candidates[~candidates["node_id"].isin(boundary["node_id"])]
    confident = remaining.sort_values(
        ["distance_to_threshold", "node_id"], ascending=[False, True], kind="stable"
    ).head(n - boundary_n)
    return pandas_module().concat(
        (boundary.assign(selection_stratum="near_threshold"),
         confident.assign(selection_stratum="high_confidence")), ignore_index=True
    )


def match_control(error, controls, available: np.ndarray) -> tuple[int, dict]:
    positions = np.flatnonzero(available)
    candidates = controls.iloc[positions]
    bin_delta = np.abs(candidates["score_bin"].to_numpy() - int(error["score_bin"]))
    degree_delta = np.abs(candidates["log1p_degree"].to_numpy() - float(error["log1p_degree"]))
    event_delta = np.abs(candidates["sampled_event_count"].to_numpy() - int(error["sampled_event_count"]))
    score_delta = np.abs(candidates["fraud_score"].to_numpy() - float(error["fraud_score"]))
    node_ids = candidates["node_id"].to_numpy(dtype=np.int64)
    local = np.lexsort((node_ids, score_delta, event_delta, degree_delta, bin_delta))[0]
    position = int(positions[int(local)])
    control = controls.iloc[position]
    return position, {
        "match_score_bin_delta": int(abs(int(control["score_bin"]) - int(error["score_bin"]))),
        "match_score_delta": float(abs(float(control["fraud_score"]) - float(error["fraud_score"]))),
        "match_log1p_degree_delta": float(abs(float(control["log1p_degree"]) - float(error["log1p_degree"]))),
        "match_event_count_delta": int(abs(int(control["sampled_event_count"]) - int(error["sampled_event_count"]))),
    }


def select_targets(frame, split: str):
    rows = []
    for error_cohort, control_cohort in (("FP", "TP"), ("FN", "TN")):
        errors = select_error_rows(frame, error_cohort)
        controls = frame[frame["cohort"] == control_cohort].copy().reset_index(drop=True)
        errors = errors.head(min(len(errors), len(controls)))
        available = np.ones(len(controls), dtype=bool)
        for order, (_, error) in enumerate(errors.iterrows()):
            position, match = match_control(error, controls, available)
            available[position] = False
            control = controls.iloc[position]
            pair_id = f"{split}_{error_cohort.lower()}_{order:02d}"
            for role, source, other in (
                ("error", error, control), ("control", control, error)
            ):
                rows.append(
                    {
                        "split": split, "pair_id": pair_id, "pair_order": order,
                        "error_cohort": error_cohort, "target_role": role,
                        "cohort": str(source["cohort"]),
                        "selection_stratum": str(error["selection_stratum"]) if role == "error" else f"matched_{error['selection_stratum']}",
                        "node_id": int(source["node_id"]), "matched_node_id": int(other["node_id"]),
                        "label": int(source["label"]), "predicted_class": int(source["predicted_class"]),
                        "fraud_logit": float(source["fraud_logit"]), "fraud_score": float(source["fraud_score"]),
                        "distance_to_threshold": float(source["distance_to_threshold"]),
                        "in_degree": int(source["in_degree"]), "log1p_degree": float(source["log1p_degree"]),
                        "sampled_event_count": int(source["sampled_event_count"]), "score_bin": int(source["score_bin"]),
                        "batch_key": f"{split}_target_{int(source['node_id'])}", **match,
                    }
                )
    result = pandas_module().DataFrame(rows)
    if not result.empty:
        if result[["split", "node_id"]].duplicated().any():
            raise AssertionError(f"duplicate selected node in {split}")
        if not (result.groupby("pair_id").size() == 2).all():
            raise AssertionError(f"invalid matching pairs in {split}")
    return result


def materialize_batches(objects: dict, targets, split: str) -> dict:
    torch, arrays, differences, counts = runtime.torch, {}, [], {}
    for row in targets.sort_values("node_id").to_dict("records"):
        node_id = int(row["node_id"])
        batch = explainer_protocol.materialize_local_batch(objects["data"], node_id, NEIGHBORHOOD_SEED)
        prefix = f"{split}_target_{node_id}"
        with torch.inference_mode():
            direct = float(objects["model"](
                batch.x, batch.edge_index, getattr(batch, "edge_type", None), batch.edge_delta
            )[0].cpu())
            adapted = float(objects["adapter"](batch.x, batch.edge_index, batch.edge_delta)[0].cpu())
        differences.append(abs(direct - adapted))
        counts[node_id] = int(batch.edge_index.shape[1])
        for name in ("x", "edge_index", "edge_delta", "edge_timestamp", "edge_type", "n_id"):
            arrays[f"{prefix}_{name}"] = getattr(batch, name).detach().cpu().numpy()
        if hasattr(batch, "e_id"):
            arrays[f"{prefix}_e_id"] = batch.e_id.detach().cpu().numpy()
        arrays[f"{prefix}_batch_size"] = np.asarray([int(batch.batch_size)], dtype=np.int64)
        del batch
    gc.collect()
    expected = {int(row.node_id): int(row.sampled_event_count) for row in targets.itertuples()}
    return {
        "arrays": arrays,
        "logical_content_sha256": arrays_hash(arrays),
        "max_adapter_logit_abs_difference": float(max(differences, default=0)),
        "event_counts_match": expected == counts,
    }


def save_batches(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload["arrays"])


def build_lock_payload(objects, threshold, validation_metrics, validation_targets, validation_batches, curve_hash):
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "sprint5_2_error_analysis_and_tgat_input_ablation",
        "data": {
            "path": runtime.DATA_PATH.relative_to(ROOT).as_posix(), "sha256": objects["data_hash"],
            "community_feature_path": runtime.FEATURE_PATH.relative_to(ROOT).as_posix(),
            "community_feature_sha256": objects["feature_manifest"]["feature_sha256"],
            "community_assignment_sha256": objects["feature_manifest"]["assignment_sha256"],
        },
        "model": {
            "variant": "C", "seed": MODEL_SEED,
            "checkpoint_path": objects["checkpoint_path"].relative_to(ROOT).as_posix(),
            "checkpoint_sha256": objects["checkpoint_lock"]["sha256"],
            "graph": "temporal_event_mirror", "task": "full_history_transductive_node_classification",
            "eval_mode": True, "weights_frozen": True,
            "community_risk_integration": "after_message_passing_seed_output_only",
        },
        "prediction_protocol": {
            "validation_seed": VALIDATION_SEED, "test_seed": TEST_SEED,
            "num_neighbors": objects["config"]["sampling"]["num_neighbors"],
            "subgraph_type": "directional", "shuffle": False,
            "decision_operator": "fraud_score >= threshold", "forward_tolerance": FORWARD_TOLERANCE,
        },
        "threshold_selection": {
            "source_split": "validation", "objective": "maximize fraud-class F1 over every unique validation score",
            "tie_break": "higher recall, then lower threshold", "selected": threshold,
            "validation_metrics": validation_metrics, "curve_path": CURVE_PATH.relative_to(ROOT).as_posix(),
            "curve_logical_content_sha256": curve_hash,
        },
        "cohort_definitions": {
            "TP": "label=1 and fraud_score>=threshold", "FP": "label=0 and fraud_score>=threshold",
            "FN": "label=1 and fraud_score<threshold", "TN": "label=0 and fraud_score<threshold",
        },
        "target_selection": {
            "error_cohorts": ["FP", "FN"], "controls": {"FP": "TP", "FN": "TN"},
            "maximum_errors_per_cohort_per_split": TARGET_LIMIT,
            "error_strata": "ceil(n/2) nearest threshold, remainder farthest; node_id tie-break",
            "matching_without_replacement": True, "score_bins": SCORE_BIN_COUNT,
            "matching_order": ["score-bin difference", "log1p-degree difference", "sampled-event-count difference", "fraud-score difference", "node_id"],
            "selection_inputs": ["label", "fraud_score", "in_degree", "sampled_event_count", "node_id"],
            "validation_target_node_ids": validation_targets["node_id"].astype(int).tolist(),
            "validation_batch_path": VALIDATION_BATCH_PATH.relative_to(ROOT).as_posix(),
            "validation_batch_logical_content_sha256": validation_batches["logical_content_sha256"],
        },
        "input_ablation_protocol": {
            "FULL": "no intervention", "T-CONST": "replace local edge_delta with local median",
            "T-PERM": "permute local edge_delta", "N-BASE": "replace non-target neighbor features with train baselines",
            "N-PERM": "permute non-target neighbor feature vectors",
            "MP-OFF": "remove neighbor messages while retaining target/root and risk paths",
            "paired_bootstrap_repetitions": 1000,
        },
        "test_policy": {
            "test_previously_opened_in_sprint4_and_sprint5": True,
            "test_predictions_after_this_lock_only": True,
            "test_not_used_to_change_threshold_targets_or_protocol": True,
            "interpretation": "replication check, not a new blind holdout",
        },
    }


def write_or_verify_lock(payload: dict, force: bool) -> tuple[dict, bool]:
    payload_sha = canonical_hash(payload)
    if LOCK_PATH.exists() and not force:
        existing = load_json(LOCK_PATH)
        if existing.get("lock_payload_sha256") != payload_sha:
            raise AssertionError("Existing protocol lock differs; review and use --force-relock only intentionally")
        return existing, False
    lock = {
        "schema_version": SCHEMA_VERSION, "status": "locked_before_test_predictions",
        "locked_at": now(), "lock_payload_sha256": payload_sha, "lock_payload": payload,
    }
    save_json(LOCK_PATH, lock)
    return lock, True


def make_figure(curve, threshold, results) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    positions = np.linspace(0, len(curve) - 1, min(5000, len(curve)), dtype=np.int64)
    plotted = curve.iloc[np.unique(positions)]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].plot(plotted["recall"], plotted["precision"])
    axes[0, 0].scatter([threshold["recall"]], [threshold["precision"]], color="red")
    axes[0, 0].set(xlabel="Recall", ylabel="Precision", title="Validation precision–recall", xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].plot(plotted["threshold"], plotted["f1"])
    axes[0, 1].axvline(threshold["threshold"], color="red", linestyle="--")
    axes[0, 1].set(xlabel="Fraud-score threshold", ylabel="F1", title=f"Locked threshold = {threshold['threshold']:.6f}", xlim=(0, 1), ylim=(0, 1))
    for axis, split in zip(axes[1], ("validation", "test"), strict=True):
        c = results[split]["confusion_matrix"]
        matrix = np.asarray([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                color = "white" if matrix[row, column] > matrix.max() / 2 else "#222222"
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", color=color)
        axis.set_xticks([0, 1], ["Pred normal", "Pred fraud"])
        axis.set_yticks([0, 1], ["True normal", "True fraud"])
        axis.set_title(f"{split.capitalize()} confusion matrix")
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def run(force_relock: bool = False) -> dict:
    print("Load locked TGAT + community-risk...", flush=True)
    runtime.load_training_pipeline()
    pd = runtime.pd
    objects = runtime.build_explainer_objects()
    state_before = runtime.model_state_sha256(objects["model"])
    hash_checks = {
        "dataset_hash_matches_plan": objects["data_hash"] == EXPECTED_DATA_HASH,
        "checkpoint_hash_matches_plan": objects["checkpoint_lock"]["sha256"] == EXPECTED_CHECKPOINT_HASH and file_hash(objects["checkpoint_path"]) == EXPECTED_CHECKPOINT_HASH,
        "community_feature_hash_matches_plan": objects["feature_manifest"]["feature_sha256"] == EXPECTED_FEATURE_HASH,
    }
    if not all(hash_checks.values()):
        raise AssertionError("locked input hash changed")
    in_degree = runtime.torch.bincount(
        objects["data"].edge_index[1], minlength=objects["data"].num_nodes
    ).cpu().numpy().astype(np.int64)

    print("Reproduce validation and lock selection rules...", flush=True)
    validation_raw = predict_split(objects["model"], objects["data"], objects["config"], "validation")
    curve = build_threshold_curve(validation_raw["label"].to_numpy(), validation_raw["fraud_score"].to_numpy())
    threshold = select_threshold(curve)
    validation = add_context(validation_raw, threshold["threshold"], in_degree)
    validation_metrics = split_metrics(validation, threshold["threshold"])
    validation_targets = select_targets(validation, "validation")
    validation_batches = materialize_batches(objects, validation_targets, "validation")
    save_batches(validation_batches, VALIDATION_BATCH_PATH)
    curve_columns = ["threshold", "true_positive", "false_positive", "false_negative", "precision", "recall", "f1"]
    curve_sha = frame_hash(curve, curve_columns)
    save_csv(curve, CURVE_PATH, compressed=True)
    payload = build_lock_payload(objects, threshold, validation_metrics, validation_targets, validation_batches, curve_sha)
    lock, created = write_or_verify_lock(payload, force_relock)
    lock_sha_before_test = file_hash(LOCK_PATH)
    test_started_at = now()

    print("Protocol locked; compute test predictions without changing it...", flush=True)
    test_raw = predict_split(objects["model"], objects["data"], objects["config"], "test")
    test = add_context(test_raw, threshold["threshold"], in_degree)
    test_metrics = split_metrics(test, threshold["threshold"])
    test_targets = select_targets(test, "test")
    test_batches = materialize_batches(objects, test_targets, "test")
    save_batches(test_batches, TEST_BATCH_PATH)
    predictions = pd.concat((validation, test), ignore_index=True)
    targets = pd.concat((validation_targets, test_targets), ignore_index=True)
    save_csv(predictions, PREDICTION_PATH, compressed=True)
    save_csv(targets, TARGET_PATH)
    results = {"validation": validation_metrics, "test": test_metrics}
    make_figure(curve, threshold, results)

    expected_validation, expected_test = objects["stored_metric"]["validation"], objects["stored_metric"]["test"]
    reproduction = {
        "validation": {
            "average_precision_absolute_difference": abs(validation_metrics["average_precision"] - float(expected_validation["average_precision"])),
            "roc_auc_absolute_difference": abs(validation_metrics["roc_auc"] - float(expected_validation["roc_auc"])),
        },
        "test": {
            "average_precision_absolute_difference": abs(test_metrics["average_precision"] - float(expected_test["average_precision"])),
            "roc_auc_absolute_difference": abs(test_metrics["roc_auc"] - float(expected_test["roc_auc"])),
        },
    }
    state_after = runtime.model_state_sha256(objects["model"])
    cohort_counts = {
        split: {name: int((frame["cohort"] == name).sum()) for name in ("TP", "FP", "FN", "TN")}
        for split, frame in (("validation", validation), ("test", test))
    }
    target_counts = {
        split: {name: int((frame["cohort"] == name).sum()) for name in ("TP", "FP", "FN", "TN")}
        for split, frame in (("validation", validation_targets), ("test", test_targets))
    }
    gates = {
        **hash_checks,
        "validation_ap_reproduced": reproduction["validation"]["average_precision_absolute_difference"] <= METRIC_TOLERANCE,
        "validation_roc_auc_reproduced": reproduction["validation"]["roc_auc_absolute_difference"] <= METRIC_TOLERANCE,
        "test_ap_reproduced": reproduction["test"]["average_precision_absolute_difference"] <= METRIC_TOLERANCE,
        "test_roc_auc_reproduced": reproduction["test"]["roc_auc_absolute_difference"] <= METRIC_TOLERANCE,
        "protocol_locked_before_test_predictions": lock["locked_at"] <= test_started_at and lock_sha_before_test == file_hash(LOCK_PATH),
        "threshold_uses_validation_only": lock["lock_payload"]["threshold_selection"]["source_split"] == "validation",
        "validation_counts_match_checkpoint": len(validation) == int(expected_validation["sample_count"]) and int((validation["label"] == 1).sum()) == int(expected_validation["positive_count"]),
        "test_counts_match_checkpoint": len(test) == int(expected_test["sample_count"]) and int((test["label"] == 1).sum()) == int(expected_test["positive_count"]),
        "prediction_nodes_unique": not predictions[["split", "node_id"]].duplicated().any(),
        "cohorts_exhaustive": all(sum(values.values()) == int((predictions["split"] == split).sum()) for split, values in cohort_counts.items()),
        "selected_targets_unique": not targets[["split", "node_id"]].duplicated().any(),
        "matched_pairs_valid": all(sorted(group["target_role"].tolist()) == ["control", "error"] for _, group in targets.groupby(["split", "pair_id"])),
        "target_limit_respected": all(int(((targets["split"] == split) & (targets["cohort"] == cohort) & (targets["target_role"] == "error")).sum()) <= TARGET_LIMIT for split in ("validation", "test") for cohort in ("FP", "FN")),
        "validation_adapter_equivalence": validation_batches["max_adapter_logit_abs_difference"] <= FORWARD_TOLERANCE,
        "test_adapter_equivalence": test_batches["max_adapter_logit_abs_difference"] <= FORWARD_TOLERANCE,
        "validation_batch_event_counts_match": validation_batches["event_counts_match"],
        "test_batch_event_counts_match": test_batches["event_counts_match"],
        "model_state_unchanged": state_before == state_after,
        "required_artifacts_exist": all(path.exists() for path in (LOCK_PATH, PREDICTION_PATH, CURVE_PATH, TARGET_PATH, VALIDATION_BATCH_PATH, TEST_BATCH_PATH, FIGURE_PATH)),
    }
    complete = all(gates.values())
    artifacts = {}
    for name, path in {
        "predictions": PREDICTION_PATH, "threshold_curve": CURVE_PATH, "targets": TARGET_PATH,
        "validation_target_batches": VALIDATION_BATCH_PATH, "test_target_batches": TEST_BATCH_PATH,
        "threshold_figure": FIGURE_PATH,
    }.items():
        artifacts[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)}
    artifacts["predictions"]["row_count"] = int(len(predictions))
    artifacts["threshold_curve"]["row_count"] = int(len(curve))
    artifacts["threshold_curve"]["logical_content_sha256"] = curve_sha
    artifacts["targets"].update(row_count=int(len(targets)), pair_count=int(len(targets) // 2))
    artifacts["validation_target_batches"]["logical_content_sha256"] = validation_batches["logical_content_sha256"]
    artifacts["test_target_batches"]["logical_content_sha256"] = test_batches["logical_content_sha256"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "preparation_complete" if complete else "preparation_needs_attention",
        "gate_passed": complete, "completed_at": now(), "test_predictions_started_at": test_started_at,
        "protocol": {"path": LOCK_PATH.relative_to(ROOT).as_posix(), "created_in_this_run": created,
                     "file_sha256": file_hash(LOCK_PATH), "payload_sha256": lock["lock_payload_sha256"]},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "torch": runtime.torch.__version__, "torch_geometric": runtime.torch_geometric.__version__,
                        "device": str(runtime.DEVICE), "omp_num_threads": os.environ["OMP_NUM_THREADS"]},
        "threshold": threshold, "reproduction": reproduction, "split_metrics": results,
        "cohort_counts": cohort_counts, "selected_target_counts": target_counts,
        "artifacts": artifacts, "gate_checks": gates,
    }
    save_json(SUMMARY_PATH, summary)
    print(
        f"Preparation {'PASSED' if complete else 'NEEDS ATTENTION'} | threshold={threshold['threshold']:.8f} | "
        f"validation FP/FN={cohort_counts['validation']['FP']}/{cohort_counts['validation']['FN']} | "
        f"test FP/FN={cohort_counts['test']['FP']}/{cohort_counts['test']['FN']}", flush=True
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-relock", action="store_true", help="Intentionally replace an existing protocol lock")
    return parser.parse_args()


if __name__ == "__main__":
    run(force_relock=parse_args().force_relock)
