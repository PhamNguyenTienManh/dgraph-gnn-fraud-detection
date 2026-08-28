"""Lock the explainer protocol, explain all cohorts, and audit robustness."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import gc
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np
import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import sprint5_explainer_feasibility as p1


PROTOCOL_VERSION = 1
PRIMARY_MODEL_SEED = 42
EXPLAINER_SEEDS = [42, 43, 44]
MODEL_SEEDS = [42, 43, 44]
NEIGHBORHOOD_SEEDS = [42, 43, 44]
COHORT_SIZE = 10
AUDIT_SIZE_PER_COHORT = 3
FEATURE_TOP_K = 5
SUFFICIENCY_THRESHOLD = 0.05
FORCE_RERUN = os.getenv(
    "SPRINT5_EXPLAINER_FORCE", os.getenv("SPRINT5_PHASE23_FORCE", "0")
) == "1"

PROJECT_ROOT = p1.PROJECT_ROOT
METRIC_DIR = PROJECT_ROOT / "artifacts" / "metrics"
PROTOCOL_LOCK_PATH = METRIC_DIR / "sprint5_explainer_protocol_lock.json"
TARGET_PATH = METRIC_DIR / "sprint5_explanation_targets.csv"
RESULT_PATH = METRIC_DIR / "sprint5_explainer_results.json"
RAW_MASK_PATH = METRIC_DIR / "sprint5_explanations.npz"
MANIFEST_PATH = METRIC_DIR / "sprint5_explainer_manifest.json"
ASSIGNMENT_PATH = METRIC_DIR / "sprint4_community_assignments.npz"

TUNING_CONFIGS = [
    {"id": "epochs50_lr001_default", "epochs": 50, "lr": 0.01},
    {"id": "epochs100_lr001_default", "epochs": 100, "lr": 0.01},
    {"id": "epochs100_lr0005_default", "epochs": 100, "lr": 0.005},
    {"id": "epochs100_lr001_edge001", "epochs": 100, "lr": 0.01, "edge_size": 0.001},
]
EDGE_TOP_K_CANDIDATES = [1, 3, 5, 10]


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pairwise_jaccard(sets: list[set[int]]) -> list[float]:
    values = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            values.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
    return values


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(np.asarray(left, dtype=np.float64))
    right_rank = rankdata(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 1.0 if np.array_equal(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def pairwise_spearman(rows: list[np.ndarray]) -> list[float]:
    values = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            values.append(spearman(rows[left], rows[right]))
    return values


def target_class_score(logit: float, predicted_class: int) -> float:
    fraud_score = float(1.0 / (1.0 + np.exp(-logit)))
    return fraud_score if predicted_class == 1 else 1.0 - fraud_score


def score_model(adapter, batch, edge_positions=None) -> dict:
    torch = p1.torch
    if edge_positions is None:
        edge_index = batch.edge_index
        edge_delta = batch.edge_delta
    else:
        edge_positions = torch.as_tensor(
            edge_positions, dtype=torch.long, device=batch.edge_index.device
        )
        edge_index = batch.edge_index[:, edge_positions]
        edge_delta = batch.edge_delta[edge_positions]
    with torch.inference_mode():
        logit = float(adapter(batch.x, edge_index, edge_delta)[0].cpu())
    fraud_score = float(1.0 / (1.0 + np.exp(-logit)))
    predicted_class = int(logit > 0)
    return {
        "logit": logit,
        "fraud_score": fraud_score,
        "predicted_class": predicted_class,
        "target_class_score": target_class_score(logit, predicted_class),
    }


def materialize_local_batch(data, target_node: int, neighborhood_seed: int):
    torch = p1.torch
    loader = p1.NeighborLoader(
        data=data,
        input_nodes=torch.tensor([target_node], dtype=torch.long),
        num_neighbors=[15],
        batch_size=1,
        num_workers=0,
        subgraph_type="directional",
        transform=p1.attach_node_relative_edge_time,
        shuffle=False,
        generator=torch.Generator().manual_seed(neighborhood_seed),
    )
    batch = next(iter(loader)).to(p1.DEVICE)
    if int(batch.batch_size) != 1 or int(batch.n_id[0]) != target_node:
        raise AssertionError("Target node is not local index 0")
    if batch.edge_delta.shape != (batch.edge_index.shape[1],):
        raise AssertionError("Local batch is missing edge_delta")
    return batch


def split_predictions(model, data, config, split: str):
    torch = p1.torch
    pd = p1.pd
    if split == "validation":
        input_nodes = data.valid_idx
        evaluation_seed = PRIMARY_MODEL_SEED + 10_000
    elif split == "test":
        input_nodes = data.test_idx
        evaluation_seed = PRIMARY_MODEL_SEED + 20_000
    else:
        raise ValueError(split)
    loader = p1.NeighborLoader(
        data=data,
        input_nodes=input_nodes,
        num_neighbors=config["sampling"]["num_neighbors"],
        batch_size=config["sampling"]["batch_size"],
        num_workers=config["sampling"]["num_workers"],
        subgraph_type="directional",
        transform=p1.attach_node_relative_edge_time,
        shuffle=False,
    )
    p1.set_seed(evaluation_seed)
    node_ids, labels, scores = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(p1.DEVICE)
            seed_count = int(batch.batch_size)
            logits = model(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_type", None),
                batch.edge_delta,
            )[:seed_count]
            node_ids.append(batch.n_id[:seed_count].cpu().numpy())
            labels.append(batch.y[:seed_count].cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return pd.DataFrame(
        {
            "node_id": np.concatenate(node_ids).astype(np.int64),
            "label": np.concatenate(labels).astype(np.int64),
            "fraud_score": np.concatenate(scores).astype(np.float64),
        }
    )


def load_context_arrays(data) -> dict:
    torch = p1.torch
    in_degree = torch.bincount(
        data.edge_index[1], minlength=data.num_nodes
    ).cpu().numpy().astype(np.int64)
    with np.load(p1.FEATURE_PATH, allow_pickle=False) as archive:
        community_risk = np.asarray(
            archive["community_risk_feature"], dtype=np.float32
        )
        community_size_feature = np.asarray(
            archive["structural_features"], dtype=np.float32
        )[:, 0]
    with np.load(ASSIGNMENT_PATH, allow_pickle=False) as archive:
        assignment_node_ids = np.asarray(archive["node_id"], dtype=np.int64)
        community_id = np.asarray(archive["community_id"], dtype=np.int64)
    if not np.array_equal(
        assignment_node_ids, np.arange(data.num_nodes, dtype=np.int64)
    ):
        raise AssertionError("Community assignment node_id mismatch")
    return {
        "in_degree": in_degree,
        "community_risk": community_risk,
        "community_size_feature": community_size_feature,
        "community_id": community_id,
    }


def matched_low_score_controls(normal, high_normal, context) -> list[dict]:
    pool = normal[normal["fraud_score"] <= normal["fraud_score"].quantile(0.25)].copy()
    pool = pool[~pool["node_id"].isin(high_normal["node_id"])]
    pool_ids = pool["node_id"].to_numpy(dtype=np.int64)
    pool_features = np.column_stack(
        (
            np.log1p(context["in_degree"][pool_ids]),
            context["community_size_feature"][pool_ids],
        )
    ).astype(np.float64)
    scale = np.std(pool_features, axis=0)
    scale[scale == 0] = 1.0
    selected = []
    available = np.ones(len(pool), dtype=bool)
    for reference_id in high_normal["node_id"].to_numpy(dtype=np.int64):
        reference = np.asarray(
            [
                np.log1p(context["in_degree"][reference_id]),
                context["community_size_feature"][reference_id],
            ],
            dtype=np.float64,
        )
        distances = np.linalg.norm((pool_features - reference) / scale, axis=1)
        distances[~available] = np.inf
        position = int(np.argmin(distances))
        available[position] = False
        row = pool.iloc[position].to_dict()
        row["matched_reference_node_id"] = int(reference_id)
        row["matching_distance"] = float(distances[position])
        selected.append(row)
    return selected


def build_cohorts(frame, split: str, context):
    pd = p1.pd
    eligible = frame.copy()
    eligible["in_degree"] = context["in_degree"][eligible["node_id"].to_numpy()]
    fraud = eligible[eligible["label"] == 1]
    normal = eligible[eligible["label"] == 0]
    if len(fraud) < COHORT_SIZE * 2 or len(normal) < COHORT_SIZE * 2:
        raise AssertionError(f"Insufficient eligible nodes in {split}")
    high_fraud = fraud.nlargest(COHORT_SIZE, "fraud_score")
    high_normal = normal.nlargest(COHORT_SIZE, "fraud_score")
    low_fraud = fraud.nsmallest(COHORT_SIZE, "fraud_score")
    groups = [
        ("high_score_fraud", high_fraud.to_dict("records")),
        ("high_score_normal", high_normal.to_dict("records")),
        ("low_score_fraud", low_fraud.to_dict("records")),
        (
            "low_score_normal_control",
            matched_low_score_controls(normal, high_normal, context),
        ),
    ]
    rows = []
    for cohort, records in groups:
        for order, record in enumerate(records):
            node_id = int(record["node_id"])
            rows.append(
                {
                    "split": split,
                    "cohort": cohort,
                    "cohort_order": order,
                    "node_id": node_id,
                    "label": int(record["label"]),
                    "fraud_score": float(record["fraud_score"]),
                    "in_degree": int(context["in_degree"][node_id]),
                    "community_id": int(context["community_id"][node_id]),
                    "community_risk": float(context["community_risk"][node_id]),
                    "community_size_feature": float(
                        context["community_size_feature"][node_id]
                    ),
                    "matched_reference_node_id": record.get(
                        "matched_reference_node_id"
                    ),
                    "matching_distance": record.get("matching_distance"),
                    "edge_explanation_unavailable": bool(
                        context["in_degree"][node_id] == 0
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != COHORT_SIZE * 4 or result["node_id"].duplicated().any():
        raise AssertionError(f"Invalid {split} cohort")
    return result


def event_ids(batch) -> np.ndarray:
    if hasattr(batch, "e_id"):
        return batch.e_id.detach().cpu().numpy().astype(np.int64)
    source = batch.n_id[batch.edge_index[0]].detach().cpu().numpy().astype(np.int64)
    target = batch.n_id[batch.edge_index[1]].detach().cpu().numpy().astype(np.int64)
    timestamp = batch.edge_timestamp.detach().cpu().numpy().astype(np.int64)
    edge_type = batch.edge_type.detach().cpu().numpy().astype(np.int64)
    values = []
    for row in zip(source, target, timestamp, edge_type, strict=True):
        values.append(int.from_bytes(hashlib.sha256(repr(row).encode()).digest()[:8], "little"))
    return np.asarray(values, dtype=np.uint64)


def explain_mask(adapter, model, batch, explainer_seed: int, config: dict) -> dict:
    torch = p1.torch
    p1.set_seed(explainer_seed)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kwargs = {}
    if "edge_size" in config:
        kwargs["edge_size"] = config["edge_size"]
    has_edges = int(batch.edge_index.shape[1]) > 0
    algorithm = p1.gradient_tracking_gnnexplainer(
        epochs=int(config["epochs"]), lr=float(config["lr"]), **kwargs
    )
    explainer = Explainer(
        model=adapter,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object" if has_edges else None,
        model_config={
            "mode": "binary_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )
    started = time.perf_counter()
    rss_before = psutil.Process().memory_info().rss
    with p1.PeakRSSMonitor() as memory:
        explanation = explainer(
            batch.x, batch.edge_index, index=0, edge_delta=batch.edge_delta
        )
    node_mask = explanation.node_mask.detach().cpu().numpy().astype(np.float32)
    edge_mask = (
        explanation.edge_mask.detach().cpu().numpy().astype(np.float32)
        if has_edges
        else np.empty(0, dtype=np.float32)
    )
    if node_mask.shape != tuple(batch.x.shape):
        raise AssertionError("Node mask shape mismatch")
    if edge_mask.shape != (batch.edge_index.shape[1],):
        raise AssertionError("Edge mask shape mismatch")
    if not np.isfinite(node_mask).all() or not np.isfinite(edge_mask).all():
        raise AssertionError("Non-finite explanation mask")
    feature_importance = node_mask.max(axis=0)
    return {
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "feature_importance": feature_importance,
        "node_mask_has_nonzero_gradient": (
            algorithm.node_mask_has_nonzero_gradient
        ),
        "edge_mask_has_nonzero_gradient": (
            algorithm.edge_mask_has_nonzero_gradient if has_edges else None
        ),
        "runtime_seconds": time.perf_counter() - started,
        "rss_before_mib": rss_before / (1024**2),
        "peak_rss_mib": memory.peak_bytes / (1024**2),
    }


def evaluate_positions(adapter, batch, selected: np.ndarray, full: dict) -> dict:
    total_edges = int(batch.edge_index.shape[1])
    selected = np.asarray(selected, dtype=np.int64)
    removed = np.setdiff1d(np.arange(total_edges, dtype=np.int64), selected)
    keep_score = score_model(adapter, batch, selected)
    remove_score = score_model(adapter, batch, removed)
    predicted_class = int(full["predicted_class"])
    keep_target_score = target_class_score(keep_score["logit"], predicted_class)
    remove_target_score = target_class_score(remove_score["logit"], predicted_class)
    return {
        "selected_edge_count": int(len(selected)),
        "edge_sparsity": 1.0 - len(selected) / total_edges if total_edges else 1.0,
        "keep_logit": keep_score["logit"],
        "remove_logit": remove_score["logit"],
        "keep_fraud_score": keep_score["fraud_score"],
        "remove_fraud_score": remove_score["fraud_score"],
        "keep_target_class_score": keep_target_score,
        "remove_target_class_score": remove_target_score,
        "sufficiency_error": abs(full["target_class_score"] - keep_target_score),
        "comprehensiveness": full["target_class_score"] - remove_target_score,
    }


def baseline_positions(batch, top_k: int, degrees: np.ndarray, target_node: int) -> dict:
    total_edges = int(batch.edge_index.shape[1])
    k = min(top_k, total_edges)
    rng = np.random.default_rng(91_337 + target_node)
    random_positions = rng.choice(total_edges, size=k, replace=False)
    sources = batch.n_id[batch.edge_index[0]].detach().cpu().numpy().astype(np.int64)
    degree_positions = np.argsort(-degrees[sources], kind="stable")[:k]
    timestamps = batch.edge_timestamp.detach().cpu().numpy()
    recency_positions = np.argsort(-timestamps, kind="stable")[:k]
    return {
        "random": np.asarray(random_positions, dtype=np.int64),
        "degree": np.asarray(degree_positions, dtype=np.int64),
        "recency": np.asarray(recency_positions, dtype=np.int64),
    }


def explanation_evaluation(
    adapter,
    model,
    batch,
    target_node: int,
    explainer_seed: int,
    config: dict,
    top_k: int,
    degrees: np.ndarray,
    global_train_rate: float,
) -> tuple[dict, dict]:
    torch = p1.torch
    masks = explain_mask(adapter, model, batch, explainer_seed, config)
    full = score_model(adapter, batch)
    selected_count = min(top_k, len(masks["edge_mask"]))
    selected = np.argsort(-masks["edge_mask"], kind="stable")[:selected_count]
    fidelity = evaluate_positions(adapter, batch, selected, full)
    baselines = {
        name: evaluate_positions(adapter, batch, positions, full)
        for name, positions in baseline_positions(
            batch, top_k, degrees, target_node
        ).items()
    }
    feature_order = np.argsort(
        -masks["feature_importance"], kind="stable"
    )[:FEATURE_TOP_K]
    selected_feature_mask = np.zeros(len(masks["feature_importance"]), dtype=np.uint8)
    selected_feature_mask[feature_order] = 1
    selected_edge_mask = np.zeros(len(masks["edge_mask"]), dtype=np.uint8)
    selected_edge_mask[selected] = 1
    fidelity["selected_feature_count"] = int(selected_feature_mask.sum())
    fidelity["feature_sparsity"] = float(1.0 - selected_feature_mask.mean())
    risk_perturbed = batch.x.clone()
    risk_perturbed[0, 34] = global_train_rate
    with torch.inference_mode():
        risk_logit = float(
            adapter(risk_perturbed, batch.edge_index, batch.edge_delta)[0].cpu()
        )
    risk_score = float(1.0 / (1.0 + np.exp(-risk_logit)))
    ids = event_ids(batch)
    result = {
        "target_node_id": int(target_node),
        "label": int(batch.y[0]),
        "predicted_class": full["predicted_class"],
        "full_logit": full["logit"],
        "fraud_score": full["fraud_score"],
        "target_class_score": full["target_class_score"],
        "sampled_node_count": int(batch.num_nodes),
        "sampled_event_count": int(batch.edge_index.shape[1]),
        "edge_explanation_unavailable": bool(batch.edge_index.shape[1] == 0),
        "selected_event_ids": [int(value) for value in ids[selected]],
        "selected_event_positions": [int(value) for value in selected],
        "selected_feature_indices": [int(value) for value in feature_order],
        "selected_feature_importance": [
            float(masks["feature_importance"][value]) for value in feature_order
        ],
        "root_community_risk_importance": float(masks["node_mask"][0, 34]),
        "community_risk_prior_logit": risk_logit,
        "community_risk_prior_fraud_score": risk_score,
        "community_risk_prior_logit_delta": full["logit"] - risk_logit,
        "community_risk_prior_score_delta": full["fraud_score"] - risk_score,
        "model_seed": PRIMARY_MODEL_SEED,
        "explainer_seed": int(explainer_seed),
        "neighborhood_seed": NEIGHBORHOOD_SEEDS[0],
        "explainer_config_id": config["id"],
        "explainer_epochs": int(config["epochs"]),
        "explainer_learning_rate": float(config["lr"]),
        "edge_top_k": int(top_k),
        "feature_top_k": FEATURE_TOP_K,
        "node_mask_has_nonzero_gradient": masks[
            "node_mask_has_nonzero_gradient"
        ],
        "edge_mask_has_nonzero_gradient": masks[
            "edge_mask_has_nonzero_gradient"
        ],
        "runtime_seconds": masks["runtime_seconds"],
        "rss_before_mib": masks["rss_before_mib"],
        "peak_rss_mib": masks["peak_rss_mib"],
        "fidelity": fidelity,
        "baselines": baselines,
    }
    raw = {
        "node_ids": batch.n_id.detach().cpu().numpy().astype(np.int64),
        "event_ids": ids,
        "edge_index_global": batch.n_id[batch.edge_index.detach().cpu()]
        .numpy()
        .astype(np.int64),
        "edge_timestamp": batch.edge_timestamp.detach().cpu().numpy(),
        "edge_delta": batch.edge_delta.detach().cpu().numpy(),
        "edge_type": batch.edge_type.detach().cpu().numpy(),
        "node_mask": masks["node_mask"],
        "edge_mask": masks["edge_mask"],
        "feature_importance": masks["feature_importance"],
        "selected_edge_mask": selected_edge_mask,
        "selected_feature_mask": selected_feature_mask,
    }
    return result, raw


def aggregate_tuning(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["config_id"], row["top_k"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (config_id, top_k), values in groups.items():
        summary.append(
            {
                "config_id": config_id,
                "top_k": top_k,
                "node_count": len(values),
                "median_sufficiency_error": statistics.median(
                    row["sufficiency_error"] for row in values
                ),
                "median_comprehensiveness": statistics.median(
                    row["comprehensiveness"] for row in values
                ),
                "median_edge_sparsity": statistics.median(
                    row["edge_sparsity"] for row in values
                ),
                "mean_runtime_seconds": statistics.fmean(
                    row["runtime_seconds"] for row in values
                ),
            }
        )
    return summary


def select_tuning_winner(summary: list[dict]) -> dict:
    eligible = [
        row
        for row in summary
        if row["median_sufficiency_error"] <= SUFFICIENCY_THRESHOLD
    ]
    if eligible:
        return sorted(
            eligible,
            key=lambda row: (
                -row["median_edge_sparsity"],
                row["median_sufficiency_error"],
                -row["median_comprehensiveness"],
                row["mean_runtime_seconds"],
                row["config_id"],
                row["top_k"],
            ),
        )[0]
    return sorted(
        summary,
        key=lambda row: (
            row["median_sufficiency_error"],
            -row["median_comprehensiveness"],
            -row["median_edge_sparsity"],
            row["mean_runtime_seconds"],
        ),
    )[0]


def run_validation_tuning(objects, validation_targets, context, global_train_rate):
    rows = []
    audit_targets = validation_targets[
        validation_targets["cohort_order"] < AUDIT_SIZE_PER_COHORT
    ]
    for config in TUNING_CONFIGS:
        print(f"Tune {config['id']} on {len(audit_targets)} validation nodes...", flush=True)
        for target in audit_targets.to_dict("records"):
            target_node = int(target["node_id"])
            batch = materialize_local_batch(
                objects["data"], target_node, NEIGHBORHOOD_SEEDS[0]
            )
            masks = explain_mask(
                objects["adapter"],
                objects["model"],
                batch,
                EXPLAINER_SEEDS[0],
                config,
            )
            full = score_model(objects["adapter"], batch)
            for top_k in EDGE_TOP_K_CANDIDATES:
                selected_count = min(top_k, len(masks["edge_mask"]))
                selected = np.argsort(-masks["edge_mask"], kind="stable")[
                    :selected_count
                ]
                fidelity = evaluate_positions(
                    objects["adapter"], batch, selected, full
                )
                rows.append(
                    {
                        "config_id": config["id"],
                        "top_k": top_k,
                        "target_node_id": target_node,
                        "cohort": target["cohort"],
                        "runtime_seconds": masks["runtime_seconds"],
                        **fidelity,
                    }
                )
            del batch
            gc.collect()
    summary = aggregate_tuning(rows)
    winner = select_tuning_winner(summary)
    selected_config = next(
        config for config in TUNING_CONFIGS if config["id"] == winner["config_id"]
    )
    return rows, summary, winner, selected_config, audit_targets


def create_or_load_protocol_lock(
    objects, validation_targets, context, global_train_rate
) -> dict:
    target_ids = [int(value) for value in validation_targets["node_id"]]
    previous_lock = (
        json_load(PROTOCOL_LOCK_PATH) if PROTOCOL_LOCK_PATH.exists() else None
    )
    if PROTOCOL_LOCK_PATH.exists() and not FORCE_RERUN:
        lock = previous_lock
        if lock["data_sha256"] != objects["data_hash"]:
            raise AssertionError("Existing protocol lock has another dataset")
        if lock["checkpoint_sha256"] != objects["checkpoint_lock"]["sha256"]:
            raise AssertionError("Existing protocol lock has another checkpoint")
        if lock["validation_target_node_ids"] != target_ids:
            raise AssertionError("Validation cohort changed after protocol lock")
        print(f"Reuse protocol lock: {PROTOCOL_LOCK_PATH.name}", flush=True)
        return lock

    rows, summary, winner, config, audit_targets = run_validation_tuning(
        objects, validation_targets, context, global_train_rate
    )
    final_lock = json_load(p1.FINAL_LOCK_PATH)
    checkpoint_locks = final_lock["models"]["C"]["checkpoints"]
    lock = {
        "schema_version": PROTOCOL_VERSION,
        "status": "locked_before_test_explanations",
        "locked_at": datetime.now().astimezone().isoformat(),
        "test_used_for_tuning": False,
        "model_variant": "C",
        "model_seed": PRIMARY_MODEL_SEED,
        "checkpoint_sha256": objects["checkpoint_lock"]["sha256"],
        "model_checkpoints": checkpoint_locks,
        "explainer_seeds": EXPLAINER_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "neighborhood_seeds": NEIGHBORHOOD_SEEDS,
        "data_sha256": objects["data_hash"],
        "community_feature_sha256": objects["feature_manifest"]["feature_sha256"],
        "cohort_size_per_group": COHORT_SIZE,
        "cohort_rule": {
            "high_score_fraud": "top fraud score among labeled fraud nodes; zero-edge nodes are retained",
            "high_score_normal": "top fraud score among labeled normal nodes; zero-edge nodes are retained",
            "low_score_fraud": "lowest fraud score among labeled fraud nodes; zero-edge nodes are retained",
            "low_score_normal_control": (
                "normal nodes in bottom score quartile greedily matched to high-score "
                "normal nodes by log1p(in_degree) and community-size feature"
            ),
        },
        "validation_target_node_ids": target_ids,
        "audit_target_node_ids": [
            int(value) for value in audit_targets["node_id"]
        ],
        "tuning_configs": TUNING_CONFIGS,
        "edge_top_k_candidates": EDGE_TOP_K_CANDIDATES,
        "feature_top_k": FEATURE_TOP_K,
        "sufficiency_threshold": SUFFICIENCY_THRESHOLD,
        "selection_rule": (
            "Among validation candidates with median sufficiency_error<=0.05, "
            "maximize edge sparsity, then minimize sufficiency error and maximize "
            "comprehensiveness; deterministic tie-breaks. "
            "If none pass, minimize median sufficiency error."
        ),
        "tuning_summary": summary,
        "selected": {
            "config": config,
            "edge_top_k": int(winner["top_k"]),
            "feature_top_k": FEATURE_TOP_K,
            "validation_metrics": winner,
        },
        "tuning_rows_sha256": canonical_json_sha256({"rows": rows}),
    }
    if previous_lock is not None:
        lock["supersedes"] = {
            "lock_payload_sha256": previous_lock.get(
                "lock_payload_sha256", previous_lock.get("lock_sha256")
            ),
            "selected": previous_lock.get("selected"),
            "reason": (
                "Protocol regenerated after the Sprint 5 completeness audit. "
                "Cohort selection and tuning remain validation-only; no test metric "
                "is used to choose or alter the protocol."
            ),
        }
    lock["lock_payload_sha256"] = canonical_json_sha256(lock)
    json_dump(PROTOCOL_LOCK_PATH, lock)
    print(
        f"Protocol locked: config={config['id']} edge_top_k={winner['top_k']}",
        flush=True,
    )
    return lock


def load_model_for_seed(seed: int, objects, checkpoint_rows: list[dict]):
    torch = p1.torch
    locked = next(row for row in checkpoint_rows if int(row["seed"]) == seed)
    checkpoint_path = PROJECT_ROOT / locked["path"]
    if p1.file_sha256(checkpoint_path) != locked["sha256"]:
        raise AssertionError(f"Checkpoint hash mismatch for seed {seed}")
    checkpoint = torch.load(checkpoint_path, map_location=p1.DEVICE, weights_only=True)
    model_type = type(objects["model"])
    model_keys = (
        "in_channels",
        "hidden_channels",
        "num_layers",
        "dropout",
        "heads",
        "attention_dropout",
        "time_dim",
        "time_scale",
    )
    model = model_type(
        **{key: checkpoint["model_config"][key] for key in model_keys}
    ).to(p1.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    adapter = type(objects["adapter"])(model).to(p1.DEVICE)
    adapter.eval()
    return model, adapter, locked


def run_main_explanations(
    objects, targets, context, lock, global_train_rate
) -> tuple[list[dict], dict]:
    config = lock["selected"]["config"]
    top_k = int(lock["selected"]["edge_top_k"])
    rows = []
    raw_arrays = {}
    total = len(targets)
    for number, target in enumerate(targets.to_dict("records"), start=1):
        target_node = int(target["node_id"])
        print(
            f"Main explanations {number:02d}/{total}: "
            f"{target['split']} {target['cohort']} node={target_node}",
            flush=True,
        )
        batch = materialize_local_batch(
            objects["data"], target_node, NEIGHBORHOOD_SEEDS[0]
        )
        result, raw = explanation_evaluation(
            objects["adapter"],
            objects["model"],
            batch,
            target_node,
            EXPLAINER_SEEDS[0],
            config,
            top_k,
            context["in_degree"],
            global_train_rate,
        )
        result.update(
            {
                "split": target["split"],
                "cohort": target["cohort"],
                "cohort_order": int(target["cohort_order"]),
                "community_id": int(target["community_id"]),
                "community_risk": float(target["community_risk"]),
                "explanation_failed_operational_threshold": (
                    result["fidelity"]["sufficiency_error"]
                    > SUFFICIENCY_THRESHOLD
                    or result["fidelity"]["comprehensiveness"] <= 0
                ),
            }
        )
        rows.append(result)
        prefix = f"{target['split']}_target_{target_node}"
        for name, array in raw.items():
            raw_arrays[f"{prefix}_{name}"] = array
        del batch
        gc.collect()
    return rows, raw_arrays


def audit_run(adapter, model, batch, seed, config, top_k):
    masks = explain_mask(adapter, model, batch, seed, config)
    k = min(top_k, len(masks["edge_mask"]))
    positions = np.argsort(-masks["edge_mask"], kind="stable")[:k]
    ids = event_ids(batch)
    full = score_model(adapter, batch)
    fidelity = evaluate_positions(adapter, batch, positions, full)
    global_edges = (
        batch.n_id[batch.edge_index[:, positions]].detach().cpu().numpy().astype(np.int64)
        if len(positions)
        else np.empty((2, 0), dtype=np.int64)
    )
    target_node = int(batch.n_id[0])
    selected_neighbor_node_ids = set(int(value) for value in np.unique(global_edges))
    selected_neighbor_node_ids.discard(target_node)
    return {
        "selected_event_ids": set(int(value) for value in ids[positions]),
        "selected_neighbor_node_ids": selected_neighbor_node_ids,
        "sampled_event_ids": set(int(value) for value in ids),
        "feature_importance": masks["feature_importance"],
        "sufficiency_error": fidelity["sufficiency_error"],
        "comprehensiveness": fidelity["comprehensiveness"],
        "runtime_seconds": masks["runtime_seconds"],
    }


def run_stability_audit(objects, validation_targets, lock) -> dict:
    config = lock["selected"]["config"]
    top_k = int(lock["selected"]["edge_top_k"])
    audit_ids = lock["audit_target_node_ids"]
    final_lock = json_load(p1.FINAL_LOCK_PATH)
    checkpoint_rows = final_lock["models"]["C"]["checkpoints"]
    models = {PRIMARY_MODEL_SEED: (objects["model"], objects["adapter"])}
    model_hashes_before = {
        PRIMARY_MODEL_SEED: p1.model_state_sha256(objects["model"])
    }
    for seed in MODEL_SEEDS:
        if seed == PRIMARY_MODEL_SEED:
            continue
        model, adapter, _ = load_model_for_seed(seed, objects, checkpoint_rows)
        models[seed] = (model, adapter)
        model_hashes_before[seed] = p1.model_state_sha256(model)

    target_rows = []
    for number, target_node in enumerate(audit_ids, start=1):
        print(f"Stability audit {number:02d}/{len(audit_ids)} node={target_node}", flush=True)
        fixed_batch = materialize_local_batch(
            objects["data"], int(target_node), NEIGHBORHOOD_SEEDS[0]
        )
        explainer_runs = [
            audit_run(
                objects["adapter"],
                objects["model"],
                fixed_batch,
                seed,
                config,
                top_k,
            )
            for seed in EXPLAINER_SEEDS
        ]
        model_runs = [
            audit_run(
                models[seed][1],
                models[seed][0],
                fixed_batch,
                EXPLAINER_SEEDS[0],
                config,
                top_k,
            )
            for seed in MODEL_SEEDS
        ]
        sampling_runs = []
        for seed in NEIGHBORHOOD_SEEDS:
            batch = materialize_local_batch(objects["data"], int(target_node), seed)
            sampling_runs.append(
                audit_run(
                    objects["adapter"],
                    objects["model"],
                    batch,
                    EXPLAINER_SEEDS[0],
                    config,
                    top_k,
                )
            )
            del batch
        target_rows.append(
            {
                "target_node_id": int(target_node),
                "explainer_seed_edge_jaccard": pairwise_jaccard(
                    [row["selected_event_ids"] for row in explainer_runs]
                ),
                "explainer_seed_node_jaccard": pairwise_jaccard(
                    [row["selected_neighbor_node_ids"] for row in explainer_runs]
                ),
                "explainer_seed_feature_spearman": pairwise_spearman(
                    [row["feature_importance"] for row in explainer_runs]
                ),
                "explainer_seed_sufficiency_std": statistics.pstdev(
                    row["sufficiency_error"] for row in explainer_runs
                ),
                "explainer_seed_comprehensiveness_std": statistics.pstdev(
                    row["comprehensiveness"] for row in explainer_runs
                ),
                "model_seed_edge_jaccard": pairwise_jaccard(
                    [row["selected_event_ids"] for row in model_runs]
                ),
                "model_seed_node_jaccard": pairwise_jaccard(
                    [row["selected_neighbor_node_ids"] for row in model_runs]
                ),
                "model_seed_feature_spearman": pairwise_spearman(
                    [row["feature_importance"] for row in model_runs]
                ),
                "model_seed_sufficiency_std": statistics.pstdev(
                    row["sufficiency_error"] for row in model_runs
                ),
                "model_seed_comprehensiveness_std": statistics.pstdev(
                    row["comprehensiveness"] for row in model_runs
                ),
                "sampling_seed_neighborhood_jaccard": pairwise_jaccard(
                    [row["sampled_event_ids"] for row in sampling_runs]
                ),
                "sampling_seed_explanation_jaccard": pairwise_jaccard(
                    [row["selected_event_ids"] for row in sampling_runs]
                ),
                "sampling_seed_node_jaccard": pairwise_jaccard(
                    [row["selected_neighbor_node_ids"] for row in sampling_runs]
                ),
                "sampling_seed_sufficiency_std": statistics.pstdev(
                    row["sufficiency_error"] for row in sampling_runs
                ),
                "sampling_seed_comprehensiveness_std": statistics.pstdev(
                    row["comprehensiveness"] for row in sampling_runs
                ),
            }
        )
        del fixed_batch
        gc.collect()

    model_state_unchanged = {}
    for seed, (model, _) in models.items():
        model_state_unchanged[str(seed)] = (
            model_hashes_before[seed] == p1.model_state_sha256(model)
        )
    fields = [key for key in target_rows[0] if key != "target_node_id"]
    aggregate = {}
    for field in fields:
        values = []
        for row in target_rows:
            value = row[field]
            values.extend(value if isinstance(value, list) else [value])
        aggregate[field] = {
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "comparison_count": len(values),
        }
    aggregate["operational_edge_stability_threshold"] = 0.5
    aggregate["explainer_seed_edge_stable"] = (
        aggregate["explainer_seed_edge_jaccard"]["median"] >= 0.5
    )
    aggregate["model_seed_edge_stable"] = (
        aggregate["model_seed_edge_jaccard"]["median"] >= 0.5
    )
    aggregate["all_model_states_unchanged"] = all(model_state_unchanged.values())
    return {
        "target_count": len(target_rows),
        "explainer_seeds": EXPLAINER_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "neighborhood_seeds": NEIGHBORHOOD_SEEDS,
        "per_target": target_rows,
        "aggregate": aggregate,
        "model_state_unchanged": model_state_unchanged,
    }


def attach_audit_stability(rows: list[dict], stability: dict) -> None:
    audit_lookup = {
        int(row["target_node_id"]): row for row in stability["per_target"]
    }
    for row in rows:
        audit = audit_lookup.get(int(row["target_node_id"]))
        row["stability_audited"] = audit is not None
        if audit is None:
            row["explainer_seed_edge_jaccard_median"] = None
            row["explainer_seed_node_jaccard_median"] = None
            row["stability_passed"] = None
            continue
        edge_median = statistics.median(audit["explainer_seed_edge_jaccard"])
        node_median = statistics.median(audit["explainer_seed_node_jaccard"])
        row["explainer_seed_edge_jaccard_median"] = edge_median
        row["explainer_seed_node_jaccard_median"] = node_median
        row["stability_passed"] = edge_median >= 0.5


def aggregate_main(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        key = f"{row['split']}::{row['cohort']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, values in groups.items():
        summary[key] = {
            "node_count": len(values),
            "median_sufficiency_error": statistics.median(
                row["fidelity"]["sufficiency_error"] for row in values
            ),
            "median_comprehensiveness": statistics.median(
                row["fidelity"]["comprehensiveness"] for row in values
            ),
            "median_edge_sparsity": statistics.median(
                row["fidelity"]["edge_sparsity"] for row in values
            ),
            "median_feature_sparsity": statistics.median(
                row["fidelity"]["feature_sparsity"] for row in values
            ),
            "edge_explanation_unavailable_count": sum(
                row["edge_explanation_unavailable"] for row in values
            ),
            "mean_runtime_seconds": statistics.fmean(
                row["runtime_seconds"] for row in values
            ),
            "mean_peak_rss_mib": statistics.fmean(
                row["peak_rss_mib"] for row in values
            ),
            "failure_count": sum(
                row["explanation_failed_operational_threshold"] for row in values
            ),
            "mean_community_risk_prior_logit_delta": statistics.fmean(
                row["community_risk_prior_logit_delta"] for row in values
            ),
            "baselines": {
                baseline: {
                    "median_sufficiency_error": statistics.median(
                        row["baselines"][baseline]["sufficiency_error"]
                        for row in values
                    ),
                    "median_comprehensiveness": statistics.median(
                        row["baselines"][baseline]["comprehensiveness"]
                        for row in values
                    ),
                }
                for baseline in ("random", "degree", "recency")
            },
        }
    return summary


def save_raw_arrays(raw_arrays: dict) -> None:
    np.savez_compressed(RAW_MASK_PATH, **raw_arrays)


def update_manifest(objects, lock, results) -> None:
    manifest = json_load(MANIFEST_PATH) if MANIFEST_PATH.exists() else {"schema_version": 1}
    manifest.update(
        {
            "status": "phase3_complete",
            "updated_at": datetime.now().astimezone().isoformat(),
            "phase": 3,
            "protocol_locked": True,
            "protocol_lock_path": PROTOCOL_LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "protocol_lock_sha256": p1.file_sha256(PROTOCOL_LOCK_PATH),
            "protocol_payload_sha256": lock["lock_payload_sha256"],
            "data": results["data"],
            "model": results["model"],
            "selected_explainer_config": lock["selected"],
            "final_explainer_protocol": {
                "locked_at": lock["locked_at"],
                "test_used_for_tuning": lock["test_used_for_tuning"],
                "explainer_seeds": lock["explainer_seeds"],
                "model_seeds": lock["model_seeds"],
                "neighborhood_seeds": lock["neighborhood_seeds"],
                "cohort_rule": lock["cohort_rule"],
                "validation_target_node_ids": lock["validation_target_node_ids"],
                "audit_target_node_ids": lock["audit_target_node_ids"],
                "selected": lock["selected"],
            },
            "target_path": TARGET_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "result_path": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "raw_mask_path": RAW_MASK_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "gate_checks": results["gate_checks"],
        }
    )
    json_dump(MANIFEST_PATH, manifest)


def run_protocol_and_evaluation() -> dict:
    if RESULT_PATH.exists() and PROTOCOL_LOCK_PATH.exists() and not FORCE_RERUN:
        existing = json_load(RESULT_PATH)
        if existing.get("status") == "phase3_complete" and existing.get("gate_passed"):
            print(f"Reuse completed explainer evaluation: {RESULT_PATH.name}", flush=True)
            return existing

    phase1_result = json_load(p1.FEASIBILITY_RESULT_PATH)
    if not phase1_result.get("gate_passed"):
        raise AssertionError("GNNExplainer feasibility checks have not passed")
    print("Load pipeline, data and primary checkpoint...", flush=True)
    p1.load_training_pipeline()
    from sklearn.metrics import average_precision_score
    from torch_geometric.explain import Explainer, GNNExplainer

    globals()["Explainer"] = Explainer
    globals()["GNNExplainer"] = GNNExplainer
    p1.GNNExplainer = GNNExplainer
    objects = p1.build_explainer_objects()
    context = load_context_arrays(objects["data"])
    global_train_rate = float(
        (objects["data"].y[objects["data"].train_idx] == 1)
        .to(p1.torch.float32)
        .mean()
    )

    print("Build validation cohorts before protocol lock...", flush=True)
    validation_predictions = split_predictions(
        objects["model"], objects["data"], objects["config"], "validation"
    )
    validation_ap = float(
        average_precision_score(
            validation_predictions["label"], validation_predictions["fraud_score"]
        )
    )
    expected_validation_ap = float(
        objects["stored_metric"]["validation"]["average_precision"]
    )
    validation_targets = build_cohorts(
        validation_predictions, "validation", context
    )
    lock = create_or_load_protocol_lock(
        objects, validation_targets, context, global_train_rate
    )

    # Test is touched only after the validation-only protocol lock exists.
    if not PROTOCOL_LOCK_PATH.exists():
        raise AssertionError("Protocol lock must exist before test cohort creation")
    print("Protocol is locked; now build the test cohorts...", flush=True)
    test_predictions = split_predictions(
        objects["model"], objects["data"], objects["config"], "test"
    )
    test_ap = float(
        average_precision_score(test_predictions["label"], test_predictions["fraud_score"])
    )
    expected_test_ap = float(objects["stored_metric"]["test"]["average_precision"])
    test_targets = build_cohorts(test_predictions, "test", context)
    targets = p1.pd.concat((validation_targets, test_targets), ignore_index=True)
    targets.to_csv(TARGET_PATH, index=False)

    print("Run 80 locked main explanations...", flush=True)
    primary_state_before = p1.model_state_sha256(objects["model"])
    rows, raw_arrays = run_main_explanations(
        objects, targets, context, lock, global_train_rate
    )
    primary_state_after = p1.model_state_sha256(objects["model"])
    print("Run explainer/model/neighborhood stability audit...", flush=True)
    stability = run_stability_audit(objects, validation_targets, lock)
    attach_audit_stability(rows, stability)
    save_raw_arrays(raw_arrays)

    main_summary = aggregate_main(rows)
    gate_checks = {
        "phase1_gate_passed": bool(phase1_result["gate_passed"]),
        "randomized_model_ranking_differs": bool(
            phase1_result["gate_checks"].get("randomized_model_ranking_differs")
        ),
        "protocol_locked_before_test": (
            lock["status"] == "locked_before_test_explanations"
            and lock["test_used_for_tuning"] is False
        ),
        "validation_ap_reproduced": abs(validation_ap - expected_validation_ap) <= 1e-6,
        "test_ap_reproduced": abs(test_ap - expected_test_ap) <= 1e-6,
        "target_count_is_80": len(targets) == 80,
        "main_explanation_count_is_80": len(rows) == 80,
        "all_main_masks_and_metrics_finite": all(
            np.isfinite(
                [
                    row["full_logit"],
                    row["fraud_score"],
                    row["fidelity"]["sufficiency_error"],
                    row["fidelity"]["comprehensiveness"],
                    row["fidelity"]["edge_sparsity"],
                    row["fidelity"]["feature_sparsity"],
                ]
            ).all()
            for row in rows
        ),
        "primary_model_state_unchanged": primary_state_before == primary_state_after,
        "audit_model_states_unchanged": stability["aggregate"][
            "all_model_states_unchanged"
        ],
        "all_main_explanations_have_provenance": all(
            all(
                key in row
                for key in (
                    "model_seed",
                    "explainer_seed",
                    "neighborhood_seed",
                    "explainer_config_id",
                    "edge_top_k",
                    "feature_top_k",
                )
            )
            for row in rows
        ),
        "all_available_masks_have_nonzero_gradient": all(
            row["node_mask_has_nonzero_gradient"]
            and (
                row["edge_explanation_unavailable"]
                or row["edge_mask_has_nonzero_gradient"]
            )
            for row in rows
        ),
        "selected_masks_saved": all(
            f"{row['split']}_target_{row['target_node_id']}_selected_edge_mask"
            in raw_arrays
            and f"{row['split']}_target_{row['target_node_id']}_selected_feature_mask"
            in raw_arrays
            for row in rows
        ),
        "required_stability_metrics_available": all(
            key in stability["aggregate"]
            for key in (
                "explainer_seed_node_jaccard",
                "explainer_seed_sufficiency_std",
                "explainer_seed_comprehensiveness_std",
                "model_seed_node_jaccard",
                "model_seed_sufficiency_std",
                "model_seed_comprehensiveness_std",
            )
        ),
        "raw_masks_saved_without_pickle": RAW_MASK_PATH.exists(),
    }
    gate_passed = all(gate_checks.values())
    results = {
        "schema_version": 1,
        "status": "phase3_complete" if gate_passed else "phase3_needs_attention",
        "gate_passed": gate_passed,
        "completed_at": datetime.now().astimezone().isoformat(),
        "protocol_lock_path": PROTOCOL_LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "protocol_lock_sha256": p1.file_sha256(PROTOCOL_LOCK_PATH),
        "protocol_payload_sha256": lock["lock_payload_sha256"],
        "data": {
            "path": p1.DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": objects["data_hash"],
            "community_feature_sha256": objects["feature_manifest"]["feature_sha256"],
            "community_assignment_sha256": objects["feature_manifest"]["assignment_sha256"],
        },
        "model": {
            "variant": "C",
            "primary_seed": PRIMARY_MODEL_SEED,
            "checkpoint_sha256": objects["checkpoint_lock"]["sha256"],
            "checkpoints": lock["model_checkpoints"],
            "state_sha256_before": primary_state_before,
            "state_sha256_after": primary_state_after,
        },
        "prediction_reproduction": {
            "validation": {
                "expected_ap": expected_validation_ap,
                "observed_ap": validation_ap,
                "absolute_difference": abs(validation_ap - expected_validation_ap),
            },
            "test": {
                "expected_ap": expected_test_ap,
                "observed_ap": test_ap,
                "absolute_difference": abs(test_ap - expected_test_ap),
            },
        },
        "sanity_checks": {
            "randomized_model": phase1_result["randomized_model_sanity"],
            "adapter_logit_equivalence": phase1_result["gate_checks"][
                "adapter_logit_equivalence"
            ],
            "model_state_unchanged": phase1_result["gate_checks"][
                "model_state_unchanged"
            ],
        },
        "targets": {
            "path": TARGET_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "validation_count": int(len(validation_targets)),
            "test_count": int(len(test_targets)),
            "total_count": int(len(targets)),
            "edge_explanation_unavailable_count": int(
                targets["edge_explanation_unavailable"].sum()
            ),
        },
        "selected_protocol": lock["selected"],
        "main_summary": main_summary,
        "main_explanations": rows,
        "stability": stability,
        "raw_mask_path": RAW_MASK_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "gate_checks": gate_checks,
    }
    json_dump(RESULT_PATH, results)
    update_manifest(objects, lock, results)
    print(
        f"Explainer evaluation {'PASSED' if gate_passed else 'NEED ATTENTION'} | "
        f"result={RESULT_PATH.relative_to(PROJECT_ROOT)}",
        flush=True,
    )
    return results


if __name__ == "__main__":
    EXPLANATION_RESULT = run_protocol_and_evaluation()
