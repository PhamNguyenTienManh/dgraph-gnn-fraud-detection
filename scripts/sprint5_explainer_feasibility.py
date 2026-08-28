"""Check GNNExplainer feasibility against the locked TGAT variant-C checkpoint."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
import copy
import gc
import hashlib
import json
import os
import platform
import threading
import time

import numpy as np
import psutil

MODEL_SEED = 42
NEIGHBORHOOD_SEED = 42
EXPLAINER_SEED = 42
EXPLAINER_EPOCHS = int(os.getenv("SPRINT5_EXPLAINER_EPOCHS", "100"))
EXPLAINER_LR = float(os.getenv("SPRINT5_EXPLAINER_LR", "0.01"))
LOGIT_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-6
PERTURBATION_TOLERANCE = 1e-8


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "data").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Project root not found")


PROJECT_ROOT = find_project_root(Path.cwd())
METRIC_DIR = PROJECT_ROOT / "artifacts" / "metrics"
TRAINING_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_gnn_training.ipynb"
DATA_PATH = PROJECT_ROOT / "data" / "dgraphfin.npz"
FEATURE_PATH = METRIC_DIR / "sprint4_community_features.npz"
FEATURE_MANIFEST_PATH = METRIC_DIR / "sprint4_community_features.manifest.json"
ABLATION_RESULT_PATH = METRIC_DIR / "sprint4_community_ablation.json"
FINAL_LOCK_PATH = METRIC_DIR / "sprint4_community_ablation_final_lock.json"
FEASIBILITY_RESULT_PATH = METRIC_DIR / "sprint5_phase1_feasibility.json"
PILOT_MASK_PATH = METRIC_DIR / "sprint5_phase1_explanations.npz"
EXPLAINER_MANIFEST_PATH = METRIC_DIR / "sprint5_explainer_manifest.json"


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PeakRSSMonitor(AbstractContextManager):
    """Sample process RSS while an explanation run is active."""

    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.peak_bytes = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join()
        return False


def load_training_pipeline() -> None:
    """Load definitions from notebook 02 without loading data or training."""
    notebook = json_load(TRAINING_NOTEBOOK_PATH)
    wanted = [
        "train-setup",
        "train-config",
        "train-loader",
        "train-prepare-functions",
        "train-sampling",
        "train-models",
    ]
    trim_markers = {
        "train-loader": "dataset = load_dgraphfin(DATA_PATH)",
        "train-prepare-functions": "data = prepare_pyg_data(dataset, config)",
        "train-sampling": "balance = class_balance(data, config",
        "train-models": "model = build_model(SELECTED_MODEL, config",
    }
    cells = {cell.get("id"): cell for cell in notebook["cells"]}
    missing = sorted(set(wanted) - set(cells))
    if missing:
        raise KeyError(f"Notebook 02 is missing cell IDs: {missing}")
    for cell_id in wanted:
        source = "".join(cells[cell_id]["source"])
        marker = trim_markers.get(cell_id)
        if marker and marker in source:
            source = source.split(marker, 1)[0]
        exec(compile(source, f"02_gnn_training.ipynb::{cell_id}", "exec"), globals())


def resolve_checkpoint() -> tuple[Path, dict, dict, dict]:
    ablation = json_load(ABLATION_RESULT_PATH)
    lock = json_load(FINAL_LOCK_PATH)
    if ablation["validation_selected_variant"] != "C":
        raise AssertionError("Sprint 4 validation winner is no longer variant C")
    locked = next(
        row
        for row in lock["models"]["C"]["checkpoints"]
        if int(row["seed"]) == MODEL_SEED
    )
    checkpoint_path = PROJECT_ROOT / locked["path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if file_sha256(checkpoint_path) != locked["sha256"]:
        raise AssertionError("Checkpoint seed 42 does not match final-lock SHA-256")
    run_dir = PROJECT_ROOT / ablation["variants"]["C"]["run_dir"]
    config = json_load(run_dir / "config.json")
    metric = json_load(run_dir / f"tgat_seed{MODEL_SEED}" / "metrics.json")
    return checkpoint_path, config, metric, locked


def build_explainer_objects() -> dict:
    feature_manifest = json_load(FEATURE_MANIFEST_PATH)
    data_hash = file_sha256(DATA_PATH)
    if feature_manifest["data_sha256"] != data_hash:
        raise AssertionError("Community feature does not match the dataset")
    if not feature_manifest["train_leave_one_out"]:
        raise AssertionError("Community-risk is missing train leave-one-out")
    if feature_manifest["validation_test_use_train_labels_only"] is not True:
        raise AssertionError("Validation/test community-risk is not train-only")

    checkpoint_path, config, stored_metric, locked = resolve_checkpoint()
    if config["community_feature_set"] != "risk":
        raise AssertionError("Variant C config is not risk-only")
    if config["community_feature_sha256"] != feature_manifest["feature_sha256"]:
        raise AssertionError("Config and community-feature hash do not match")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if checkpoint["variant"] != "C" or int(checkpoint["seed"]) != MODEL_SEED:
        raise AssertionError("Checkpoint metadata is not variant C seed 42")

    class TGATClassifierRisk(TGAT):
        def __init__(self, **model_config):
            super().__init__(**model_config)
            self.core_in_channels = int(model_config["in_channels"])
            self.output_linear = nn.Linear(model_config["hidden_channels"] + 1, 1)

        def forward(self, x, edge_index, edge_type=None, edge_delta=None):
            core_x = x[:, : self.core_in_channels]
            risk = x[:, self.core_in_channels : self.core_in_channels + 1]
            if risk.shape[1] != 1:
                raise ValueError("TGATClassifierRisk requires one risk column")
            self.validate_inputs(core_x, edge_index)
            if edge_delta is None or edge_delta.shape != (edge_index.shape[1],):
                raise ValueError("TGATClassifierRisk requires one edge_delta per edge")
            edge_time = self.time_encoder(edge_delta)
            hidden = F.dropout(
                F.relu(self.input_linear(core_x)),
                p=self.dropout,
                training=self.training,
            )
            hidden = self.conv(hidden, edge_index, edge_time)
            return self.output_linear(torch.cat((hidden, risk), dim=1)).squeeze(-1)

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
    model = TGATClassifierRisk(
        **{key: checkpoint["model_config"][key] for key in model_keys}
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = load_dgraphfin(DATA_PATH)
    data = prepare_pyg_data(dataset, config)
    del dataset
    with np.load(FEATURE_PATH, allow_pickle=False) as archive:
        node_ids = np.asarray(archive["node_id"], dtype=np.int64)
        risk = np.asarray(archive["community_risk_feature"], dtype=np.float32)[:, None]
    if not np.array_equal(node_ids, np.arange(data.num_nodes, dtype=np.int64)):
        raise AssertionError("Community feature node_id mismatch")
    data.x = torch.cat((data.x, torch.from_numpy(risk)), dim=1)
    if data.x.shape[1] != 35:
        raise AssertionError(f"Expected 35 features, got {data.x.shape[1]}")
    del node_ids, risk
    gc.collect()

    class TGATExplainerAdapter(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x, edge_index, edge_delta):
            return self.wrapped(x, edge_index, edge_type=None, edge_delta=edge_delta)

    adapter = TGATExplainerAdapter(model).to(DEVICE)
    adapter.eval()
    return {
        "model": model,
        "adapter": adapter,
        "data": data,
        "config": config,
        "stored_metric": stored_metric,
        "checkpoint_path": checkpoint_path,
        "checkpoint_lock": locked,
        "feature_manifest": feature_manifest,
        "data_hash": data_hash,
    }


def validation_predictions(model, data, config) -> pd.DataFrame:
    loader = NeighborLoader(
        data=data,
        input_nodes=data.valid_idx,
        num_neighbors=config["sampling"]["num_neighbors"],
        batch_size=config["sampling"]["batch_size"],
        num_workers=config["sampling"]["num_workers"],
        subgraph_type="directional",
        transform=attach_node_relative_edge_time,
        shuffle=False,
    )
    set_seed(MODEL_SEED + 10_000)
    node_ids, labels, scores = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(DEVICE)
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


def choose_pilot_targets(frame: pd.DataFrame, data) -> pd.DataFrame:
    in_degree = torch.bincount(data.edge_index[1], minlength=data.num_nodes).cpu().numpy()
    candidates = frame.copy()
    candidates["in_degree"] = in_degree[candidates["node_id"].to_numpy()]
    candidates = candidates[candidates["in_degree"] > 0]
    fraud = candidates[candidates["label"] == 1]
    normal = candidates[candidates["label"] == 0]
    if fraud.empty or normal.empty:
        raise AssertionError("Validation has no eligible fraud/normal nodes")
    rows = [
        {
            "cohort": "high_score_fraud",
            **fraud.nlargest(1, "fraud_score").iloc[0].to_dict(),
        },
        {
            "cohort": "high_score_normal",
            **normal.nlargest(1, "fraud_score").iloc[0].to_dict(),
        },
        {
            "cohort": "low_score_fraud",
            **fraud.nsmallest(1, "fraud_score").iloc[0].to_dict(),
        },
    ]
    result = pd.DataFrame(rows)
    for column in ("node_id", "label", "in_degree"):
        result[column] = result[column].astype(np.int64)
    return result


def materialize_local_batch(data, target_node: int):
    loader = NeighborLoader(
        data=data,
        input_nodes=torch.tensor([target_node], dtype=torch.long),
        num_neighbors=[15],
        batch_size=1,
        num_workers=0,
        subgraph_type="directional",
        transform=attach_node_relative_edge_time,
        shuffle=False,
        generator=torch.Generator().manual_seed(NEIGHBORHOOD_SEED),
    )
    batch = next(iter(loader)).to(DEVICE)
    if int(batch.batch_size) != 1 or int(batch.n_id[0]) != target_node:
        raise AssertionError("Target node is not local index 0")
    if batch.edge_delta.shape != (batch.edge_index.shape[1],):
        raise AssertionError("Local batch is missing edge_delta")
    return batch


def target_score(logit: float, predicted_class: int) -> float:
    fraud_score = float(1.0 / (1.0 + np.exp(-logit)))
    return fraud_score if predicted_class == 1 else 1.0 - fraud_score


def gradient_tracking_gnnexplainer(*, epochs: int, lr: float, **kwargs):
    """Create a GNNExplainer that records whether first-step mask gradients are nonzero."""

    class GradientTrackingGNNExplainer(GNNExplainer):
        def __init__(self, **algorithm_kwargs):
            super().__init__(**algorithm_kwargs)
            self.node_mask_has_nonzero_gradient = False
            self.edge_mask_has_nonzero_gradient = False

        @staticmethod
        def _has_nonzero(mask) -> bool:
            if isinstance(mask, dict):
                return any(
                    value is not None and bool(value.any())
                    for value in mask.values()
                )
            return mask is not None and bool(mask.any())

        def _collect_gradients(self):
            super()._collect_gradients()
            self.node_mask_has_nonzero_gradient = self._has_nonzero(
                self.hard_node_mask
            )
            self.edge_mask_has_nonzero_gradient = self._has_nonzero(
                self.hard_edge_mask
            )

    return GradientTrackingGNNExplainer(epochs=epochs, lr=lr, **kwargs)


def run_one_explanation(adapter, model, batch, cohort: str, global_train_rate: float):
    set_seed(EXPLAINER_SEED)
    before_rss = psutil.Process().memory_info().rss
    with torch.inference_mode():
        direct_logit = model(batch.x, batch.edge_index, None, batch.edge_delta)[0]
        adapter_logit = adapter(batch.x, batch.edge_index, batch.edge_delta)[0]
    logit_difference = float(torch.abs(direct_logit - adapter_logit).cpu())
    predicted_class = int(adapter_logit > 0)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    algorithm = gradient_tracking_gnnexplainer(
        epochs=EXPLAINER_EPOCHS, lr=EXPLAINER_LR
    )
    explainer = Explainer(
        model=adapter,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "binary_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )
    started = time.perf_counter()
    with PeakRSSMonitor() as memory:
        explanation = explainer(
            batch.x, batch.edge_index, index=0, edge_delta=batch.edge_delta
        )
    elapsed = time.perf_counter() - started

    node_mask = explanation.node_mask.detach().cpu()
    edge_mask = explanation.edge_mask.detach().cpu()
    finite = bool(torch.isfinite(node_mask).all() and torch.isfinite(edge_mask).all())
    shapes_valid = (
        tuple(node_mask.shape) == tuple(batch.x.shape)
        and tuple(edge_mask.shape) == (batch.edge_index.shape[1],)
    )
    with torch.inference_mode():
        full_logit = float(adapter(batch.x, batch.edge_index, batch.edge_delta)[0].cpu())

    edge_removal_delta = None
    top_edge = None
    if edge_mask.numel() > 0:
        top_edge_index = int(torch.argmax(edge_mask))
        keep = torch.ones(edge_mask.numel(), dtype=torch.bool, device=DEVICE)
        keep[top_edge_index] = False
        with torch.inference_mode():
            removed_logit = float(
                adapter(
                    batch.x,
                    batch.edge_index[:, keep],
                    batch.edge_delta[keep],
                )[0].cpu()
            )
        edge_removal_delta = full_logit - removed_logit
        local_source = int(batch.edge_index[0, top_edge_index])
        local_target = int(batch.edge_index[1, top_edge_index])
        top_edge = {
            "event_position": top_edge_index,
            "source_node_id": int(batch.n_id[local_source]),
            "target_node_id": int(batch.n_id[local_target]),
            "importance": float(edge_mask[top_edge_index]),
            "edge_timestamp": int(batch.edge_timestamp[top_edge_index]),
            "edge_delta": float(batch.edge_delta[top_edge_index]),
            "edge_type": int(batch.edge_type[top_edge_index]),
        }
        if hasattr(batch, "e_id"):
            top_edge["global_event_id"] = int(batch.e_id[top_edge_index])

    top_flat = int(torch.argmax(node_mask))
    top_local_node = top_flat // int(node_mask.shape[1])
    top_feature = top_flat % int(node_mask.shape[1])
    feature_perturbed = batch.x.clone()
    feature_perturbed[top_local_node, top_feature] = 0
    with torch.inference_mode():
        feature_removed_logit = float(
            adapter(feature_perturbed, batch.edge_index, batch.edge_delta)[0].cpu()
        )
    risk_perturbed = batch.x.clone()
    risk_perturbed[0, 34] = global_train_rate
    with torch.inference_mode():
        risk_prior_logit = float(
            adapter(risk_perturbed, batch.edge_index, batch.edge_delta)[0].cpu()
        )

    result = {
        "cohort": cohort,
        "target_node_id": int(batch.n_id[0]),
        "label": int(batch.y[0]),
        "predicted_class": predicted_class,
        "full_logit": full_logit,
        "fraud_score": float(torch.sigmoid(torch.tensor(full_logit))),
        "target_class_score": target_score(full_logit, predicted_class),
        "adapter_logit_abs_difference": logit_difference,
        "sampled_node_count": int(batch.num_nodes),
        "sampled_event_count": int(batch.edge_index.shape[1]),
        "node_mask_shape": list(node_mask.shape),
        "edge_mask_shape": list(edge_mask.shape),
        "masks_finite": finite,
        "mask_shapes_valid": bool(shapes_valid),
        "edge_mask_min": float(edge_mask.min()) if edge_mask.numel() else None,
        "edge_mask_max": float(edge_mask.max()) if edge_mask.numel() else None,
        "node_mask_min": float(node_mask.min()),
        "node_mask_max": float(node_mask.max()),
        "top_edge": top_edge,
        "top_feature": {
            "local_node": top_local_node,
            "global_node_id": int(batch.n_id[top_local_node]),
            "feature_index": top_feature,
            "feature_name": (
                "community_risk" if top_feature == 34 else f"zero_indicator_{top_feature}"
            ),
            "importance": float(node_mask[top_local_node, top_feature]),
        },
        "edge_removal_logit_delta": edge_removal_delta,
        "top_feature_removal_logit_delta": full_logit - feature_removed_logit,
        "community_risk_prior": global_train_rate,
        "community_risk_prior_logit_delta": full_logit - risk_prior_logit,
        "runtime_seconds": elapsed,
        "rss_before_mib": before_rss / (1024**2),
        "peak_rss_mib": memory.peak_bytes / (1024**2),
        "peak_rss_delta_mib": (memory.peak_bytes - before_rss) / (1024**2),
        "node_mask_has_nonzero_gradient": (
            algorithm.node_mask_has_nonzero_gradient
        ),
        "edge_mask_has_nonzero_gradient": (
            algorithm.edge_mask_has_nonzero_gradient
        ),
    }
    raw = {
        "node_ids": batch.n_id.detach().cpu().numpy().astype(np.int64),
        "edge_index_global": batch.n_id[batch.edge_index.detach().cpu()]
        .numpy()
        .astype(np.int64),
        "edge_timestamp": batch.edge_timestamp.detach().cpu().numpy(),
        "edge_delta": batch.edge_delta.detach().cpu().numpy(),
        "edge_type": batch.edge_type.detach().cpu().numpy(),
        "node_mask": node_mask.numpy().astype(np.float32),
        "edge_mask": edge_mask.numpy().astype(np.float32),
    }
    return result, raw


def save_raw_masks(raw_by_target: dict[int, dict]) -> None:
    arrays = {}
    for target_node, payload in raw_by_target.items():
        for name, array in payload.items():
            arrays[f"target_{target_node}_{name}"] = array
    np.savez_compressed(PILOT_MASK_PATH, **arrays)


def randomized_model_ranking_check(objects, validation) -> dict:
    """Verify that a randomized detector does not reproduce the trained edge ranking."""
    model = objects["model"]
    data = objects["data"]
    in_degree = torch.bincount(
        data.edge_index[1], minlength=data.num_nodes
    ).cpu().numpy().astype(np.int64)
    candidates = validation.copy()
    candidates["in_degree"] = in_degree[candidates["node_id"].to_numpy()]
    row = candidates.sort_values(
        ["in_degree", "node_id"], ascending=[False, True]
    ).iloc[0]
    target_node = int(row["node_id"])
    batch = materialize_local_batch(data, target_node)
    if int(batch.edge_index.shape[1]) < 3:
        raise AssertionError("Randomization sanity target needs at least three events")

    random_model = copy.deepcopy(model)
    set_seed(20_260_824)
    with torch.no_grad():
        for parameter in random_model.parameters():
            parameter.normal_(mean=0.0, std=0.05)
    random_model.eval()
    random_adapter = type(objects["adapter"])(random_model).to(DEVICE)
    random_adapter.eval()

    def edge_mask(adapter, wrapped_model) -> np.ndarray:
        for parameter in wrapped_model.parameters():
            parameter.requires_grad_(False)
        set_seed(EXPLAINER_SEED)
        explainer = Explainer(
            model=adapter,
            algorithm=GNNExplainer(epochs=EXPLAINER_EPOCHS, lr=EXPLAINER_LR),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config={
                "mode": "binary_classification",
                "task_level": "node",
                "return_type": "raw",
            },
        )
        explanation = explainer(
            batch.x, batch.edge_index, index=0, edge_delta=batch.edge_delta
        )
        return explanation.edge_mask.detach().cpu().numpy().astype(np.float64)

    trained_mask = edge_mask(objects["adapter"], model)
    randomized_mask = edge_mask(random_adapter, random_model)
    trained_order = np.argsort(-trained_mask, kind="stable")
    randomized_order = np.argsort(-randomized_mask, kind="stable")
    trained_ranks = np.argsort(trained_order, kind="stable")
    randomized_ranks = np.argsort(randomized_order, kind="stable")
    rank_correlation = float(np.corrcoef(trained_ranks, randomized_ranks)[0, 1])
    k = min(3, len(trained_order))
    trained_top = set(int(value) for value in trained_order[:k])
    randomized_top = set(int(value) for value in randomized_order[:k])
    top_k_jaccard = len(trained_top & randomized_top) / len(
        trained_top | randomized_top
    )
    exact_ranking_match = bool(np.array_equal(trained_order, randomized_order))
    return {
        "target_rule": "validation node with largest in-degree; lowest node ID tie-break",
        "target_node_id": target_node,
        "sampled_event_count": int(batch.edge_index.shape[1]),
        "random_model_seed": 20_260_824,
        "edge_rank_spearman": rank_correlation,
        "top_3_edge_jaccard": top_k_jaccard,
        "exact_edge_ranking_match": exact_ranking_match,
        "passed": not exact_ranking_match,
    }


def run_feasibility_checks() -> dict:
    print("Load pipeline and locked Sprint 4 inputs...", flush=True)
    load_training_pipeline()
    from sklearn.metrics import average_precision_score
    from torch_geometric.explain import Explainer, GNNExplainer

    globals()["Explainer"] = Explainer
    globals()["GNNExplainer"] = GNNExplainer
    objects = build_explainer_objects()
    model = objects["model"]
    adapter = objects["adapter"]
    data = objects["data"]
    state_before = model_state_sha256(model)

    print("Recompute validation scores for pilot selection...", flush=True)
    validation = validation_predictions(model, data, objects["config"])
    validation_ap = float(
        average_precision_score(validation["label"], validation["fraud_score"])
    )
    expected_ap = float(objects["stored_metric"]["validation"]["average_precision"])
    metric_delta = abs(validation_ap - expected_ap)
    targets = choose_pilot_targets(validation, data)
    print(targets.to_string(index=False), flush=True)
    global_train_rate = float(
        (data.y[data.train_idx] == 1).to(torch.float32).mean()
    )

    pilot_results = []
    raw_by_target = {}
    for row in targets.to_dict("records"):
        target_node = int(row["node_id"])
        print(
            f"Explain {row['cohort']} target={target_node} "
            f"score={row['fraud_score']:.6f}...",
            flush=True,
        )
        batch = materialize_local_batch(data, target_node)
        result, raw = run_one_explanation(
            adapter, model, batch, str(row["cohort"]), global_train_rate
        )
        result["validation_selection_score"] = float(row["fraud_score"])
        result["validation_in_degree"] = int(row["in_degree"])
        pilot_results.append(result)
        raw_by_target[target_node] = raw
        del batch
        gc.collect()

    save_raw_masks(raw_by_target)
    print("Run randomized-model edge-ranking sanity check...", flush=True)
    randomization_check = randomized_model_ranking_check(objects, validation)
    state_after = model_state_sha256(model)
    perturbation_deltas = [
        abs(value)
        for row in pilot_results
        for value in (
            row["edge_removal_logit_delta"],
            row["top_feature_removal_logit_delta"],
        )
        if value is not None
    ]
    gate_checks = {
        "checkpoint_hash_matches_final_lock": (
            file_sha256(objects["checkpoint_path"])
            == objects["checkpoint_lock"]["sha256"]
        ),
        "validation_ap_reproduced": metric_delta <= METRIC_TOLERANCE,
        "adapter_logit_equivalence": all(
            row["adapter_logit_abs_difference"] <= LOGIT_TOLERANCE
            for row in pilot_results
        ),
        "all_masks_finite": all(row["masks_finite"] for row in pilot_results),
        "all_mask_shapes_valid": all(
            row["mask_shapes_valid"] for row in pilot_results
        ),
        "all_explanations_have_edges": all(
            row["sampled_event_count"] > 0 for row in pilot_results
        ),
        "mask_gradient_path_valid": all(
            row["node_mask_has_nonzero_gradient"]
            and row["edge_mask_has_nonzero_gradient"]
            for row in pilot_results
        ),
        "randomized_model_ranking_differs": randomization_check["passed"],
        "perturbation_changes_output": (
            bool(perturbation_deltas)
            and max(perturbation_deltas) > PERTURBATION_TOLERANCE
        ),
        "model_state_unchanged": state_before == state_after,
    }
    gate_passed = all(gate_checks.values())
    completed_at = datetime.now().astimezone().isoformat()
    result = {
        "schema_version": 1,
        "status": "phase1_complete" if gate_passed else "phase1_needs_attention",
        "gate_passed": gate_passed,
        "completed_at": completed_at,
        "scope": "sprint5_gnnexplainer_phase1_feasibility",
        "model": {
            "variant": "C",
            "seed": MODEL_SEED,
            "checkpoint_path": objects["checkpoint_path"]
            .relative_to(PROJECT_ROOT)
            .as_posix(),
            "checkpoint_sha256": objects["checkpoint_lock"]["sha256"],
            "state_sha256_before": state_before,
            "state_sha256_after": state_after,
        },
        "data": {
            "path": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": objects["data_hash"],
            "community_feature_path": FEATURE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "community_feature_sha256": objects["feature_manifest"]["feature_sha256"],
            "community_assignment_sha256": objects["feature_manifest"]["assignment_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "numpy": np.__version__,
            "device": str(DEVICE),
        },
        "explainer_config": {
            "algorithm": "GNNExplainer",
            "explanation_type": "model",
            "node_mask_type": "attributes",
            "edge_mask_type": "object",
            "epochs": EXPLAINER_EPOCHS,
            "learning_rate": EXPLAINER_LR,
            "model_seed": MODEL_SEED,
            "neighborhood_seed": NEIGHBORHOOD_SEED,
            "explainer_seed": EXPLAINER_SEED,
            "num_neighbors": [15],
        },
        "validation_reproduction": {
            "expected_average_precision": expected_ap,
            "observed_average_precision": validation_ap,
            "absolute_difference": metric_delta,
            "sample_count": int(len(validation)),
            "positive_count": int((validation["label"] == 1).sum()),
        },
        "pilot_target_rule": (
            "validation nodes with in_degree>0: highest-score fraud, "
            "highest-score normal, lowest-score fraud"
        ),
        "pilot_targets": targets.to_dict("records"),
        "pilot_results": pilot_results,
        "randomized_model_sanity": randomization_check,
        "gate_checks": gate_checks,
        "raw_mask_path": PILOT_MASK_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    json_dump(FEASIBILITY_RESULT_PATH, result)
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "phase": 1,
        "created_at": completed_at,
        "protocol_locked": False,
        "note": (
            "Phase 1 only proves feasibility. Cohort and final explainer "
            "hyperparameters remain unlocked until Sprint 5 Gate 2."
        ),
        "data": result["data"],
        "model": result["model"],
        "environment": result["environment"],
        "phase1_explainer_config": result["explainer_config"],
        "phase1_target_node_ids": [
            int(row["target_node_id"]) for row in pilot_results
        ],
        "phase1_result_path": FEASIBILITY_RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "phase1_raw_mask_path": PILOT_MASK_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "gate_checks": gate_checks,
    }
    json_dump(EXPLAINER_MANIFEST_PATH, manifest)
    print(
        f"Gate 1 {'PASSED' if gate_passed else 'NEEDS ATTENTION'} | "
        f"result={FEASIBILITY_RESULT_PATH.relative_to(PROJECT_ROOT)}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    FEASIBILITY_RESULT = run_feasibility_checks()
