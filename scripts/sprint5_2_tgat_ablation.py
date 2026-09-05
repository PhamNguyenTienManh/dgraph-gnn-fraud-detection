"""Evaluate locked inference-time interventions on the frozen TGAT model."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sprint5_explainer_feasibility as runtime


ROOT = runtime.PROJECT_ROOT
METRICS = ROOT / "artifacts" / "metrics"
FIGURES = ROOT / "artifacts" / "figures" / "sprint5_2"
LOCK_PATH = METRICS / "sprint5_2_protocol_lock.json"
PREDICTION_LOCK_PATH = METRICS / "sprint5_2_predictions.csv.gz"
RESULT_PATH = METRICS / "sprint5_2_ablation_results.json"
PREDICTION_PATH = METRICS / "sprint5_2_ablation_predictions.npz"
FIGURE_PATH = FIGURES / "03_tgat_input_ablation.png"

VARIANTS = ("FULL", "T-CONST", "N-BASE", "MP-OFF")


def pandas_module():
    if hasattr(runtime, "pd"):
        return runtime.pd
    import pandas
    return pandas


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def constant_time(batch):
    torch = runtime.torch
    edge_count = int(batch.edge_index.shape[1])
    destinations = batch.edge_index[1].detach().cpu().numpy().astype(np.int64)
    if edge_count and np.any(destinations >= int(batch.batch_size)):
        raise AssertionError("One-hop directional edges must terminate at seed nodes")
    const = batch.edge_delta.clone()
    for local_target in np.unique(destinations):
        positions = np.flatnonzero(destinations == local_target)
        const[torch.as_tensor(positions, device=const.device)] = torch.median(batch.edge_delta[torch.as_tensor(positions, device=const.device)])
    return const


def neighbor_baseline(batch, train_rate: float):
    count = int(batch.batch_size)
    base = batch.x.clone()
    if int(batch.x.shape[0]) > count:
        base[count:, :] = 0
        base[count:, 34] = train_rate
    return base


def forward(model, batch, *, x=None, edge_delta=None, message_passing=True) -> np.ndarray:
    torch = runtime.torch
    x = batch.x if x is None else x
    edge_delta = batch.edge_delta if edge_delta is None else edge_delta
    edge_index = batch.edge_index
    edge_type = getattr(batch, "edge_type", None)
    if not message_passing:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=batch.x.device)
        edge_delta = torch.empty((0,), dtype=batch.edge_delta.dtype, device=batch.x.device)
        edge_type = torch.empty((0,), dtype=torch.long, device=batch.x.device)
    with torch.inference_mode():
        output = model(x, edge_index, edge_type, edge_delta)[: int(batch.batch_size)]
    return output.detach().cpu().numpy().astype(np.float64)


def evaluate_split(objects: dict, split: str, train_rate: float) -> dict:
    torch = runtime.torch
    data, model, config = objects["data"], objects["model"], objects["config"]
    input_nodes = data.valid_idx if split == "validation" else data.test_idx
    split_seed = 10_042 if split == "validation" else 20_042
    loader = runtime.NeighborLoader(
        data=data, input_nodes=input_nodes,
        num_neighbors=config["sampling"]["num_neighbors"],
        batch_size=config["sampling"]["batch_size"],
        num_workers=config["sampling"]["num_workers"],
        subgraph_type="directional", transform=runtime.attach_node_relative_edge_time,
        shuffle=False,
    )
    runtime.set_seed(split_seed)
    node_parts, label_parts = [], []
    outputs = {name: [] for name in VARIANTS}
    for batch in loader:
        batch = batch.to(runtime.DEVICE)
        n = int(batch.batch_size)
        node_parts.append(batch.n_id[:n].cpu().numpy().astype(np.int64))
        label_parts.append(batch.y[:n].cpu().numpy().astype(np.int64))
        time_const = constant_time(batch)
        neighbor_base = neighbor_baseline(batch, train_rate)
        outputs["FULL"].append(forward(model, batch))
        outputs["T-CONST"].append(forward(model, batch, edge_delta=time_const))
        outputs["N-BASE"].append(forward(model, batch, x=neighbor_base))
        outputs["MP-OFF"].append(forward(model, batch, message_passing=False))
        del batch
    arrays = {
        "node_id": np.concatenate(node_parts), "label": np.concatenate(label_parts),
        **{f"{name}_logit": np.concatenate(parts) for name, parts in outputs.items()},
    }
    if not all(np.isfinite(arrays[f"{name}_logit"]).all() for name in VARIANTS):
        raise AssertionError(f"Non-finite logits in {split}")
    return arrays


def summarize_split(arrays: dict, split: str) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = arrays["label"]
    rows = []
    for variant in VARIANTS:
        scores = sigmoid(arrays[f"{variant}_logit"])
        rows.append({
            "split": split, "variant": variant, "n": int(len(labels)),
            "average_precision": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
        })
    return rows


def root_path_check(objects: dict, train_rate: float) -> dict:
    torch = runtime.torch
    target = int(objects["data"].valid_idx[0])
    loader = runtime.NeighborLoader(
        data=objects["data"], input_nodes=torch.tensor([target]), num_neighbors=[15],
        batch_size=1, num_workers=0, subgraph_type="directional",
        transform=runtime.attach_node_relative_edge_time, shuffle=False,
        generator=torch.Generator().manual_seed(42),
    )
    batch = next(iter(loader)).to(runtime.DEVICE)
    base = forward(objects["model"], batch, message_passing=False)[0]
    core = batch.x.clone(); core[0, :34] += 0.5
    risk = batch.x.clone(); risk[0, 34] = train_rate if float(batch.x[0, 34]) != train_rate else train_rate + 0.25
    neighbors = batch.x.clone()
    if neighbors.shape[0] > 1:
        neighbors[1:] += 7.0
    core_logit = forward(objects["model"], batch, x=core, message_passing=False)[0]
    risk_logit = forward(objects["model"], batch, x=risk, message_passing=False)[0]
    neighbor_logit = forward(objects["model"], batch, x=neighbors, message_passing=False)[0]
    result = {
        "target_node_id": target, "base_logit": float(base),
        "target_core_change_absolute_logit_delta": float(abs(core_logit - base)),
        "target_risk_change_absolute_logit_delta": float(abs(risk_logit - base)),
        "neighbor_change_absolute_logit_delta": float(abs(neighbor_logit - base)),
    }
    result["passed"] = bool(result["target_core_change_absolute_logit_delta"] > 1e-7 and result["target_risk_change_absolute_logit_delta"] > 1e-7 and result["neighbor_change_absolute_logit_delta"] <= 1e-7)
    return result


def make_figure(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, split in enumerate(("validation", "test")):
        chosen = [r for r in rows if r["split"] == split]
        full = next(r for r in chosen if r["variant"] == "FULL")
        names = [r["variant"] for r in chosen if r["variant"] != "FULL"]
        ap = [r["average_precision"] - full["average_precision"] for r in chosen if r["variant"] != "FULL"]
        roc_auc = [r["roc_auc"] - full["roc_auc"] for r in chosen if r["variant"] != "FULL"]
        axes[0, column].bar(names, ap, color="#dd8452")
        axes[0, column].axhline(0, color="black", lw=0.8)
        axes[0, column].set_title(f"{split}: Δaverage precision")
        axes[1, column].bar(names, roc_auc, color="#4c72b0")
        axes[1, column].axhline(0, color="black", lw=0.8)
        axes[1, column].set_title(f"{split}: ΔROC-AUC")
        for axis in axes[:, column]:
            axis.tick_params(axis="x", rotation=30)
    fig.suptitle("Frozen TGAT sensitivity to inference-time input interventions")
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


def run() -> dict:
    started = time.perf_counter()
    for path in (LOCK_PATH, PREDICTION_LOCK_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    runtime.load_training_pipeline()
    objects = runtime.build_explainer_objects()
    model = objects["model"]
    before_hash = runtime.model_state_sha256(model)
    lock = load_json(LOCK_PATH)
    train_rate = float(objects["data"].y[objects["data"].train_idx].float().mean())
    split_arrays, summaries, archive_payload = {}, [], {}
    for split in ("validation", "test"):
        arrays = evaluate_split(objects, split, train_rate)
        split_arrays[split] = arrays
        summaries.extend(summarize_split(arrays, split))
        for key, value in arrays.items():
            archive_payload[f"{split}_{key}"] = value
    locked = pandas_module().read_csv(PREDICTION_LOCK_PATH)
    reproduction = {}
    for split in ("validation", "test"):
        expected = locked[locked.split == split].sort_values("node_id")
        arrays = split_arrays[split]
        order = np.argsort(arrays["node_id"])
        if not np.array_equal(arrays["node_id"][order], expected.node_id.to_numpy(dtype=np.int64)):
            raise AssertionError(f"FULL node alignment failed for {split}")
        difference = np.max(np.abs(arrays["FULL_logit"][order] - expected.fraud_logit.to_numpy(dtype=np.float64)))
        reproduction[split] = {"max_absolute_logit_difference": float(difference), "passed": bool(difference <= 1e-6)}
    root_check = root_path_check(objects, train_rate)
    after_hash = runtime.model_state_sha256(model)
    gates = {
        "full_predictions_reproduced": all(x["passed"] for x in reproduction.values()),
        "all_variants_both_splits": len(summaries) == 8,
        "all_outputs_finite": bool(all(np.isfinite(split_arrays[s][f"{v}_logit"]).all() for s in split_arrays for v in VARIANTS)),
        "model_state_unchanged": before_hash == after_hash,
        "mp_off_retains_target_root_and_risk_only": bool(root_check["passed"]),
    }
    np.savez_compressed(PREDICTION_PATH, **archive_payload)
    make_figure(summaries)
    result = {
        "schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
        "protocol_lock_sha256": lock["lock_payload_sha256"],
        "interpretation_scope": "frozen-model sensitivity under inference-time counterfactual inputs; not a causal effect",
        "variants": list(VARIANTS), "split_results": summaries,
        "full_reproduction": reproduction, "mp_off_root_path_check": root_check,
        "technical_gates": gates,
        "artifacts": {"variant_predictions": PREDICTION_PATH.relative_to(ROOT).as_posix(), "figure": FIGURE_PATH.relative_to(ROOT).as_posix()},
        "runtime_seconds": float(time.perf_counter() - started),
    }
    save_json(RESULT_PATH, result)
    if not all(gates.values()):
        raise AssertionError(f"Technical gates failed: {gates}")
    gc.collect()
    return result


def main() -> None:
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({
        "result": RESULT_PATH.relative_to(ROOT).as_posix(),
        "technical_gates": result["technical_gates"],
        "root_path_check": result["mp_off_root_path_check"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
