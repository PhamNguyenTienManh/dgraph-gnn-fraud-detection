"""Run target-level feature and community-risk error attribution.

The script consumes the locked target table and frozen local batches.  It never
reselects targets and it keeps validation and test summaries separate.
"""

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
import sprint5_explainer_protocol_and_evaluation as protocol
import sprint5_explainer_reassessment as reassessment


ROOT = runtime.PROJECT_ROOT
PAIR_ANALYSIS_SCHEMA_VERSION = 2
METRICS = ROOT / "artifacts" / "metrics"
FIGURES = ROOT / "artifacts" / "figures" / "sprint5_2"
LOCK_PATH = METRICS / "sprint5_2_protocol_lock.json"
TARGET_PATH = METRICS / "sprint5_2_targets.csv"
BATCH_PATHS = {
    "validation": METRICS / "sprint5_2_validation_target_batches.npz",
    "test": METRICS / "sprint5_2_test_target_batches.npz",
}
RESULT_PATH = METRICS / "sprint5_2_error_attribution.json"
ARRAY_PATH = METRICS / "sprint5_2_error_attribution_arrays.npz"
FEATURE_FIGURE = FIGURES / "02_fp_fn_feature_push.png"
CASE_FIGURE = FIGURES / "04_error_case_panels.png"

FEATURE_NAMES = (
    [f"F{i:02d}" for i in range(1, 18)]
    + [f"F{i:02d}-missing" for i in range(1, 18)]
    + ["community-risk"]
)


def pandas_module():
    if hasattr(runtime, "pd"):
        return runtime.pd
    import pandas
    return pandas


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_hash(model) -> str:
    return runtime.model_state_sha256(model)


def load_frozen_batch(archive, prefix: str):
    torch = runtime.torch
    kwargs = {}
    for name in ("x", "edge_index", "edge_delta", "edge_timestamp", "edge_type", "n_id"):
        value = archive[f"{prefix}_{name}"]
        if name in {"edge_index", "edge_timestamp", "edge_type", "n_id"}:
            tensor = torch.as_tensor(value, dtype=torch.long)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        kwargs[name] = tensor
    if f"{prefix}_e_id" in archive:
        kwargs["e_id"] = torch.as_tensor(archive[f"{prefix}_e_id"], dtype=torch.long)
    batch = runtime.Data(**kwargs)
    batch.batch_size = int(archive[f"{prefix}_batch_size"][0])
    return batch.to(runtime.DEVICE)


def fraud_score(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(logit))))


def score(adapter, batch, *, x=None) -> float:
    torch = runtime.torch
    x = batch.x if x is None else x
    with torch.inference_mode():
        return float(adapter(x, batch.edge_index, batch.edge_delta)[0].cpu())


def prediction_push(full_logit: float, changed_logit: float, predicted_class: int) -> float:
    return float(full_logit - changed_logit if predicted_class == 1 else changed_logit - full_logit)


def rank_desc(values: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(values), kind="stable")


def top_jaccard(left: np.ndarray, right: np.ndarray, k: int = 5) -> float:
    a, b = set(rank_desc(left)[:k]), set(rank_desc(right)[:k])
    return float(len(a & b) / len(a | b))


def explain_target(row: dict, batch, adapter, model, explainer_cls, neutral_cls,
                   threshold: float, train_rate: float) -> dict:
    node_id = int(row["node_id"])
    if int(batch.n_id[0]) != node_id or int(batch.batch_size) != 1:
        raise AssertionError("Frozen batch target alignment failed")
    full_logit = score(adapter, batch)
    local_prediction = int(fraud_score(full_logit) >= threshold)
    locked_prediction = int(row["predicted_class"])
    baseline = runtime.torch.zeros(35, dtype=batch.x.dtype, device=batch.x.device)
    baseline[34] = train_rate
    direct = np.empty(35, dtype=np.float64)
    changed_logits = np.empty(35, dtype=np.float64)
    for feature in range(35):
        changed_x = batch.x.clone()
        changed_x[0, feature] = baseline[feature]
        changed_logits[feature] = score(adapter, batch, x=changed_x)
        direct[feature] = prediction_push(full_logit, changed_logits[feature], locked_prediction)

    runtime.set_seed(42_000 + node_id)
    feature_explanation = reassessment.run_mask(
        explainer_cls, neutral_cls, adapter, model, batch, mask_kind="feature"
    )
    feature_mask = np.asarray(feature_explanation["mask"], dtype=np.float64)
    direct_order, mask_order = rank_desc(np.abs(direct)), rank_desc(feature_mask)

    result = {
        "split": row["split"], "pair_id": row["pair_id"],
        "error_cohort": row["error_cohort"], "target_role": row["target_role"],
        "cohort": row["cohort"], "selection_stratum": row["selection_stratum"],
        "node_id": node_id, "label": int(row["label"]),
        "locked_predicted_class": locked_prediction,
        "local_predicted_class": local_prediction,
        "local_matches_locked_prediction": local_prediction == locked_prediction,
        "locked_fraud_logit": float(row["fraud_logit"]), "local_fraud_logit": full_logit,
        "local_fraud_score": fraud_score(full_logit),
        "feature_names": FEATURE_NAMES,
        "feature_direct_prediction_push": direct.tolist(),
        "feature_direct_absolute_effect": np.abs(direct).tolist(),
        "feature_changed_logits": changed_logits.tolist(),
        "feature_gnnexplainer_importance": feature_mask.tolist(),
        "direct_top5": [FEATURE_NAMES[i] for i in direct_order[:5]],
        "gnnexplainer_top5": [FEATURE_NAMES[i] for i in mask_order[:5]],
        "feature_spearman": protocol.spearman(np.abs(direct), feature_mask),
        "feature_top5_jaccard": top_jaccard(np.abs(direct), feature_mask),
        "community_risk": {
            "direct_prediction_push": float(direct[34]),
            "direct_absolute_effect": float(abs(direct[34])),
            "direct_rank": int(np.flatnonzero(direct_order == 34)[0] + 1),
            "gnnexplainer_importance": float(feature_mask[34]),
            "gnnexplainer_rank": int(np.flatnonzero(mask_order == 34)[0] + 1),
        },
        "feature_explainer_runtime_seconds": float(feature_explanation["runtime_seconds"]),
        "feature_gradient_ok": bool(feature_explanation["gradient_ok"]),
    }
    return result


def summarize_targets(records: list[dict]) -> list[dict]:
    rows = []
    for split in ("validation", "test"):
        for cohort in ("TP", "FP", "FN", "TN"):
            chosen = [r for r in records if r["split"] == split and r["cohort"] == cohort]
            if not chosen:
                continue
            direct = np.asarray([r["feature_direct_prediction_push"] for r in chosen])
            masks = np.asarray([r["feature_gnnexplainer_importance"] for r in chosen])
            top_direct = np.argsort(-np.median(np.abs(direct), axis=0), kind="stable")[:5]
            top_mask = np.argsort(-np.median(masks, axis=0), kind="stable")[:5]
            risk = direct[:, 34]
            rows.append({
                "split": split, "cohort": cohort, "n": len(chosen),
                "direct_top5_by_median_absolute_effect": [FEATURE_NAMES[i] for i in top_direct],
                "gnnexplainer_top5_by_median_importance": [FEATURE_NAMES[i] for i in top_mask],
                "median_feature_spearman": float(np.median([r["feature_spearman"] for r in chosen])),
                "median_feature_top5_jaccard": float(np.median([r["feature_top5_jaccard"] for r in chosen])),
                "community_risk_median_prediction_push": float(np.median(risk)),
                "community_risk_iqr_prediction_push": [float(x) for x in np.quantile(risk, [0.25, 0.75])],
                "community_risk_positive_rate": float(np.mean(risk > 0)),
                "community_risk_direct_top5_rate": float(np.mean([r["community_risk"]["direct_rank"] <= 5 for r in chosen])),
                "community_risk_gnnexplainer_top5_rate": float(np.mean([r["community_risk"]["gnnexplainer_rank"] <= 5 for r in chosen])),
            })
    return rows


def error_control_pairs(records: list[dict], split: str, error_cohort: str):
    grouped = {}
    for record in records:
        if record['split'] == split and record['error_cohort'] == error_cohort:
            grouped.setdefault(record['pair_id'], {})[record['target_role']] = record
    expected_control = {'FP': 'TP', 'FN': 'TN'}[error_cohort]
    pairs = []
    for pair_id, members in sorted(grouped.items()):
        if set(members) != {'error', 'control'}:
            raise AssertionError(f'Incomplete pair: {pair_id}')
        error, control = members['error'], members['control']
        if error['cohort'] != error_cohort or control['cohort'] != expected_control:
            raise AssertionError(f'Unexpected cohorts: {pair_id}')
        if error['locked_predicted_class'] != control['locked_predicted_class']:
            raise AssertionError(f'Different prediction sides: {pair_id}')
        pairs.append((error, control))
    return pairs


def feature_pair_contrast(error_push, control_push, index: int, name: str) -> dict:
    difference = error_push[:, index] - control_push[:, index]
    return {
        'feature_index': index,
        'feature_name': name,
        'error_median_prediction_push': float(np.median(error_push[:, index])),
        'control_median_prediction_push': float(np.median(control_push[:, index])),
        'paired_mean_prediction_push_difference': float(np.mean(difference)),
        'paired_median_prediction_push_difference': float(np.median(difference)),
        'paired_prediction_push_difference_iqr': [
            float(x) for x in np.quantile(difference, [0.25, 0.75])
        ],
        'paired_difference_positive_rate': float(np.mean(difference > 0)),
        'paired_difference_zero_rate': float(np.mean(difference == 0)),
        'error_positive_push_rate': float(np.mean(error_push[:, index] > 0)),
        'control_positive_push_rate': float(np.mean(control_push[:, index] > 0)),
    }


def summarize_error_control_pairs(records: list[dict]) -> list[dict]:
    rows = []
    for split in ('validation', 'test'):
        for error_cohort, control_cohort in (('FP', 'TP'), ('FN', 'TN')):
            pairs = error_control_pairs(records, split, error_cohort)
            if not pairs:
                continue
            error_push = np.asarray([x['feature_direct_prediction_push'] for x, _ in pairs])
            control_push = np.asarray([x['feature_direct_prediction_push'] for _, x in pairs])
            difference = error_push - control_push
            order = np.argsort(-np.median(np.abs(difference), axis=0), kind='stable')
            features = [
                feature_pair_contrast(error_push, control_push, index, name)
                for index, name in enumerate(FEATURE_NAMES)
            ]
            candidates = sorted(
                [feature for feature in features
                 if feature['error_median_prediction_push'] > 0
                 and feature['paired_median_prediction_push_difference'] > 0],
                key=lambda feature: feature['paired_median_prediction_push_difference'],
                reverse=True,
            )
            rows.append({
                'split': split,
                'error_cohort': error_cohort,
                'control_cohort': control_cohort,
                'n_pairs': len(pairs),
                'difference_definition': 'error_prediction_push_minus_control_prediction_push',
                'largest_contrasts_by_median_absolute_paired_difference': [
                    FEATURE_NAMES[index] for index in order[:5]
                ],
                'positive_error_specific_candidates': [
                    feature['feature_name'] for feature in candidates
                ],
                'feature_contrasts': features,
            })
    return rows


def pair_gates(records: list[dict], summaries: list[dict]) -> dict[str, bool]:
    pair_ids = {(record['split'], record['pair_id']) for record in records}
    finite = all(
        np.isfinite([
            feature['paired_mean_prediction_push_difference'],
            feature['paired_median_prediction_push_difference'],
            *feature['paired_prediction_push_difference_iqr'],
        ]).all()
        for row in summaries for feature in row['feature_contrasts']
    )
    return {
        'all_80_error_control_pairs_processed': len(pair_ids) == 80,
        'four_error_control_groups_summarized': len(summaries) == 4,
        'twenty_pairs_per_error_group': all(row['n_pairs'] == 20 for row in summaries),
        'all_pair_contrasts_finite': bool(finite),
    }


def plot_pair_contrast(axis, records: list[dict], summary: dict) -> None:
    pairs = error_control_pairs(records, summary['split'], summary['error_cohort'])
    error = np.asarray([x['feature_direct_prediction_push'] for x, _ in pairs])
    control = np.asarray([x['feature_direct_prediction_push'] for _, x in pairs])
    difference = error - control
    order = np.argsort(-np.median(np.abs(difference), axis=0), kind='stable')[:6]
    axis.boxplot(
        [difference[:, index] for index in order[::-1]], vert=False,
        tick_labels=[FEATURE_NAMES[index] for index in order[::-1]], showfliers=True,
    )
    axis.axvline(0, color='black', lw=0.8)
    axis.set_xscale('symlog', linthresh=0.01)
    axis.set_title('{} · {}–{} (n={})'.format(
        summary['split'], summary['error_cohort'],
        summary['control_cohort'], summary['n_pairs'],
    ))
    axis.set_xlabel('error push − control push')


def make_pair_figure(records: list[dict]) -> None:
    import matplotlib.pyplot as plt
    summaries = summarize_error_control_pairs(records)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, summary in zip(axes.flat, summaries, strict=True):
        plot_pair_contrast(axis, records, summary)
    fig.suptitle('Paired feature contrast: error versus same-prediction control')
    fig.savefig(FEATURE_FIGURE, dpi=160)
    plt.close(fig)


def make_figures(records: list[dict]) -> None:
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    errors = [r for r in records if r["cohort"] in {"FP", "FN"}]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, (split, cohort) in zip(axes.flat, [(s, c) for s in ("validation", "test") for c in ("FP", "FN")], strict=True):
        chosen = [r for r in errors if r["split"] == split and r["cohort"] == cohort]
        values = np.asarray([r["feature_direct_prediction_push"] for r in chosen])
        median = np.median(values, axis=0)
        order = np.argsort(-np.median(np.abs(values), axis=0))[:8][::-1]
        axis.barh([FEATURE_NAMES[i] for i in order], median[order], color=np.where(median[order] >= 0, "#c44e52", "#4c72b0"))
        axis.axvline(0, color="black", lw=0.8)
        axis.set_title(f"{split} · {cohort} (n={len(chosen)})")
        axis.set_xlabel("median push toward operating prediction")
    fig.suptitle("Direct target-feature ablation on error cohorts")
    fig.savefig(FEATURE_FIGURE, dpi=160)
    plt.close(fig)
    make_pair_figure(records)

    cases = []
    for split in ("validation", "test"):
        for cohort in ("FP", "FN"):
            chosen = [r for r in errors if r["split"] == split and r["cohort"] == cohort]
            cases.append(max(chosen, key=lambda r: sum(r["feature_direct_absolute_effect"])))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, case in zip(axes.flat, cases, strict=True):
        values = np.asarray(case["feature_direct_prediction_push"])
        order = np.argsort(-np.abs(values))[:6][::-1]
        axis.barh([FEATURE_NAMES[i] for i in order], values[order], color=np.where(values[order] >= 0, "#dd8452", "#55a868"))
        axis.axvline(0, color="black", lw=0.8)
        axis.set_title(f"{case['split']} · {case['cohort']} · node {case['node_id']}")
        axis.set_xlabel("prediction-direction push")
    fig.suptitle("Representative high-effect error cases")
    fig.savefig(CASE_FIGURE, dpi=160)
    plt.close(fig)


def run() -> dict:
    global THRESHOLD
    started = time.perf_counter()
    for path in (LOCK_PATH, TARGET_PATH, *BATCH_PATHS.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    runtime.load_training_pipeline()
    from torch_geometric.explain import Explainer, GNNExplainer
    objects = runtime.build_explainer_objects()
    model, adapter, data = objects["model"], objects["adapter"], objects["data"]
    before_hash = model_hash(model)
    pd = pandas_module()
    lock = load_json(LOCK_PATH)
    payload = lock["lock_payload"]
    THRESHOLD = float(payload["threshold_selection"]["selected"]["threshold"])
    targets = pd.read_csv(TARGET_PATH)
    if len(targets) != 160 or targets[["split", "node_id"]].duplicated().any():
        raise AssertionError("Target lock must contain 160 unique split/node rows")
    train_rate = float(data.y[data.train_idx].float().mean())
    neutral_cls = reassessment.neutral_gnnexplainer_class(GNNExplainer)
    records = []
    array_payload = {}
    for split in ("validation", "test"):
        split_targets = targets[targets.split == split].sort_values(["cohort", "node_id"])
        with np.load(BATCH_PATHS[split], allow_pickle=False) as archive:
            for row in split_targets.to_dict("records"):
                batch = load_frozen_batch(archive, row["batch_key"])
                record = explain_target(
                    row, batch, adapter, model, Explainer, neutral_cls, THRESHOLD, train_rate
                )
                records.append(record)
                key = f"{split}_{int(row['node_id'])}"
                array_payload[f"{key}_feature_direct_push"] = np.asarray(record["feature_direct_prediction_push"], dtype=np.float32)
                array_payload[f"{key}_feature_mask"] = np.asarray(record["feature_gnnexplainer_importance"], dtype=np.float32)
                del batch
    np.savez_compressed(ARRAY_PATH, **array_payload)
    after_hash = model_hash(model)
    paired_summaries = summarize_error_control_pairs(records)
    gates = {
        "all_160_locked_targets_processed": len(records) == 160,
        "model_state_unchanged": before_hash == after_hash,
        "all_feature_values_finite": bool(all(
            np.isfinite(r["feature_direct_prediction_push"]).all()
            and np.isfinite(r["feature_gnnexplainer_importance"]).all()
            for r in records
        )),
        "feature_masks_have_gradient": all(r["feature_gradient_ok"] for r in records),
    }
    gates.update(pair_gates(records, paired_summaries))
    make_figures(records)
    result = {
        "schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
        "protocol_lock_sha256": lock["lock_payload_sha256"],
        "threshold": THRESHOLD, "feature_names": FEATURE_NAMES,
        "target_summaries": summarize_targets(records),
        "diagnostics": {
            "feature_nonzero_gradient_rate": float(np.mean([r["feature_gradient_ok"] for r in records])),
        },
        "technical_gates": gates,
        "targets": records,
        "artifacts": {
            "arrays": ARRAY_PATH.relative_to(ROOT).as_posix(),
            "feature_figure": FEATURE_FIGURE.relative_to(ROOT).as_posix(),
            "case_figure": CASE_FIGURE.relative_to(ROOT).as_posix(),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    result['schema_version'] = PAIR_ANALYSIS_SCHEMA_VERSION
    result['error_control_summaries'] = paired_summaries
    save_json(RESULT_PATH, result)
    if not all(gates.values()):
        raise AssertionError(f"Technical gates failed: {gates}")
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(json.dumps({
        "result": RESULT_PATH.relative_to(ROOT).as_posix(),
        "targets": len(result["targets"]),
        "technical_gates": result["technical_gates"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, ensure_ascii=False))


THRESHOLD = 0.5

if __name__ == "__main__":
    main()
