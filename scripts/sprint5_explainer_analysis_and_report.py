"""Build Sprint 5 analysis tables, figures, report, and package checks."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = PROJECT_ROOT / "artifacts" / "metrics"
FIGURE_DIR = PROJECT_ROOT / "artifacts" / "figures" / "sprint5"
DOC_DIR = PROJECT_ROOT / "docs"

RESULT_PATH = METRIC_DIR / "sprint5_explainer_results.json"
LOCK_PATH = METRIC_DIR / "sprint5_explainer_protocol_lock.json"
TARGET_PATH = METRIC_DIR / "sprint5_explanation_targets.csv"
RAW_PATH = METRIC_DIR / "sprint5_explanations.npz"
MANIFEST_PATH = METRIC_DIR / "sprint5_explainer_manifest.json"
COMMUNITY_TABLE_PATH = METRIC_DIR / "sprint4_community_table.csv.gz"
COMMUNITY_ASSIGNMENT_PATH = METRIC_DIR / "sprint4_community_assignments.npz"
DATA_PATH = PROJECT_ROOT / "data" / "dgraphfin.npz"

ANALYSIS_CSV_PATH = METRIC_DIR / "sprint5_explanation_analysis.csv"
CASE_PATH = METRIC_DIR / "sprint5_case_selection.json"
ANALYSIS_RESULT_PATH = METRIC_DIR / "sprint5_phase45_results.json"
REPORT_PATH = DOC_DIR / "sprint5_report.md"

COHORT_ORDER = [
    "high_score_fraud",
    "high_score_normal",
    "low_score_fraud",
    "low_score_normal_control",
]
COHORT_LABELS = {
    "high_score_fraud": "High-score fraud",
    "high_score_normal": "High-score normal",
    "low_score_fraud": "Low-score fraud",
    "low_score_normal_control": "Low-score normal control",
}
SPLIT_COLORS = {"validation": "#4C78A8", "test": "#F58518"}
FIGURE_FILES = [
    "01_cohort_overview.png",
    "02_fidelity_sparsity.png",
    "03_stability.png",
    "04_feature_importance.png",
    "05_risk_counterfactual.png",
    "06_risky_subgraph_cases.png",
    "07_failure_cases.png",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def load_inputs():
    required = [
        RESULT_PATH,
        LOCK_PATH,
        TARGET_PATH,
        RAW_PATH,
        MANIFEST_PATH,
        COMMUNITY_TABLE_PATH,
        COMMUNITY_ASSIGNMENT_PATH,
        DATA_PATH,
    ]
    missing = [path.relative_to(PROJECT_ROOT).as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Sprint 5 input artifacts: {missing}")
    results = json_load(RESULT_PATH)
    lock = json_load(LOCK_PATH)
    if not results.get("gate_passed"):
        raise RuntimeError("Sprint 5 explainer evaluation gate has not passed")
    if lock.get("status") != "locked_before_test_explanations":
        raise RuntimeError("Explainer protocol is not locked")
    targets = pd.read_csv(TARGET_PATH)
    with gzip.open(COMMUNITY_TABLE_PATH, "rt", encoding="utf-8-sig", newline="") as handle:
        community_table = pd.read_csv(handle)
    assignment_npz = np.load(COMMUNITY_ASSIGNMENT_PATH, allow_pickle=False)
    node_ids = assignment_npz["node_id"].astype(np.int64, copy=False)
    community_ids = assignment_npz["community_id"].astype(np.int64, copy=False)
    if not np.array_equal(node_ids, np.arange(len(node_ids), dtype=np.int64)):
        raise AssertionError("Community assignment node IDs are not dense/global aligned")
    data_npz = np.load(DATA_PATH, allow_pickle=False)
    labels = data_npz["y"].astype(np.int64, copy=False)
    raw = np.load(RAW_PATH, allow_pickle=False)
    return results, lock, targets, community_table, community_ids, labels, raw


def raw_prefix(row: dict) -> str:
    return f"{row['split']}_target_{int(row['target_node_id'])}_"


def build_analysis(
    results: dict,
    targets: pd.DataFrame,
    community_table: pd.DataFrame,
    community_ids: np.ndarray,
    raw,
) -> tuple[pd.DataFrame, list[dict]]:
    target_lookup = targets.set_index(["split", "node_id"]).to_dict("index")
    community_lookup = community_table.set_index("community_id").to_dict("index")
    analysis_rows = []
    feature_rows = []
    for explanation in results["main_explanations"]:
        split = explanation["split"]
        node_id = int(explanation["target_node_id"])
        target = target_lookup[(split, node_id)]
        prefix = raw_prefix(explanation)
        edge_index = raw[prefix + "edge_index_global"]
        edge_mask = raw[prefix + "edge_mask"]
        feature_importance = raw[prefix + "feature_importance"]
        selected_positions = np.asarray(
            explanation["selected_event_positions"], dtype=np.int64
        )
        selected_edges = edge_index[:, selected_positions]
        target_community = int(explanation["community_id"])
        if selected_edges.size:
            source_communities = community_ids[selected_edges[0]]
            target_communities = community_ids[selected_edges[1]]
            internal = (source_communities == target_community) & (
                target_communities == target_community
            )
            internal_ratio = float(internal.mean())
            selected_nodes = np.unique(selected_edges)
            distinct_communities = int(np.unique(community_ids[selected_nodes]).size)
        else:
            internal_ratio = float("nan")
            distinct_communities = 1
        community = community_lookup[target_community]
        perturbed_logit = float(explanation["community_risk_prior_logit"])
        perturbed_score = float(explanation["community_risk_prior_fraud_score"])
        row = {
            "split": split,
            "cohort": explanation["cohort"],
            "cohort_order": int(explanation["cohort_order"]),
            "target_node_id": node_id,
            "label": int(explanation["label"]),
            "predicted_class": int(explanation["predicted_class"]),
            "fraud_score": float(explanation["fraud_score"]),
            "in_degree": int(target["in_degree"]),
            "community_id": target_community,
            "community_size": int(community["size"]),
            "community_risk": float(explanation["community_risk"]),
            "is_risky_community": bool(community["is_selected_risky"]),
            "community_train_fraud_lift": float(community["train_fraud_lift"]),
            "community_conductance": float(community["conductance"]),
            "selected_internal_edge_ratio": internal_ratio,
            "selected_distinct_community_count": distinct_communities,
            "sampled_node_count": int(explanation["sampled_node_count"]),
            "sampled_event_count": int(explanation["sampled_event_count"]),
            "edge_explanation_unavailable": bool(
                explanation["edge_explanation_unavailable"]
            ),
            "selected_edge_count": int(explanation["fidelity"]["selected_edge_count"]),
            "edge_sparsity": float(explanation["fidelity"]["edge_sparsity"]),
            "feature_sparsity": float(explanation["fidelity"]["feature_sparsity"]),
            "sufficiency_error": float(explanation["fidelity"]["sufficiency_error"]),
            "comprehensiveness": float(explanation["fidelity"]["comprehensiveness"]),
            "community_risk_feature_importance": float(
                explanation["root_community_risk_importance"]
            ),
            "community_risk_prior_logit_delta": float(
                explanation["community_risk_prior_logit_delta"]
            ),
            "community_risk_prior_fraud_score": perturbed_score,
            "community_risk_prior_score_delta": float(explanation["fraud_score"])
            - perturbed_score,
            "model_seed": int(explanation["model_seed"]),
            "explainer_seed": int(explanation["explainer_seed"]),
            "neighborhood_seed": int(explanation["neighborhood_seed"]),
            "explainer_config_id": explanation["explainer_config_id"],
            "stability_audited": bool(explanation["stability_audited"]),
            "explainer_seed_edge_jaccard_median": explanation[
                "explainer_seed_edge_jaccard_median"
            ],
            "explainer_seed_node_jaccard_median": explanation[
                "explainer_seed_node_jaccard_median"
            ],
            "stability_passed": explanation["stability_passed"],
            "failure_flag": bool(
                explanation["explanation_failed_operational_threshold"]
            ),
            "runtime_seconds": float(explanation["runtime_seconds"]),
            "peak_rss_mib": float(explanation["peak_rss_mib"]),
            "edge_mask_max": float(np.max(edge_mask)) if edge_mask.size else float("nan"),
        }
        for baseline_name, metrics in explanation["baselines"].items():
            row[f"{baseline_name}_sufficiency_error"] = float(
                metrics["sufficiency_error"]
            )
            row[f"{baseline_name}_comprehensiveness"] = float(
                metrics["comprehensiveness"]
            )
        analysis_rows.append(row)
        feature_rows.append(
            {
                "split": split,
                "cohort": explanation["cohort"],
                "target_node_id": node_id,
                "values": feature_importance.astype(float).tolist(),
            }
        )
    analysis = pd.DataFrame(analysis_rows).sort_values(
        ["split", "cohort", "cohort_order"]
    )
    if len(analysis) != 80:
        raise AssertionError(f"Expected 80 analysis rows, got {len(analysis)}")
    if analysis["target_node_id"].nunique() != 80:
        raise AssertionError("Target node IDs are not unique across the analysis")
    return analysis, feature_rows


def aggregate_cohorts(analysis: pd.DataFrame) -> dict:
    output = {}
    for (split, cohort), frame in analysis.groupby(["split", "cohort"], sort=False):
        key = f"{split}::{cohort}"
        output[key] = {
            "node_count": int(len(frame)),
            "fraud_score_median": float(frame["fraud_score"].median()),
            "sufficiency_error_median": float(frame["sufficiency_error"].median()),
            "comprehensiveness_median": float(frame["comprehensiveness"].median()),
            "edge_sparsity_median": float(frame["edge_sparsity"].median()),
            "feature_sparsity_median": float(frame["feature_sparsity"].median()),
            "edge_explanation_unavailable_count": int(
                frame["edge_explanation_unavailable"].sum()
            ),
            "selected_internal_edge_ratio_mean": float(
                frame["selected_internal_edge_ratio"].mean()
            ),
            "selected_distinct_community_count_mean": float(
                frame["selected_distinct_community_count"].mean()
            ),
            "risky_community_rate": float(frame["is_risky_community"].mean()),
            "community_train_fraud_lift_mean": float(
                frame["community_train_fraud_lift"].mean()
            ),
            "community_conductance_mean": float(
                frame["community_conductance"].mean()
            ),
            "community_risk_feature_importance_mean": float(
                frame["community_risk_feature_importance"].mean()
            ),
            "community_risk_prior_score_delta_mean": float(
                frame["community_risk_prior_score_delta"].mean()
            ),
            "failure_rate": float(frame["failure_flag"].mean()),
        }
    return output


def select_cases(analysis: pd.DataFrame) -> dict:
    validation = analysis[analysis["split"] == "validation"].copy()
    representatives = []
    for cohort in COHORT_ORDER:
        candidates = validation[
            (validation["cohort"] == cohort) & (~validation["failure_flag"])
        ].sort_values(
            ["comprehensiveness", "sufficiency_error", "target_node_id"],
            ascending=[False, True, True],
        )
        if candidates.empty:
            candidates = validation[validation["cohort"] == cohort].sort_values(
                ["failure_flag", "comprehensiveness", "target_node_id"],
                ascending=[True, False, True],
            )
        representatives.append(int(candidates.iloc[0]["target_node_id"]))

    false_alarm = validation[
        (validation["label"] == 0)
        & (validation["predicted_class"] == 1)
        & (validation["failure_flag"])
    ].sort_values(["fraud_score", "target_node_id"], ascending=[False, True])
    miss = validation[
        (validation["label"] == 1)
        & (validation["predicted_class"] == 0)
        & (validation["failure_flag"])
    ].sort_values(["fraud_score", "target_node_id"], ascending=[True, True])
    if false_alarm.empty or miss.empty:
        raise AssertionError("Required validation failure cases are unavailable")
    candidate_risky = validation[
        validation["cohort"].isin(["high_score_fraud", "high_score_normal"])
        & (~validation["failure_flag"])
        & (validation["stability_audited"])
        & (validation["stability_passed"] == True)
        & (~validation["edge_explanation_unavailable"])
    ].sort_values(
        ["fraud_score", "comprehensiveness", "target_node_id"],
        ascending=[False, False, True],
    )
    payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "representative_rule": (
            "For each cohort, select the non-failure explanation with highest "
            "comprehensiveness; then lowest sufficiency error and lowest node ID."
        ),
        "representative_target_node_ids": representatives,
        "candidate_risky_subgraph_rule": (
            "Validation high-score cohort; fidelity passes; explainer-seed edge "
            "stability median >=0.5 on the predeclared audit; edge provenance available."
        ),
        "candidate_risky_subgraph_target_node_ids": [
            int(value) for value in candidate_risky["target_node_id"]
        ],
        "failure_rule": {
            "false_alarm_like": (
                "Among validation label=0, predicted=1, failure_flag=true nodes, "
                "select highest fraud score; then lowest node ID."
            ),
            "miss_like": (
                "Among validation label=1, predicted=0, failure_flag=true nodes, "
                "select lowest fraud score; then lowest node ID."
            ),
        },
        "failure_target_node_ids": {
            "false_alarm_like": int(false_alarm.iloc[0]["target_node_id"]),
            "miss_like": int(miss.iloc[0]["target_node_id"]),
        },
    }
    return payload


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grouped_boxplot(ax, analysis, column, title, log=False):
    positions, data, colors = [], [], []
    for cohort_index, cohort in enumerate(COHORT_ORDER):
        for split_index, split in enumerate(["validation", "test"]):
            values = analysis.loc[
                (analysis["cohort"] == cohort) & (analysis["split"] == split), column
            ].to_numpy(float)
            if log:
                values = np.log1p(values)
            positions.append(cohort_index * 3 + split_index)
            data.append(values)
            colors.append(SPLIT_COLORS[split])
    boxes = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks([index * 3 + 0.5 for index in range(4)])
    ax.set_xticklabels([COHORT_LABELS[value] for value in COHORT_ORDER], rotation=18, ha="right")
    ax.set_title(title)


def figure_01(analysis: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    grouped_boxplot(axes[0, 0], analysis, "fraud_score", "Fraud score")
    grouped_boxplot(axes[0, 1], analysis, "in_degree", "log(1 + in-degree)", log=True)
    grouped_boxplot(axes[1, 0], analysis, "community_size", "log(1 + community size)", log=True)
    label_counts = (
        analysis.groupby(["cohort", "label"]).size().unstack(fill_value=0).reindex(COHORT_ORDER)
    )
    bottom = np.zeros(len(label_counts))
    for label, color, text in [(0, "#4C78A8", "Normal"), (1, "#E45756", "Fraud")]:
        values = label_counts[label].to_numpy() if label in label_counts else np.zeros(4)
        axes[1, 1].bar(range(4), values, bottom=bottom, color=color, label=text)
        bottom += values
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_xticklabels(
        [COHORT_LABELS[value] for value in COHORT_ORDER], rotation=18, ha="right"
    )
    axes[1, 1].set_title("Label count across both splits")
    axes[1, 1].legend()
    handles = [
        Line2D([0], [0], color=color, lw=7, label=split.title())
        for split, color in SPLIT_COLORS.items()
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=2, frameon=False,
    )
    fig.suptitle("Sprint 5 cohort overview", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, FIGURE_FILES[0])


def figure_02(analysis: pd.DataFrame, lock: dict) -> None:
    selected_config = lock["selected"]["config"]["id"]
    tuning = pd.DataFrame(lock["tuning_summary"])
    tuning = tuning[tuning["config_id"] == selected_config].sort_values("top_k")
    validation = analysis[analysis["split"] == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(
        tuning["top_k"], tuning["median_sufficiency_error"], "o-", lw=2,
        color="#4C78A8", label="GNNExplainer tuning",
    )
    axes[1].plot(
        tuning["top_k"], tuning["median_comprehensiveness"], "o-", lw=2,
        color="#4C78A8", label="GNNExplainer tuning",
    )
    markers = {"random": "s", "degree": "^", "recency": "D"}
    colors = {"random": "#E45756", "degree": "#72B7B2", "recency": "#F2CF5B"}
    for baseline in markers:
        axes[0].scatter(
            [1], [validation[f"{baseline}_sufficiency_error"].median()],
            marker=markers[baseline], s=65, color=colors[baseline], label=f"{baseline} @ locked k",
        )
        axes[1].scatter(
            [1], [validation[f"{baseline}_comprehensiveness"].median()],
            marker=markers[baseline], s=65, color=colors[baseline], label=f"{baseline} @ locked k",
        )
    axes[0].axhline(lock["sufficiency_threshold"], color="black", ls="--", lw=1, label="Sufficiency threshold")
    axes[0].set(title="Sufficiency error (lower is better)", xlabel="Requested top-k edges", ylabel="Median error")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(title="Comprehensiveness (higher is better)", xlabel="Requested top-k edges", ylabel="Median score drop")
    for ax in axes:
        ax.set_xticks(tuning["top_k"].tolist())
        ax.legend(fontsize=7)
    fig.suptitle("Validation fidelity–sparsity trade-off; baselines evaluated at locked k=1", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIGURE_FILES[1])


def figure_03(results: dict) -> None:
    stability = results["stability"]["aggregate"]
    similarity_keys = [
        ("Explainer seed\nedge Jaccard", "explainer_seed_edge_jaccard"),
        ("Explainer seed\nnode Jaccard", "explainer_seed_node_jaccard"),
        ("Explainer seed\nfeature Spearman", "explainer_seed_feature_spearman"),
        ("Model seed\nedge Jaccard", "model_seed_edge_jaccard"),
        ("Model seed\nnode Jaccard", "model_seed_node_jaccard"),
        ("Model seed\nfeature Spearman", "model_seed_feature_spearman"),
        ("Sampling seed\nneighborhood Jaccard", "sampling_seed_neighborhood_jaccard"),
        ("Sampling seed\nexplanation Jaccard", "sampling_seed_explanation_jaccard"),
    ]
    score_keys = [
        ("Explainer\nsufficiency", "explainer_seed_sufficiency_std"),
        ("Explainer\ncomprehensiveness", "explainer_seed_comprehensiveness_std"),
        ("Model\nsufficiency", "model_seed_sufficiency_std"),
        ("Model\ncomprehensiveness", "model_seed_comprehensiveness_std"),
        ("Sampling\nsufficiency", "sampling_seed_sufficiency_std"),
        ("Sampling\ncomprehensiveness", "sampling_seed_comprehensiveness_std"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    medians = [stability[key]["median"] for _, key in similarity_keys]
    x = np.arange(len(similarity_keys))
    bars = axes[0].bar(x, medians, color="#4C78A8", alpha=0.85)
    axes[0].axhline(
        stability["operational_edge_stability_threshold"],
        color="#E45756", ls="--", label="Edge threshold",
    )
    axes[0].set_ylim(0, 1.08)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for label, _ in similarity_keys], fontsize=7)
    axes[0].set_ylabel("Median similarity / correlation")
    axes[0].set_title("Which nodes, events and features stay the same?")
    axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    axes[0].legend(fontsize=8)
    score_medians = [stability[key]["median"] for _, key in score_keys]
    sx = np.arange(len(score_keys))
    score_bars = axes[1].bar(sx, score_medians, color="#72B7B2", alpha=0.85)
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels([label for label, _ in score_keys], fontsize=7)
    axes[1].set_ylabel("Median standard deviation of score metric")
    axes[1].set_title("How much do fidelity scores move across seeds?")
    axes[1].bar_label(score_bars, fmt="%.4f", padding=2, fontsize=7)
    fig.suptitle("Explanation stability across 12 validation targets (3 seeds per audit)")
    fig.tight_layout()
    save_figure(fig, FIGURE_FILES[2])


def figure_04(feature_rows: list[dict]) -> dict:
    feature_names = [f"F{index + 1:02d}" for index in range(34)] + ["community-risk"]
    groups = []
    matrix = []
    for split in ["validation", "test"]:
        for cohort in COHORT_ORDER:
            values = [
                row["values"] for row in feature_rows
                if row["split"] == split and row["cohort"] == cohort
            ]
            split_label = "val" if split == "validation" else "test"
            groups.append(f"{split_label} · {COHORT_LABELS[cohort]}")
            matrix.append(np.mean(np.asarray(values, dtype=float), axis=0))
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=float(np.quantile(matrix, 0.98)))
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xticks(range(35))
    ax.set_xticklabels(feature_names, rotation=75, ha="right", fontsize=7)
    ax.axvline(33.5, color="white", lw=1.5)
    ax.set_title("Mean GNNExplainer feature importance by split and cohort")
    fig.colorbar(image, ax=ax, label="Mean mask importance", fraction=0.025)
    fig.tight_layout()
    save_figure(fig, FIGURE_FILES[3])
    return {
        group: {name: float(value) for name, value in zip(feature_names, row)}
        for group, row in zip(groups, matrix)
    }


def figure_05(analysis: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    grouped_boxplot(
        axes[0], analysis, "community_risk_prior_score_delta",
        "Score change after replacing community-risk with train prior",
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Original fraud score − counterfactual score")
    for risky, color, label in [(False, "#4C78A8", "Other community"), (True, "#E45756", "Risky community")]:
        frame = analysis[analysis["is_risky_community"] == risky]
        axes[1].scatter(
            frame["community_train_fraud_lift"],
            frame["community_risk_prior_score_delta"],
            s=35, alpha=0.75, color=color, label=label,
        )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Target community train-only fraud lift (log scale)")
    axes[1].set_ylabel("Original fraud score − counterfactual score")
    axes[1].set_title("Community association and counterfactual response")
    axes[1].legend()
    fig.suptitle("Community-risk counterfactual (association, not causality)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIGURE_FILES[4])


def case_row(results: dict, target_node_id: int) -> dict:
    return next(
        row for row in results["main_explanations"]
        if int(row["target_node_id"]) == int(target_node_id)
    )


def plot_case(ax, explanation: dict, raw, community_ids: np.ndarray, labels: np.ndarray, subtitle: str) -> None:
    prefix = raw_prefix(explanation)
    node_ids = raw[prefix + "node_ids"].astype(np.int64)
    edges = raw[prefix + "edge_index_global"].astype(np.int64)
    edge_mask = raw[prefix + "edge_mask"].astype(float)
    edge_type = raw[prefix + "edge_type"].astype(np.int64)
    edge_delta = raw[prefix + "edge_delta"].astype(float)
    selected = set(int(value) for value in explanation["selected_event_positions"])
    target = int(explanation["target_node_id"])
    other_nodes = [int(value) for value in node_ids if int(value) != target]
    positions = {target: np.array([0.0, 0.0])}
    for index, node_id in enumerate(other_nodes):
        angle = 2 * math.pi * index / max(1, len(other_nodes))
        positions[node_id] = np.array([math.cos(angle), math.sin(angle)])
    max_mask = max(float(edge_mask.max()) if edge_mask.size else 1.0, 1e-8)
    for edge_position in range(edges.shape[1]):
        source, destination = int(edges[0, edge_position]), int(edges[1, edge_position])
        start, end = positions[source], positions[destination]
        is_selected = edge_position in selected
        width = 0.7 + 3.8 * float(edge_mask[edge_position]) / max_mask
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#222222" if is_selected else "#B8B8B8",
                "lw": width if is_selected else max(0.6, width * 0.45),
                "alpha": 0.95 if is_selected else 0.5,
                "connectionstyle": f"arc3,rad={0.05 * ((edge_position % 3) - 1)}",
            },
        )
        if is_selected:
            midpoint = (start + end) / 2
            ax.text(
                midpoint[0], midpoint[1],
                f"type={edge_type[edge_position]}\nΔ={edge_delta[edge_position]:.0f}",
                fontsize=6, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.8, "ec": "none"},
            )
    label_colors = {0: "#4C78A8", 1: "#E45756", 2: "#9D9D9D", 3: "#B279A2"}
    for node_id in node_ids:
        node_id = int(node_id)
        is_target = node_id == target
        ax.scatter(
            [positions[node_id][0]], [positions[node_id][1]],
            s=180 if is_target else 90,
            marker="D" if is_target else "o",
            c=label_colors.get(int(labels[node_id]), "#B279A2"),
            edgecolors="black" if is_target else "white",
            linewidths=1.3 if is_target else 0.7,
            zorder=4,
        )
        ax.text(positions[node_id][0], positions[node_id][1] - 0.13, str(node_id), fontsize=6, ha="center", va="top")
    target_community = int(community_ids[target])
    selected_internal = 0
    for edge_position in selected:
        source, destination = int(edges[0, edge_position]), int(edges[1, edge_position])
        selected_internal += int(
            community_ids[source] == target_community
            and community_ids[destination] == target_community
        )
    ax.set_title(
        f"{subtitle}\nnode={target}, score={explanation['fraud_score']:.3f}, "
        f"selected internal={selected_internal}/{len(selected)}",
        fontsize=9,
    )
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def figure_06(results, cases, raw, community_ids, labels) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, cohort, node_id in zip(
        axes.flat, COHORT_ORDER, cases["representative_target_node_ids"]
    ):
        plot_case(
            ax, case_row(results, node_id), raw, community_ids, labels,
            COHORT_LABELS[cohort],
        )
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E45756", markeredgecolor="black", markersize=8, label="Target (diamond)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", markersize=7, label="Normal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E45756", markersize=7, label="Fraud"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9D9D9D", markersize=7, label="Background"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B279A2", markersize=7, label="Unlabeled/other"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Rule-selected validation case studies — local prediction explanations, not confirmed fraud rings",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_figure(fig, FIGURE_FILES[5])


def figure_07(results, cases, raw, community_ids, labels) -> None:
    failure_ids = cases["failure_target_node_ids"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_case(
        axes[0], case_row(results, failure_ids["false_alarm_like"]), raw,
        community_ids, labels, "False-alarm-like failure",
    )
    plot_case(
        axes[1], case_row(results, failure_ids["miss_like"]), raw,
        community_ids, labels, "Miss-like failure",
    )
    fig.suptitle(
        "Predefined validation failure cases — local explanations, not confirmed fraud rings",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, FIGURE_FILES[6])


def render_report(
    results: dict,
    lock: dict,
    analysis: pd.DataFrame,
    cohort_summary: dict,
    cases: dict,
    feature_summary: dict,
) -> str:
    selected = lock["selected"]
    stability = results["stability"]["aggregate"]
    validation = analysis[analysis["split"] == "validation"]
    test = analysis[analysis["split"] == "test"]
    risky_rate = analysis["is_risky_community"].mean()
    internal_by_cohort = analysis.groupby("cohort")["selected_internal_edge_ratio"].mean()
    delta_by_cohort = analysis.groupby("cohort")["community_risk_prior_score_delta"].mean()
    risk_feature_means = {
        cohort: float(
            analysis.loc[analysis["cohort"] == cohort, "community_risk_feature_importance"].mean()
        )
        for cohort in COHORT_ORDER
    }
    failure_false = cases["failure_target_node_ids"]["false_alarm_like"]
    failure_miss = cases["failure_target_node_ids"]["miss_like"]
    lines = [
        "# Sprint 5 — Giải thích dự đoán TGAT bằng GNNExplainer",
        "",
        f"<!-- generated-from: {RESULT_PATH.relative_to(PROJECT_ROOT).as_posix()} sha256={file_sha256(RESULT_PATH)} -->",
        "",
        "## Tóm tắt",
        "",
        "Sprint 5 giải thích cục bộ dự đoán node-level của cấu hình C — TGAT + community-risk. "
        "Protocol được chọn trên validation và khóa trước khi tạo explanation cho test. "
        "Kết quả không phải bằng chứng về quan hệ nhân quả, sự phối hợp giữa các tài khoản, "
        "hay một fraud ring đã được xác nhận.",
        "",
        f"Cấu hình đã khóa dùng {selected['config']['epochs']} epoch, learning rate "
        f"{selected['config']['lr']}, top-{selected['edge_top_k']} edge và top-"
        f"{selected['feature_top_k']} feature. Tổng cộng có {len(analysis)} explanation "
        f"({len(validation)} validation, {len(test)} test).",
        "",
        "## Phương pháp",
        "",
        "- Model và checkpoint: cấu hình C, seed 42; checkpoint/hash không thay đổi trong khi giải thích.",
        "- Cohort: high-score fraud, high-score normal, low-score fraud và low-score normal control; mỗi split có 10 node/cohort.",
        "- Chọn protocol: validation-only theo sufficiency–sparsity aggregate; test không tham gia tuning.",
        "- Đánh giá: sufficiency error, comprehensiveness, edge sparsity, random/degree/recency baseline, runtime và RAM.",
        "- Robustness: 12 validation node qua explainer seed, model checkpoint seed và neighborhood seed 42/43/44.",
        "- Liên hệ Sprint 4: risky-community flag, train-only fraud lift, conductance, tỷ lệ selected edge nội bộ community và counterfactual community-risk.",
        "",
        "## Kết quả fidelity và độ ổn định",
        "",
        f"Validation tuning đạt median sufficiency error {selected['validation_metrics']['median_sufficiency_error']:.6f}, "
        f"median comprehensiveness {selected['validation_metrics']['median_comprehensiveness']:.6f} và "
        f"median edge sparsity {selected['validation_metrics']['median_edge_sparsity']:.3f}. "
        "Các neighborhood rất nhỏ khiến top-k lớn hơn thường giữ toàn bộ cạnh; vì vậy top-1 được chọn theo rule subgraph nhỏ nhất còn đủ faithful.",
        "",
        f"Median edge Jaccard theo explainer seed là {stability['explainer_seed_edge_jaccard']['median']:.3f}; "
        f"median feature Spearman chỉ {stability['explainer_seed_feature_spearman']['median']:.3f}. "
        f"Theo model seed, hai giá trị tương ứng là {stability['model_seed_edge_jaccard']['median']:.3f} và "
        f"{stability['model_seed_feature_spearman']['median']:.3f}. Neighborhood/explanation Jaccard theo sampling seed đều có median "
        f"{stability['sampling_seed_neighborhood_jaccard']['median']:.3f}/"
        f"{stability['sampling_seed_explanation_jaccard']['median']:.3f}.",
        "",
        "![Fidelity–sparsity](../artifacts/figures/sprint5/02_fidelity_sparsity.png)",
        "",
        "![Stability](../artifacts/figures/sprint5/03_stability.png)",
        "",
        "## Liên hệ với risky community của Sprint 4",
        "",
        f"Trong 80 target, tỷ lệ thuộc danh sách risky community là {risky_rate:.1%}. "
        "Tỷ lệ selected edge nằm hoàn toàn trong target community theo cohort lần lượt là: "
        + "; ".join(
            f"{COHORT_LABELS[c]} {internal_by_cohort[c]:.1%}" for c in COHORT_ORDER
        )
        + ".",
        "",
        "Thay community-risk bằng global train prior tạo thay đổi fraud score trung bình: "
        + "; ".join(
            f"{COHORT_LABELS[c]} {delta_by_cohort[c]:+.4f}" for c in COHORT_ORDER
        )
        + ". Đây là perturbation của một feature trong model, không phải can thiệp nhân quả lên community.",
        "",
        "Mean mask importance của community-risk theo cohort: "
        + "; ".join(
            f"{COHORT_LABELS[c]} {risk_feature_means[c]:.3f}" for c in COHORT_ORDER
        )
        + ". Kết quả này bổ sung góc nhìn local cho ablation C > A ở Sprint 4, nhưng không tách biệt hoàn toàn tương quan giữa feature và neighborhood.",
        "",
        "![Cohort overview](../artifacts/figures/sprint5/01_cohort_overview.png)",
        "",
        "![Feature importance](../artifacts/figures/sprint5/04_feature_importance.png)",
        "",
        "![Community-risk counterfactual](../artifacts/figures/sprint5/05_risk_counterfactual.png)",
        "",
        "## Case study và failure case",
        "",
        "Bốn case đại diện được chọn hoàn toàn trên validation bằng rule: trong mỗi cohort, lấy explanation không bị gắn failure có comprehensiveness cao nhất, sau đó ưu tiên sufficiency error thấp hơn và node ID nhỏ hơn. "
        f"Các node được chọn là {', '.join(map(str, cases['representative_target_node_ids']))}.",
        "",
        "![Rule-selected local cases](../artifacts/figures/sprint5/06_risky_subgraph_cases.png)",
        "",
        f"Failure false-alarm-like là node {failure_false}; failure miss-like là node {failure_miss}. "
        "Cả hai được chọn bằng rule định lượng trên validation, không chọn thủ công sau khi xem hình.",
        "",
        "![Failure cases](../artifacts/figures/sprint5/07_failure_cases.png)",
        "",
        "## Hạn chế",
        "",
        "- GNNExplainer là post-hoc local explanation; mask quan trọng không đồng nghĩa quan hệ nhân quả.",
        "- DGraphFin không cung cấp ground-truth fraud ring, nên các hình chỉ là candidate risky subgraph quanh một prediction.",
        "- Nhiều sampled neighborhood chỉ có 1–3 event; khi top-1 là toàn bộ neighborhood, explainer và baseline có thể trùng nhau.",
        "- Edge selection ổn định hơn feature ranking; feature Spearman theo explainer seed ở mức trung bình.",
        "- Static NeighborLoader không áp temporal cutoff theo query time; kết quả thuộc bài toán full-history transductive classification.",
        "- Một số low-score fraud/control explanation không vượt operational fidelity threshold; failure được giữ lại thay vì loại khỏi báo cáo.",
        "- Counterfactual community-risk thay một feature nhưng giữ nguyên graph/neighborhood, nên chỉ đo độ nhạy của model trong điều kiện đó.",
        "",
        "## Kết luận và câu hỏi mở",
        "",
        "GNNExplainer tạo được explanation nhỏ và tái lập tốt ở mức edge trong protocol hiện tại, nhưng giá trị bổ sung so với baseline bị giới hạn khi neighborhood quá nhỏ và feature ranking chưa ổn định hoàn toàn. Community-risk có ảnh hưởng không đồng đều giữa cohort; kết quả ủng hộ việc tiếp tục kiểm tra feature này nhưng không đủ để kết luận community gây ra fraud.",
        "",
        "Các câu hỏi mở cho sprint sau:",
        "",
        "1. Integrated Gradients hoặc một attribution method khác có ổn định hơn cho 35 feature không?",
        "2. Temporal cutoff/edge-time perturbation có làm thay đổi candidate subgraph không?",
        "3. Có thể đánh giá ring discovery nếu bổ sung ground truth hoặc investigator feedback không?",
        "4. Có cần so sánh trực tiếp model A và C trên cùng target để tách vai trò community-risk không?",
        "",
        "## Artifact",
        "",
        "- `artifacts/metrics/sprint5_explanation_analysis.csv`: bảng 80 explanation đã nối dữ liệu Sprint 4.",
        "- `artifacts/metrics/sprint5_case_selection.json`: rule và node ID case study/failure case.",
        "- `artifacts/metrics/sprint5_phase45_results.json`: aggregate, checksum và package checks.",
        "- `artifacts/figures/sprint5/01_...png` đến `07_...png`: toàn bộ hình bắt buộc.",
    ]
    return "\n".join(lines) + "\n"


def render_story_report(
    results: dict,
    lock: dict,
    analysis: pd.DataFrame,
    cohort_summary: dict,
    cases: dict,
    feature_summary: dict,
) -> str:
    """Render a reader-first report while keeping the technical audit trail."""
    selected = lock["selected"]
    stability = results["stability"]["aggregate"]
    validation = analysis[analysis["split"] == "validation"]
    test = analysis[analysis["split"] == "test"]
    sanity = results["sanity_checks"]["randomized_model"]
    candidate_ids = cases["candidate_risky_subgraph_target_node_ids"]
    failure_false = cases["failure_target_node_ids"]["false_alarm_like"]
    failure_miss = cases["failure_target_node_ids"]["miss_like"]
    risky_rate = float(analysis["is_risky_community"].mean())
    zero_edge_count = int(analysis["edge_explanation_unavailable"].sum())
    mean_runtime = float(analysis["runtime_seconds"].mean())
    max_peak_rss = float(analysis["peak_rss_mib"].max())

    cohort_names = {
        "high_score_fraud": "Fraud được model chấm cao",
        "high_score_normal": "Normal nhưng model chấm cao",
        "low_score_fraud": "Fraud bị model chấm thấp",
        "low_score_normal_control": "Normal điểm thấp đối chứng",
    }

    cohort_table = [
        "| Tập dữ liệu | Nhóm node | Số node | Sai lệch khi chỉ giữ phần giải thích ↓ | Mức giảm khi bỏ phần giải thích ↑ | Số lời giải thích không đạt |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, split_label in (("validation", "Validation"), ("test", "Test")):
        for cohort in COHORT_ORDER:
            frame = analysis[
                (analysis["split"] == split) & (analysis["cohort"] == cohort)
            ]
            cohort_table.append(
                f"| {split_label} | {cohort_names[cohort]} | {len(frame)} | "
                f"{frame['sufficiency_error'].median():.6f} | "
                f"{frame['comprehensiveness'].median():.6f} | "
                f"{int(frame['failure_flag'].sum())} |"
            )

    comparison_table = [
        "| Cách chọn một event trên validation | Sai lệch khi chỉ giữ event ↓ | Mức giảm khi bỏ event ↑ |",
        "|---|---:|---:|",
        (
            f"| GNNExplainer | {validation['sufficiency_error'].median():.6f} | "
            f"{validation['comprehensiveness'].median():.6f} |"
        ),
    ]
    for baseline, label in (
        ("random", "Chọn ngẫu nhiên"),
        ("degree", "Chọn theo degree"),
        ("recency", "Chọn event gần nhất"),
    ):
        comparison_table.append(
            f"| {label} | {validation[f'{baseline}_sufficiency_error'].median():.6f} | "
            f"{validation[f'{baseline}_comprehensiveness'].median():.6f} |"
        )

    stability_table = [
        "| Nguồn thay đổi | Event được chọn giống nhau | Node hàng xóm giống nhau | Thứ hạng feature giống nhau | Độ lệch chuẩn keep-error | Độ lệch chuẩn remove-effect |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Explainer seed | {stability['explainer_seed_edge_jaccard']['median']:.3f} | "
            f"{stability['explainer_seed_node_jaccard']['median']:.3f} | "
            f"{stability['explainer_seed_feature_spearman']['median']:.3f} | "
            f"{stability['explainer_seed_sufficiency_std']['median']:.6f} | "
            f"{stability['explainer_seed_comprehensiveness_std']['median']:.6f} |"
        ),
        (
            f"| Model checkpoint | {stability['model_seed_edge_jaccard']['median']:.3f} | "
            f"{stability['model_seed_node_jaccard']['median']:.3f} | "
            f"{stability['model_seed_feature_spearman']['median']:.3f} | "
            f"{stability['model_seed_sufficiency_std']['median']:.6f} | "
            f"{stability['model_seed_comprehensiveness_std']['median']:.6f} |"
        ),
        (
            f"| Neighborhood seed | {stability['sampling_seed_explanation_jaccard']['median']:.3f} | "
            f"{stability['sampling_seed_node_jaccard']['median']:.3f} | — | "
            f"{stability['sampling_seed_sufficiency_std']['median']:.6f} | "
            f"{stability['sampling_seed_comprehensiveness_std']['median']:.6f} |"
        ),
    ]

    risk_delta = analysis.groupby("cohort")["community_risk_prior_score_delta"].mean()
    risk_importance = analysis.groupby("cohort")[
        "community_risk_feature_importance"
    ].mean()
    internal_ratio = analysis.groupby("cohort")[
        "selected_internal_edge_ratio"
    ].mean()
    candidate_text = (
        ", ".join(str(value) for value in candidate_ids)
        if candidate_ids
        else "không có node nào"
    )

    lines = [
        "# Sprint 5 — Mô hình đã dựa vào đâu để đánh giá một node là đáng ngờ?",
        "",
        f"<!-- generated-from: {RESULT_PATH.relative_to(PROJECT_ROOT).as_posix()} sha256={file_sha256(RESULT_PATH)} -->",
        "",
        "Sprint 5 này sử dụng GNNExplainer để xem mô hình dựa vào thông tin nào khi đánh giá "
        "một node là đáng ngờ, và quan trọng hơn, lời giải thích đó có đủ ổn định và đáng tin "
        "để sử dụng hay không.",
        "",
        "Sprint trước đã chốt cấu hình C — TGAT kết hợp community-risk — bằng kết quả validation. "
        "Sprint 5 không huấn luyện lại detector và cũng không tìm một model tốt hơn. Ta giữ nguyên "
        "model đó, quan sát cách nó đưa ra từng dự đoán, rồi chủ động thử làm yếu lời giải thích để "
        "xem dự đoán có thật sự phụ thuộc vào phần được chọn hay không.",
        "",
        "## Đường đi của thí nghiệm",
        "",
        "Thí nghiệm được thực hiện theo thứ tự sau:",
        "",
        "1. Khóa model C và checkpoint seed 42; kiểm tra lại hash, validation AP và đầu ra của adapter.",
        "2. Trên validation, tạo bốn nhóm node để quan sát cả trường hợp model làm tốt lẫn trường hợp dễ sai.",
        "3. Chỉ dùng validation để chọn cấu hình GNNExplainer và số event cần giữ.",
        "4. Khóa cấu hình, sau đó áp dụng nguyên vẹn cho 40 node validation và 40 node test.",
        "5. Đổi explainer seed, checkpoint model và neighborhood seed trên 12 node audit để đo độ ổn định.",
        "6. So lời giải thích với cách chọn event ngẫu nhiên/đơn giản và với một model có trọng số ngẫu nhiên.",
        "",
        "Bốn nhóm node gồm:",
        "",
        "- fraud được model chấm cao: trường hợp đúng và dễ phát hiện;",
        "- normal nhưng model chấm cao: trường hợp giống cảnh báo nhầm;",
        "- fraud bị model chấm thấp: trường hợp giống bỏ sót;",
        "- normal điểm thấp: nhóm đối chứng được ghép gần theo degree và community size.",
        "",
        "Label chỉ được dùng để chia nhóm và đánh giá sau khi model đã khóa. Label của hàng xóm "
        "không đi vào model. Node không có sampled edge vẫn được giữ theo đúng cohort rule; trong "
        f"lần chạy này có {zero_edge_count} trên 80 node thuộc trường hợp đó.",
        "",
        "## Một lời giải thích trong Sprint 5 là gì?",
        "",
        "Với mỗi node, TGAT chỉ nhìn một neighborhood cục bộ một hop. GNNExplainer gán mức quan trọng "
        "cho từng event và từng feature trong neighborhood đó. Cấu hình đã khóa giữ "
        f"{selected['edge_top_k']} event quan trọng nhất và {selected['feature_top_k']} trong 35 feature. "
        f"Điều này tương ứng feature sparsity {analysis['feature_sparsity'].median():.1%}: phần lớn feature "
        "không nằm trong mask nhị phân cuối cùng.",
        "",
        "“Event quan trọng” ở đây chỉ có nghĩa là event ảnh hưởng đến phép tính của model. Nó không "
        "chứng minh giao dịch đó gây ra gian lận, và cũng không chứng minh các tài khoản đang phối hợp.",
        "",
        "## Kiểm tra 1 — Giữ hoặc bỏ phần được chọn có làm dự đoán thay đổi không?",
        "",
        "Ta dùng hai phép kiểm tra:",
        "",
        "- **Sai lệch khi chỉ giữ phần giải thích** (sufficiency error): càng gần 0 càng tốt. Nghĩa là "
        "chỉ giữ phần được chọn mà model vẫn cho kết quả gần như cũ.",
        "- **Mức giảm khi bỏ phần giải thích** (comprehensiveness): nên lớn hơn 0. Nghĩa là bỏ phần "
        "được chọn thì độ tin của model vào lớp đang dự đoán giảm xuống.",
        "",
        "Một lời giải thích được đánh dấu đạt khi sai lệch giữ lại không quá 0,05 và mức giảm khi bỏ "
        "phần giải thích là số dương. Các trường hợp không đạt vẫn được giữ trong báo cáo.",
        "",
        *cohort_table,
        "",
        f"Tổng cộng có {int(analysis['failure_flag'].sum())}/80 lời giải thích không vượt cả hai điều kiện. "
        "Phần lớn failure nằm ở các nhóm fraud điểm thấp và normal đối chứng; đây chính là các trường "
        "hợp không nên diễn giải quá tự tin.",
        "",
        "Cấu hình được chọn trên 12 node validation có median keep-error "
        f"{selected['validation_metrics']['median_sufficiency_error']:.6f}, remove-effect "
        f"{selected['validation_metrics']['median_comprehensiveness']:.6f} và edge sparsity "
        f"{selected['validation_metrics']['median_edge_sparsity']:.3f}. Top-1 được chọn vì đó là "
        "subgraph nhỏ nhất vẫn qua ngưỡng giữ lại.",
        "",
        "### So với cách chọn event đơn giản",
        "",
        *comparison_table,
        "",
        "GNNExplainer không thắng rõ ràng ở mọi nhóm. Nhiều neighborhood chỉ có rất ít event, nên "
        "top-1 đôi khi cũng chính là toàn bộ neighborhood và các baseline có thể chọn trùng event. "
        "Do đó kết quả cho thấy lời giải thích có thể giữ được dự đoán, nhưng giá trị bổ sung so với "
        "heuristic đơn giản còn hạn chế.",
        "",
        "## Kiểm tra 2 — Lời giải thích có lặp lại khi đổi seed hoặc checkpoint không?",
        "",
        "Bảng dưới dùng thang 0–1 cho độ giống nhau: 1 là giống hoàn toàn. Feature dùng tương quan "
        "thứ hạng; càng gần 1 càng ổn định. Hai cột cuối cho biết metric fidelity dao động bao nhiêu; "
        "càng nhỏ càng tốt.",
        "",
        *stability_table,
        "",
        f"Event và node được chọn ổn định hơn thứ hạng feature. Với explainer seed, feature-rank "
        f"chỉ đạt median {stability['explainer_seed_feature_spearman']['median']:.3f}; vì vậy không nên "
        "đọc một chênh lệch nhỏ giữa hai feature như một kết luận chắc chắn.",
        "",
        "Negative control dùng một model cùng kiến trúc nhưng trọng số ngẫu nhiên trên node "
        f"{sanity['target_node_id']} ({sanity['sampled_event_count']} sampled events). Ranking event "
        f"có trùng hoàn toàn hay không: **{sanity['exact_edge_ranking_match']}**; Spearman "
        f"{sanity['edge_rank_spearman']:.3f}, top-3 Jaccard {sanity['top_3_edge_jaccard']:.3f}. "
        "Control vượt tiêu chí tối thiểu vì ranking không trùng hoàn toàn, nhưng Spearman còn cao; "
        "vì vậy đây chỉ là sanity check có giới hạn, chưa phải bằng chứng tách biệt mạnh giữa model "
        "đã học và model ngẫu nhiên.",
        "",
        "## Kiểm tra 3 — Community-risk đóng vai trò gì?",
        "",
        f"Chỉ {risky_rate:.1%} target nằm trong danh sách risky community của Sprint 4. "
        "Trong local subgraph được chọn, tỷ lệ event nằm hoàn toàn trong community của target là "
        + "; ".join(
            f"{cohort_names[c]} {internal_ratio[c]:.1%}" for c in COHORT_ORDER
        )
        + ". Đây là mối liên hệ cấu trúc, không phải bằng chứng community gây ra fraud.",
        "",
        "Để kiểm tra trực tiếp feature community-risk, ta thay riêng giá trị của target bằng tỷ lệ "
        "fraud chung trong train, giữ nguyên graph và mọi feature khác. Thay đổi fraud score trung bình là "
        + "; ".join(
            f"{cohort_names[c]} {risk_delta[c]:+.4f}" for c in COHORT_ORDER
        )
        + ".",
        "",
        "Mức quan trọng trung bình mà GNNExplainer gán cho community-risk là "
        + "; ".join(
            f"{cohort_names[c]} {risk_importance[c]:.3f}" for c in COHORT_ORDER
        )
        + ". Feature này có ảnh hưởng, đặc biệt ở nhóm fraud được chấm cao, nhưng ảnh hưởng không "
        "đồng đều và không đủ để nói community là nguyên nhân.",
        "",
        "## Những case nào nên và không nên xem là ứng viên?",
        "",
        "Một node chỉ được gắn nhãn **candidate risky subgraph** khi thuộc nhóm điểm cao, qua kiểm tra "
        "giữ/bỏ, nằm trong tập audit và đạt edge stability từ 0,5 trở lên. Theo rule đã khóa, các node "
        f"đạt điều kiện là: {candidate_text}. Đây vẫn chỉ là ứng viên để xem xét, không phải fraud ring đã xác nhận.",
        "",
        "Bốn hình minh họa dưới đây được chọn bằng rule trên validation, mỗi cohort một case. Chúng giúp "
        "quan sát model trong bốn tình huống khác nhau; không phải cả bốn đều là candidate risky subgraph.",
        "",
        "![Bốn lời giải thích đại diện](../artifacts/figures/sprint5/06_risky_subgraph_cases.png)",
        "",
        f"Hai failure case được chọn trước theo rule định lượng: node {failure_false} giống cảnh báo nhầm "
        f"và node {failure_miss} giống bỏ sót. Việc giữ lại hai case này giúp tránh chỉ trình bày hình đẹp.",
        "",
        "![Hai failure case](../artifacts/figures/sprint5/07_failure_cases.png)",
        "",
        "## Kết luận",
        "",
        "Từ thí nghiệm này có thể kết luận:",
        "",
        "- Pipeline có thể tạo local explanation cho model C và giữ đầy đủ provenance để tái kiểm tra.",
        "- Phần event của lời giải thích khá ổn định trong tập audit hiện tại.",
        "- Phần feature kém ổn định hơn; chỉ nên đọc các pattern lớn ở cấp cohort.",
        "- Community-risk có ảnh hưởng đến một số dự đoán nhưng không chi phối đồng đều mọi nhóm.",
        "- GNNExplainer có fidelity tốt ở nhiều node, nhưng lợi thế so với baseline bị giới hạn bởi "
        "neighborhood quá nhỏ và không phải mọi lời giải thích đều vượt ngưỡng vận hành.",
        "",
        "Không thể kết luận:",
        "",
        "- event hoặc community được chọn là nguyên nhân gây fraud;",
        "- các node trong hình đang phối hợp với nhau;",
        "- candidate risky subgraph là một fraud ring đã được xác nhận;",
        "- score sigmoid là xác suất đã được calibration.",
        "",
        "Kết quả chỉ áp dụng cho node-level, full-history transductive classification trên DGraphFin "
        "và local post-hoc explanation của model đã khóa.",
        "",
        "## Dấu vết tái lập",
        "",
        f"- Model: variant C; checkpoint chính seed 42; audit checkpoint seed {lock['model_seeds']}.",
        f"- Explainer seed: {lock['explainer_seeds']}; neighborhood seed: {lock['neighborhood_seeds']}.",
        f"- Cấu hình khóa: {selected['config']['epochs']} epoch, learning rate "
        f"{selected['config']['lr']}, top-{selected['edge_top_k']} event, top-{selected['feature_top_k']} feature.",
        f"- Quy mô: {len(validation)} validation + {len(test)} test; 12 validation node audit.",
        f"- Runtime trung bình mỗi explanation: {mean_runtime:.3f} giây; peak RSS lớn nhất ghi nhận: "
        f"{max_peak_rss:.1f} MiB.",
        "- Callback gradient của PyG xác nhận node mask và mọi edge mask khả dụng đều nhận gradient "
        "khác 0; detector giữ nguyên state trước và sau giải thích.",
        "- Raw artifact lưu global node/event ID, temporal metadata, continuous mask, binary selected "
        "mask, full/keep/remove logit-score, config và seed cho từng explanation.",
        "",
        "Các artifact chính:",
        "",
        "- artifacts/metrics/sprint5_explainer_protocol_lock.json: protocol khóa trước test.",
        "- artifacts/metrics/sprint5_explainer_results.json: metric và provenance của 80 explanation.",
        "- artifacts/metrics/sprint5_explanations.npz: mask và ánh xạ ID ở mức raw.",
        "- artifacts/metrics/sprint5_explanation_analysis.csv: bảng phân tích đã nối Sprint 4.",
        "- artifacts/metrics/sprint5_case_selection.json: rule chọn case, failure và candidate.",
        "- artifacts/figures/sprint5/01_...png đến 07_...png: đủ bảy hình kiểm toán; report chỉ nhúng hai hình dễ đọc nhất.",
        "",
        "## Câu hỏi để tiếp tục",
        "",
        "Các hướng Integrated Gradients, so sánh model A/C, time perturbation, calibration hoặc "
        "điều tra fraud ring cần một thí nghiệm riêng ở sprint sau; chúng không được dùng để thay đổi "
        "kết luận Sprint 5 này.",
    ]
    return "\n".join(lines) + "\n"


def notebook_has_no_errors(path: Path) -> bool:
    notebook = json_load(path)
    return not any(
        output.get("output_type") == "error"
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )


def build_package_checks(analysis: pd.DataFrame) -> dict:
    notebook_path = PROJECT_ROOT / "notebooks" / "06_gnn_explainer.ipynb"
    notebook_text = notebook_path.read_text(encoding="utf-8")
    notebook_readme = (PROJECT_ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    protocol = (DOC_DIR / "experiment_protocol.md").read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    results = json_load(RESULT_PATH)
    manifest = json_load(MANIFEST_PATH)
    cases = json_load(CASE_PATH)
    raw_files = np.load(RAW_PATH, allow_pickle=False).files
    required_analysis_columns = {
        "feature_sparsity",
        "edge_explanation_unavailable",
        "stability_audited",
        "explainer_seed_edge_jaccard_median",
        "explainer_seed_node_jaccard_median",
        "stability_passed",
        "model_seed",
        "explainer_seed",
        "neighborhood_seed",
        "explainer_config_id",
    }
    checks = {
        "phase3_gate_passed": bool(results["gate_passed"]),
        "analysis_has_80_unique_targets": len(analysis) == 80
        and analysis["target_node_id"].nunique() == 80,
        "analysis_has_40_validation_and_40_test": analysis["split"].value_counts().to_dict()
        == {"validation": 40, "test": 40},
        "all_seven_figures_exist": all((FIGURE_DIR / name).exists() for name in FIGURE_FILES),
        "all_seven_figures_nonempty": all(
            (FIGURE_DIR / name).stat().st_size > 10_000 for name in FIGURE_FILES
        ),
        "report_generated_from_artifact": "generated-from:" in report
        and "sprint5_explainer_results.json" in report,
        "report_has_reader_first_story": (
            "## Đường đi của thí nghiệm" in report
            and "## Kết luận" in report
            and "Không thể kết luận:" in report
        ),
        "analysis_has_required_sparsity_stability_provenance": (
            required_analysis_columns <= set(analysis.columns)
        ),
        "manifest_has_all_three_checkpoint_locks": (
            len(manifest.get("model", {}).get("checkpoints", [])) == 3
        ),
        "protocol_checksum_consistent": (
            results["protocol_lock_sha256"] == file_sha256(LOCK_PATH)
            and manifest["protocol_lock_sha256"] == file_sha256(LOCK_PATH)
        ),
        "raw_has_80_binary_edge_and_feature_masks": (
            sum(name.endswith("_selected_edge_mask") for name in raw_files) == 80
            and sum(name.endswith("_selected_feature_mask") for name in raw_files) == 80
        ),
        "candidate_rule_uses_fidelity_and_stability": (
            "fidelity" in cases["candidate_risky_subgraph_rule"]
            and "stability" in cases["candidate_risky_subgraph_rule"]
        ),
        "notebook_index_mentions_notebook_06": "06_gnn_explainer.ipynb" in notebook_readme,
        "experiment_protocol_has_sprint5_xai": "## Sprint 5 — GNNExplainer" in protocol,
        "notebook_contains_no_error_output": notebook_has_no_errors(notebook_path),
        "notebook_default_expensive_flags_are_false": (
            "os.getenv('RUN_SPRINT5_FEASIBILITY', '0')" in notebook_text
            and "os.getenv('RUN_SPRINT5_EXPLANATIONS', '0')" in notebook_text
            and "os.getenv('RUN_SPRINT5_ANALYSIS', '0')" in notebook_text
        ),
        "raw_masks_load_without_pickle": bool(raw_files),
    }
    return checks


def update_manifest(result: dict) -> None:
    manifest = json_load(MANIFEST_PATH)
    manifest.update(
        {
            "status": "phase5_complete" if result["gate_passed"] else "phase5_needs_attention",
            "phase": 5,
            "updated_at": result["completed_at"],
            "phase45_result_path": ANALYSIS_RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "analysis_path": ANALYSIS_CSV_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "case_selection_path": CASE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "report_path": REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "figure_paths": [
                (FIGURE_DIR / name).relative_to(PROJECT_ROOT).as_posix()
                for name in FIGURE_FILES
            ],
            "phase45_gate_checks": result["gate_checks"],
        }
    )
    json_dump(MANIFEST_PATH, manifest)


def run_analysis_and_report() -> dict:
    set_plot_style()
    (
        results,
        lock,
        targets,
        community_table,
        community_ids,
        labels,
        raw,
    ) = load_inputs()
    analysis, feature_rows = build_analysis(
        results, targets, community_table, community_ids, raw
    )
    ANALYSIS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(ANALYSIS_CSV_PATH, index=False, encoding="utf-8")
    cohort_summary = aggregate_cohorts(analysis)
    cases = select_cases(analysis)
    json_dump(CASE_PATH, cases)

    print("Create Sprint 5 figures 01-07...", flush=True)
    figure_01(analysis)
    figure_02(analysis, lock)
    figure_03(results)
    feature_summary = figure_04(feature_rows)
    figure_05(analysis)
    figure_06(results, cases, raw, community_ids, labels)
    figure_07(results, cases, raw, community_ids, labels)

    report = render_story_report(
        results, lock, analysis, cohort_summary, cases, feature_summary
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    gate_checks = build_package_checks(analysis)
    gate_passed = all(gate_checks.values())
    result = {
        "schema_version": 1,
        "status": "phase5_complete" if gate_passed else "phase5_needs_attention",
        "gate_passed": gate_passed,
        "completed_at": datetime.now().astimezone().isoformat(),
        "source_artifacts": {
            "explainer_results": {
                "path": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(RESULT_PATH),
            },
            "protocol_lock": {
                "path": LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(LOCK_PATH),
            },
            "raw_masks": {
                "path": RAW_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(RAW_PATH),
            },
            "sprint4_community_table": {
                "path": COMMUNITY_TABLE_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(COMMUNITY_TABLE_PATH),
            },
        },
        "selected_protocol": lock["selected"],
        "analysis": {
            "path": ANALYSIS_CSV_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(ANALYSIS_CSV_PATH),
            "row_count": int(len(analysis)),
            "validation_count": int((analysis["split"] == "validation").sum()),
            "test_count": int((analysis["split"] == "test").sum()),
            "cohort_summary": cohort_summary,
        },
        "case_selection": cases,
        "case_selection_sha256": file_sha256(CASE_PATH),
        "feature_importance_by_group": feature_summary,
        "figures": [
            {
                "path": (FIGURE_DIR / name).relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(FIGURE_DIR / name),
                "bytes": (FIGURE_DIR / name).stat().st_size,
            }
            for name in FIGURE_FILES
        ],
        "report": {
            "path": REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(REPORT_PATH),
        },
        "gate_checks": gate_checks,
    }
    json_dump(ANALYSIS_RESULT_PATH, result)
    update_manifest(result)
    print(
        f"Analysis and report {'PASSED' if gate_passed else 'NEED ATTENTION'} | "
        f"result={ANALYSIS_RESULT_PATH.relative_to(PROJECT_ROOT)}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    ANALYSIS_RESULT = run_analysis_and_report()
    reassessment_path = METRIC_DIR / "sprint5_explainer_reassessment.json"
    if reassessment_path.exists():
        import runpy

        runpy.run_path(
            str(PROJECT_ROOT / "scripts" / "sprint5_reassessment_report.py"),
            run_name="__main__",
        )
