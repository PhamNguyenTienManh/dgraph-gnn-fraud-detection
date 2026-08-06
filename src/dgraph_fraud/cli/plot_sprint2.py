"""Create reproducible charts from the Sprint 2 report results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


# Values are mean/std over three seeds from docs/sprint2_report.md.
FEATURE_RESULTS = {
    "GCN raw 17D": {"test_auc": (0.681957, 0.000872), "test_ap": (0.027691, 0.000117)},
    "GCN zero-indicator 34D": {"test_auc": (0.685631, 0.000929), "test_ap": (0.028441, 0.000212)},
    "GraphSAGE raw 17D": {"test_auc": (0.758562, 0.000857), "test_ap": (0.038309, 0.000074)},
    "GraphSAGE zero-indicator 34D": {"test_auc": (0.758553, 0.000816), "test_ap": (0.038601, 0.000242)},
}

GCN_ABLATION_DELTAS = {
    "Global z-score": (-0.001177, -0.000399, -0.000177, -0.000959),
    "Dropout 0.5 -> 0.0": (0.000985, -0.000044, -0.000943, 0.000392),
    "Weight decay 5e-4 -> 5e-7": (0.000437, 0.000345, 0.000327, 0.000742),
    "Learning rate 0.001 -> 0.01": (-0.008579, -0.001451, -0.012768, -0.002485),
    "Directed -> undirected": (0.069894, 0.009955, 0.064261, 0.006807),
}

DIRECTION_RESULTS = {
    "GCN directed": {"test_auc": (0.685631, 0.000929), "test_ap": (0.028441, 0.000212)},
    "GCN undirected": {"test_auc": (0.749892, 0.000737), "test_ap": (0.035248, 0.000255)},
    "GraphSAGE directed": {"test_auc": (0.758553, 0.000816), "test_ap": (0.038601, 0.000242)},
    "GraphSAGE undirected": {"test_auc": (0.777018, 0.001842), "test_ap": (0.043408, 0.000346)},
}

PROGRESSION_RESULTS = {
    "GCN\nraw 17D\ndirected": (0.681957, 0.027691),
    "GCN\n34D\ndirected": (0.685631, 0.028441),
    "GCN\n34D\nundirected": (0.749892, 0.035248),
    "GraphSAGE\n34D\nundirected": (0.777018, 0.043408),
    "RGCN-BG\n34D\ndirected*": (0.766536, 0.041171),
}

METRIC_LABELS = ("Validation ROC-AUC", "Validation AP", "Test ROC-AUC", "Test AP")


def _load_plotting() -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Matplotlib is required. Install it with: "
            r".\.venv\Scripts\python.exe -m pip install -e '.[viz]'"
        ) from exc
    return plt, np


def _annotate_bars(axis: Any, bars: Any) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            f"{height:.4f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _save(figure: Any, output_dir: Path, filename: str, dpi: int) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def _paired_bar_chart(
    plt: Any,
    np: Any,
    results: dict[str, dict[str, tuple[float, float]]],
    title: str,
    output_dir: Path,
    filename: str,
    dpi: int,
) -> Path:
    labels = list(results)
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, metric, metric_title in zip(
        axes, ("test_auc", "test_ap"), ("Test ROC-AUC", "Test AP"), strict=True
    ):
        values = [results[label][metric][0] for label in labels]
        errors = [results[label][metric][1] for label in labels]
        bars = axis.bar(np.arange(len(labels)), values, yerr=errors, capsize=4, color=colors)
        axis.set_title(metric_title)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
        axis.set_ylim(0, max(values) * 1.18)
        axis.grid(axis="y", alpha=0.25)
        _annotate_bars(axis, bars)
    figure.suptitle(title)
    figure.tight_layout()
    path = _save(figure, output_dir, filename, dpi)
    plt.close(figure)
    return path


def plot_gcn_ablation(plt: Any, np: Any, output_dir: Path, dpi: int) -> Path:
    labels = list(GCN_ABLATION_DELTAS)
    values = np.asarray(list(GCN_ABLATION_DELTAS.values())) * 100
    y = np.arange(len(labels))
    height = 0.18
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    figure, axis = plt.subplots(figsize=(12, 7))
    for index, (metric, color) in enumerate(zip(METRIC_LABELS, colors, strict=True)):
        offset = (index - 1.5) * height
        axis.barh(y + offset, values[:, index], height=height, label=metric, color=color)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Change from GCN 34D directed baseline (percentage points)")
    axis.set_title("One-factor-at-a-time GCN ablation")
    axis.legend(ncols=2)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = _save(figure, output_dir, "02_gcn_ablation.png", dpi)
    plt.close(figure)
    return path


def plot_progression(plt: Any, np: Any, output_dir: Path, dpi: int) -> Path:
    labels = list(PROGRESSION_RESULTS)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, value_index, title, color in (
        (axes[0], 0, "Test ROC-AUC", "#4C78A8"),
        (axes[1], 1, "Test AP", "#F58518"),
    ):
        values = [PROGRESSION_RESULTS[label][value_index] for label in labels]
        axis.plot(x, values, marker="o", linewidth=2.2, markersize=7, color=color)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.18)
        axis.grid(alpha=0.25)
        for position, value in zip(x, values, strict=True):
            axis.annotate(f"{value:.4f}", (position, value), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=8)
    figure.suptitle("Sprint 2 result progression")
    figure.text(
        0.5, 0.01,
        "* RGCN-BG is a separate directed experiment, not the next step of the undirected run.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path = _save(figure, output_dir, "04_result_progression.png", dpi)
    plt.close(figure)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from docs/sprint2_report.md results."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/figures/sprint2"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output resolution.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be a positive integer")
    plt, np = _load_plotting()
    plt.style.use("seaborn-v0_8-whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _paired_bar_chart(
            plt, np, FEATURE_RESULTS,
            "Feature encoding and model comparison (mean +/- std, 3 seeds)",
            args.output_dir, "01_feature_encoding.png", args.dpi,
        ),
        plot_gcn_ablation(plt, np, args.output_dir, args.dpi),
        _paired_bar_chart(
            plt, np, DIRECTION_RESULTS,
            "Effect of graph direction (34D zero-indicator, mean +/- std)",
            args.output_dir, "03_graph_direction.png", args.dpi,
        ),
        plot_progression(plt, np, args.output_dir, args.dpi),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
