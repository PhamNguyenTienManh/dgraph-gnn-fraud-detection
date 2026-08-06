"""Run logged GCN and GraphSAGE baseline experiments."""

import argparse
import json
from pathlib import Path

from dgraph_fraud.training.config import load_experiment_config
from dgraph_fraud.training.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train và so sánh GCN/GraphSAGE")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/baseline_full.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    run_dir = run_experiment(config)
    comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    print(json.dumps({"run_dir": str(run_dir), "models": comparison["models"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
