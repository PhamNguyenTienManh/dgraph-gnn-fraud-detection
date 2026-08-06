"""Benchmark one directed neighbor-sampling batch on DGraphFin."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dgraph_fraud.data.loader import load_dgraphfin
from dgraph_fraud.data.sampling import DirectedNeighborSampler, NeighborSamplingPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark neighbor sampling DGraphFin")
    parser.add_argument("--data", type=Path, default=Path("data/dgraphfin.npz"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--fanout", type=int, nargs="+", default=[15, 10])
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_started = time.perf_counter()
    dataset = load_dgraphfin(args.data)
    load_seconds = time.perf_counter() - load_started

    policy = NeighborSamplingPolicy(
        num_neighbors=tuple(args.fanout), batch_size=args.batch_size, seed=args.seed
    )
    index_started = time.perf_counter()
    sampler = DirectedNeighborSampler(dataset.edge_index, dataset.num_nodes, policy)
    index_seconds = time.perf_counter() - index_started

    batch = dataset.train_mask[: args.batch_size]
    sample_started = time.perf_counter()
    subgraph = sampler.sample(batch)
    sample_seconds = time.perf_counter() - sample_started

    report = {
        "policy": {
            "direction": "incoming (source -> target)",
            "fanout": list(policy.num_neighbors),
            "batch_size": policy.batch_size,
            "num_workers": policy.num_workers,
            "seed": policy.seed,
        },
        "timing_seconds": {
            "load_dataset": load_seconds,
            "build_csr_index": index_seconds,
            "sample_one_batch": sample_seconds,
        },
        "memory_bytes": {
            "canonical_numpy_arrays": int(
                sum(getattr(dataset, name).nbytes for name in dataset.__slots__)
            ),
            "sampler_csr_index": sampler.index_nbytes,
        },
        "sample": {
            "seed_node_count": int(subgraph.seed_nodes.size),
            "sampled_node_count": int(subgraph.nodes.size),
            "sampled_edge_count": int(subgraph.edge_index.shape[1]),
            "hop_edge_counts": list(subgraph.hop_edge_counts),
        },
        "versions": {"numpy": np.__version__},
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
