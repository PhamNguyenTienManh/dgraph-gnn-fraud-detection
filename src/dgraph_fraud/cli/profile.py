"""Validate and profile a DGraphFin NPZ file."""

import argparse
import hashlib
import json
from pathlib import Path

from dgraph_fraud.data.loader import load_dgraphfin
from dgraph_fraud.data.profiler import profile_dataset
from dgraph_fraud.data.validator import validate_dataset


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kiểm định và thống kê DGraphFin")
    parser.add_argument("--data", type=Path, default=Path("data/dgraphfin.npz"))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_dgraphfin(args.data)
    validation = validate_dataset(dataset)
    report = {
        "source": str(args.data),
        "sha256": _sha256(args.data),
        "validation": validation.as_dict(),
        "profile": profile_dataset(dataset),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if validation.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
