#!/usr/bin/env python3
"""Generate raw mHC Expand fixtures and metadata from the declared cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tests.common.cases import (  # noqa: E402
    Case,
    accepted_cases,
    case_by_name,
)
from tests.common.reference import make_io
from tests.common.tensor_io import write_raw


DEFAULT_MAX_ELEMENTS = 20_000_000


def _tensor_metadata(tensor: np.ndarray, filename: str, logical_dtype: str) -> dict[str, Any]:
    return {
        "file": filename,
        "shape": list(tensor.shape),
        "dtype": logical_dtype,
        "numel": tensor.size,
        "bytes": tensor.size * tensor.dtype.itemsize,
        "layout": "contiguous_nd",
        "byte_order": "little_endian",
    }


def _comparison(dtype: str, backward: bool) -> dict[str, Any]:
    if not backward:
        return {"exact": True, "reason": "forward is a bitwise copy"}
    if dtype == "float16":
        return {"exact": False, "rtol": 1.0e-3, "atol": 1.0e-3}
    return {"exact": False, "rtol": 8.0e-3, "atol": 1.0e-2}


def _case_metadata(case: Case, tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    calls = [
        {
            "input": "input.bin",
            "output": "golden.bin",
            "attrs": {
                "mhc_mult": case.mhc_mult,
                "backward": case.backward,
            },
            "comparison": _comparison(case.dtype, backward=case.backward),
        }
    ]

    return {
        "name": case.name,
        "suite": case.suite,
        "mode": case.mode,
        "dtype": case.dtype,
        "S": case.s,
        "D": case.d,
        "mhc_mult": case.mhc_mult,
        "pattern": case.pattern,
        "seed": case.seed,
        "tags": list(case.tags),
        "calls": calls,
        "tensors": {
            role: _tensor_metadata(tensor, f"{role}.bin", case.dtype)
            for role, tensor in tensors.items()
        },
        "runner_checks": {
            "explicit_backward_attr": True,
            "prefill_output_with_nan": True,
            "guard_regions": {
                "bytes_before": 64,
                "bytes_after": 64,
                "canary_byte": "0xA5",
            },
            "repeat_count": 3 if case.backward else 1,
        },
        "abi_note": (
            "Required by the problem statement; the candidate must expose the "
            "bfloat16 ACLNN schema before this case can execute."
            if case.dtype == "bfloat16"
            else None
        ),
    }


def materialize_case(case: Case, output_root: Path) -> Path:
    case_dir = output_root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    tensors = make_io(case)
    for role, tensor in tensors.items():
        write_raw(case_dir / f"{role}.bin", tensor, case.dtype)
    metadata = _case_metadata(case, tensors)
    (case_dir / "case.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return case_dir


def _print_cases(suite: str) -> None:
    print(f"{suite} cases:")
    for case in accepted_cases(suite):
        shape = "x".join(str(dim) for dim in case.input_shape)
        print(
            f"  {case.name:46} mode={case.mode:9} dtype={case.dtype:8} "
            f"input={shape:18} m={case.mhc_mult} max_numel={case.largest_tensor_numel}"
        )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", action="append", help="case name; may be repeated")
    selection.add_argument("--all", action="store_true", help="generate the selected suite")
    selection.add_argument("--list", action="store_true", help="list the matrix without writing data")
    parser.add_argument(
        "--suite",
        choices=("coverage", "correctness", "full"),
        default="full",
        help="case category or their complete union",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/mhc_expand_cases"),
        help="ST fixture output root (default: /tmp/mhc_expand_cases)",
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=DEFAULT_MAX_ELEMENTS,
        help=(
            "skip cases whose largest tensor exceeds this many elements when generating a suite; "
            "use 0 to remove the cap"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list or (not args.case and not args.all):
        _print_cases(args.suite)
        return 0

    names = args.case if args.case else [case.name for case in accepted_cases(args.suite)]

    generated = 0
    skipped = 0
    for name in names:
        try:
            case = case_by_name(name)
        except KeyError:
            print(f"unknown case: {name}", file=sys.stderr)
            return 2

        if args.all and args.max_elements and case.largest_tensor_numel > args.max_elements:
            print(
                f"skipped {name}: max_numel={case.largest_tensor_numel} "
                f"> limit={args.max_elements}"
            )
            skipped += 1
            continue
        path = materialize_case(case, args.output_dir)
        print(f"generated {name} -> {path}")
        generated += 1

    print(f"summary: generated={generated}, skipped={skipped}, output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
