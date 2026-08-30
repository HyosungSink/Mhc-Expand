"""Deterministic inputs, CPU formulas, and output verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from .cases import Case
from .tensor_io import decode_logical, encode_logical, storage_dtype


def make_input(case: Case) -> np.ndarray:
    shape = case.input_shape
    if case.pattern == "random":
        rng = np.random.default_rng(case.seed)
        values = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    elif case.pattern == "ramp":
        values = np.arange(prod(shape), dtype=np.int64).reshape(shape)
        values = ((values % 257) - 128).astype(np.float32) / 32.0
    elif case.pattern == "precision_triplet":
        if case.mode != "backward" or case.mhc_mult != 3:
            raise ValueError("precision_triplet requires backward with mhc_mult=3")
        scale = 2048.0 if case.dtype == "float16" else 256.0
        values = np.empty(shape, dtype=np.float32)
        values[:, 0, :] = scale
        values[:, 1, :] = 1.0
        values[:, 2, :] = -scale
    elif case.pattern == "large_values":
        if case.mode != "backward":
            raise ValueError("large_values is only defined for backward inputs")
        values = np.full(shape, 7000.0, dtype=np.float32)
        values[:, -1, :] = 6500.0
    elif case.pattern == "cancellation":
        if case.mode != "backward" or case.mhc_mult != 8:
            raise ValueError("cancellation requires backward with mhc_mult=8")
        lanes = np.asarray(
            [4096.0, -4096.0, 8.0, -8.0, 2.0, -1.0, 0.5, -0.5],
            dtype=np.float32,
        )
        values = np.broadcast_to(lanes.reshape(1, 8, 1), shape).copy()
    else:
        raise ValueError(f"unknown input pattern: {case.pattern}")
    return encode_logical(values, case.dtype)


def forward_golden(value: np.ndarray, mhc_mult: int) -> np.ndarray:
    if value.ndim != 2 or mhc_mult <= 0:
        raise ValueError("forward golden expects rank-2 input and positive mhc_mult")
    return np.repeat(value[:, np.newaxis, :], mhc_mult, axis=1)


def backward_golden(value: np.ndarray, mhc_mult: int, logical_dtype: str) -> np.ndarray:
    if value.ndim != 3 or mhc_mult <= 0 or value.shape[1] != mhc_mult:
        raise ValueError("backward golden expects [S, mhc_mult, D]")
    decoded = decode_logical(value, logical_dtype)
    accumulator = np.zeros((value.shape[0], value.shape[2]), dtype=np.float32)
    for lane in range(mhc_mult):
        np.add(accumulator, decoded[:, lane, :], out=accumulator)
    return encode_logical(accumulator, logical_dtype)


def make_io(case: Case) -> dict[str, np.ndarray]:
    value = make_input(case)
    if case.mode == "forward":
        golden = forward_golden(value, case.mhc_mult)
    elif case.mode == "backward":
        golden = backward_golden(value, case.mhc_mult, case.dtype)
    else:
        raise ValueError(f"unknown case mode: {case.mode}")
    return {"input": value, "golden": golden}


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    message: str
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mismatch_count: int = 0


def verify_result(
    actual_path: Path,
    golden_path: Path,
    dtype: str,
    comparison: dict[str, Any],
    expected_bytes: int,
) -> VerificationResult:
    if not actual_path.is_file():
        return VerificationResult(False, f"actual output does not exist: {actual_path}")
    if not golden_path.is_file():
        return VerificationResult(False, f"golden output does not exist: {golden_path}")
    if actual_path.stat().st_size != expected_bytes:
        return VerificationResult(
            False,
            f"actual byte size mismatch: expected={expected_bytes}, actual={actual_path.stat().st_size}",
        )
    if golden_path.stat().st_size != expected_bytes:
        return VerificationResult(
            False,
            f"golden byte size mismatch: expected={expected_bytes}, actual={golden_path.stat().st_size}",
        )

    actual_storage = np.fromfile(actual_path, dtype=storage_dtype(dtype))
    golden_storage = np.fromfile(golden_path, dtype=storage_dtype(dtype))
    if comparison.get("exact", False):
        mismatch_count = int(np.count_nonzero(
            actual_storage.view(np.uint16) != golden_storage.view(np.uint16)
        ))
        return VerificationResult(
            mismatch_count == 0,
            "exact comparison passed" if mismatch_count == 0 else "exact comparison failed",
            mismatch_count=mismatch_count,
        )

    actual = decode_logical(actual_storage, dtype)
    golden = decode_logical(golden_storage, dtype)
    finite = np.isfinite(actual) & np.isfinite(golden)
    absolute_error = np.abs(actual - golden)
    tolerance = float(comparison["atol"]) + float(comparison["rtol"]) * np.abs(golden)
    mismatch_count = int(np.count_nonzero(~finite | (absolute_error > tolerance)))
    finite_error = absolute_error[np.isfinite(absolute_error)]
    relative_error = absolute_error / np.maximum(np.abs(golden), 1.0e-12)
    finite_relative = relative_error[np.isfinite(relative_error)]
    return VerificationResult(
        mismatch_count == 0,
        "tolerance comparison passed" if mismatch_count == 0 else (
            f"tolerance comparison failed: mismatches={mismatch_count}/{golden.size}"
        ),
        max_abs_error=float(finite_error.max(initial=0.0)),
        max_rel_error=float(finite_relative.max(initial=0.0)),
        mismatch_count=mismatch_count,
    )


def verify_case(case_dir: Path, actual_path: Path, call_index: int = 0) -> VerificationResult:
    metadata_path = case_dir / "case.json"
    if not metadata_path.is_file():
        return VerificationResult(False, f"case metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    calls = metadata["calls"]
    if call_index < 0 or call_index >= len(calls):
        return VerificationResult(False, f"call index {call_index} is outside [0, {len(calls)})")
    call = calls[call_index]
    tensor = next(
        item for item in metadata["tensors"].values() if item["file"] == call["output"]
    )
    return verify_result(
        actual_path,
        case_dir / call["output"],
        tensor["dtype"],
        call["comparison"],
        int(tensor["bytes"]),
    )
