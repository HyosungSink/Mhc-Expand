"""Fixed Mock catalog, deterministic fixtures, and output comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tensor_io import decode_logical, encode_logical


MOCK_CONFIG = Path(__file__).resolve().parents[1] / "st/cases/mock.json"


def load_mock_suite(path: Path = MOCK_CONFIG) -> tuple[dict, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("category") != "mock":
        raise ValueError(f"invalid Mock category: {path}")
    cases = tuple(document.get("cases", ()))
    points = [case.get("test_point") for case in cases]
    if len(cases) != 8 or set(points) != set(range(1, 9)):
        raise ValueError("Mock suite must contain test points 1 through 8")
    for case in cases:
        if case.get("dtype") != "float16":
            raise ValueError("measured Mock cases use float16")
        if any(int(case.get(field, 0)) <= 0 for field in ("s", "d", "m")):
            raise ValueError("Mock shapes and multipliers must be positive")
        if case.get("pattern") not in {"uniform", "normal", "integer", "zeros", "ones"}:
            raise ValueError("unsupported Mock input pattern")
    return cases


MOCK_CASES = load_mock_suite()


def load_mock_runtime(path: Path = MOCK_CONFIG) -> dict:
    runtime = json.loads(path.read_text())["runtime"]
    if runtime["allocation_policy"] not in ("normal-only", "huge-first") or runtime["profiler_backend"] not in ("operator", "task"):
        raise ValueError("unsupported Mock runtime")
    if any(type(runtime[k]) is not int or runtime[k] < 1 for k in ("repeat", "processes", "guard_bytes")):
        raise ValueError("Mock repeat, processes and guards must be positive integers")
    if runtime["warmup"] < 0 or runtime["vector_cores"] < 0 or runtime["resource_mode"] not in ("device", "stream"):
        raise ValueError("invalid Mock warmup or device resources")
    return runtime


MOCK_RUNTIME = load_mock_runtime()


def _mock_values(case: dict) -> np.ndarray:
    shape = (case["s"], case["m"], case["d"]) if case["backward"] else (
        case["s"], case["d"]
    )
    rng = np.random.default_rng(case["seed"])
    pattern = case["pattern"]
    if pattern == "uniform":
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if pattern == "normal":
        return rng.standard_normal(size=shape, dtype=np.float32)
    if pattern == "integer":
        return rng.integers(-64, 65, size=shape).astype(np.float32)
    if pattern == "zeros":
        return np.zeros(shape, dtype=np.float32)
    if pattern == "ones":
        return np.ones(shape, dtype=np.float32)
    raise ValueError(f"unsupported Mock pattern: {pattern}")


def materialize_mock_fixture(case: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    encoded = encode_logical(_mock_values(case), case["dtype"])
    encoded.tofile(directory / "input.bin")
    if case["backward"]:
        decoded = decode_logical(encoded, case["dtype"])
        accumulator = np.zeros((case["s"], case["d"]), dtype=np.float32)
        for lane in range(case["m"]):
            np.add(accumulator, decoded[:, lane, :], out=accumulator)
        golden = encode_logical(accumulator, case["dtype"])
    else:
        golden = np.repeat(encoded[:, np.newaxis, :], case["m"], axis=1)
    golden.tofile(directory / "golden.bin")
    (directory / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return directory


def compare_mock_output(
    actual_path: Path,
    fixture: Path,
    *,
    atol: float,
    rtol: float,
    mismatch_ratio: float,
) -> dict:
    case = json.loads((fixture / "case.json").read_text())
    actual = np.fromfile(actual_path, dtype="<u2")
    golden = np.fromfile(fixture / "golden.bin", dtype="<u2")
    if actual.size != golden.size:
        raise ValueError("output shape mismatch")
    actual_value = decode_logical(actual, case["dtype"])
    golden_value = decode_logical(golden, case["dtype"])
    error = np.abs(actual_value - golden_value)
    finite = np.isfinite(actual_value) & np.isfinite(golden_value)
    mismatches = int(np.count_nonzero(
        ~finite | (error > atol + rtol * np.abs(golden_value))
    ))
    ratio = mismatches / actual.size if actual.size else 0.0
    finite_error = error[np.isfinite(error)]
    return {
        "status": "Pass" if ratio <= mismatch_ratio else "Wrong Answer",
        "mismatches": mismatches,
        "elements": int(actual.size),
        "ratio_percent": ratio * 100.0,
        "bit_mismatches": int(np.count_nonzero(actual != golden)),
        "max_abs_error": float(finite_error.max(initial=0.0)),
        "atol": atol,
        "rtol": rtol,
        "mismatch_ratio": mismatch_ratio,
    }
