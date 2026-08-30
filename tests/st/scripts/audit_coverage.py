#!/usr/bin/env python3
"""Machine-executable audit for the declarative mHC Expand case matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.common.cases import CASES, CORRECTNESS_CASES, COVERAGE_CASES, DOCUMENTED_DTYPES


def audit() -> dict[str, object]:
    issues: list[str] = []

    def require(condition: bool, description: str) -> None:
        if not condition:
            issues.append(description)

    require(len({case.name for case in CASES}) == len(CASES), "case names are not unique")
    require(
        all(case.s > 0 and case.d > 0 and case.mhc_mult > 0 for case in CASES),
        "S, D, and mhc_mult must be positive",
    )
    require(
        all(case.dtype in DOCUMENTED_DTYPES and case.mode in ("forward", "backward") for case in CASES),
        "cases use an unsupported dtype or direction",
    )

    documented_shapes = ((64, 256, 2), (1024, 4096, 4), (8192, 7168, 8))
    expected_coverage = {
        (mode, dtype, s, d, multiplier)
        for mode in ("forward", "backward")
        for dtype in DOCUMENTED_DTYPES
        for s, d, multiplier in documented_shapes
    }
    expected_coverage.update(
        (mode, dtype, 1, 1, 2)
        for mode in ("forward", "backward")
        for dtype in DOCUMENTED_DTYPES
    )
    actual_coverage = {
        (case.mode, case.dtype, case.s, case.d, case.mhc_mult) for case in COVERAGE_CASES
    }
    require(
        len(COVERAGE_CASES) == 16 and actual_coverage == expected_coverage,
        "Coverage must contain the documented scales and combined scalar boundary",
    )

    precision = tuple(case for case in CORRECTNESS_CASES if "precision" in case.tags)
    expected_precision = {
        (dtype, pattern)
        for dtype in DOCUMENTED_DTYPES
        for pattern in ("precision_triplet", "large_values", "cancellation")
    }
    require(
        len(precision) == 6
        and {(case.dtype, case.pattern) for case in precision} == expected_precision
        and all(case.mode == "backward" for case in precision),
        "Correctness must contain six backward precision cases",
    )
    sensitivity = tuple(case for case in CORRECTNESS_CASES if "sensitivity" in case.tags)
    require(
        len(sensitivity) == 4
        and {(case.dtype, case.mode) for case in sensitivity}
        == {(dtype, mode) for dtype in DOCUMENTED_DTYPES for mode in ("forward", "backward")},
        "Correctness must contain four dtype/direction sensitivity cases",
    )
    require(
        len(CORRECTNESS_CASES) == 10
        and set(CORRECTNESS_CASES) == set(precision + sensitivity),
        "Correctness must contain only precision and sensitivity cases",
    )
    return {
        "status": "PASS" if not issues else "FAIL",
        "summary": {
            "total_cases": len(CASES),
            "coverage_cases": len(COVERAGE_CASES),
            "correctness_cases": len(CORRECTNESS_CASES),
            "forward_cases": sum(case.mode == "forward" for case in CASES),
            "backward_cases": sum(case.mode == "backward" for case in CASES),
            "audit_checks": 7,
            "audit_failures": len(issues),
        },
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = audit()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
