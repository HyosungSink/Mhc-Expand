from __future__ import annotations

import pytest
import numpy as np

from tests.common.cases import (
    CASES,
    CORRECTNESS_CASES,
    COVERAGE_CASES,
    accepted_cases,
)
from tests.common.reference import backward_golden, make_input
from tests.common.mock_cases import MOCK_CASES
from tests.common.tensor_io import decode_logical, encode_logical
from tests.st.scripts.audit_coverage import audit
from tests.st.scripts.run_tests import select_cases


def test_declared_suites_are_disjoint_and_complete() -> None:
    coverage_names = {case.name for case in COVERAGE_CASES}
    correctness_names = {case.name for case in CORRECTNESS_CASES}
    assert len(COVERAGE_CASES) == 16
    assert len(CORRECTNESS_CASES) == 10
    assert not coverage_names & correctness_names
    assert coverage_names | correctness_names == {case.name for case in CASES}
    assert accepted_cases("full") == CASES
    assert all(case.suite == "coverage" for case in COVERAGE_CASES)
    assert all(case.suite == "correctness" for case in CORRECTNESS_CASES)
    assert len(MOCK_CASES) == 8
    assert {case["test_point"] for case in MOCK_CASES} == set(range(1, 9))


def test_documented_coverage_audit_passes() -> None:
    result = audit()
    assert result["status"] == "PASS", "\n".join(result["issues"])


def test_precision_cases_belong_to_correctness() -> None:
    precision = select_cases("correctness", (), ("precision",))
    assert len(precision) == 6
    assert all(case.mode == "backward" for case in precision)
    assert {(case.dtype, case.pattern) for case in precision} == {
        (dtype, pattern)
        for dtype in ("float16", "bfloat16")
        for pattern in ("precision_triplet", "large_values", "cancellation")
    }


def test_sensitivity_cases_belong_to_correctness() -> None:
    sensitive = select_cases("correctness", (), ("sensitivity",))
    assert len(sensitive) == 4
    assert all(case in CORRECTNESS_CASES for case in sensitive)
    assert {case.mode for case in sensitive} == {"forward", "backward"}
    with pytest.raises(ValueError, match="no matching"):
        select_cases("coverage", (), ("sensitivity",))

    for case in sensitive:
        tensor = make_input(case)
        if case.mode == "forward":
            assert np.unique(tensor).size > 100
            assert case.d > 16384
        else:
            assert case.d > 8192
            expected = encode_logical(np.ones((case.s, case.d), dtype=np.float32), case.dtype)
            np.testing.assert_array_equal(
                backward_golden(tensor, case.mhc_mult, case.dtype), expected
            )
            low_precision = np.zeros((case.s, case.d), dtype=np.float32)
            decoded = decode_logical(tensor, case.dtype)
            for lane in range(case.mhc_mult):
                low_precision = decode_logical(
                    encode_logical(low_precision + decoded[:, lane, :], case.dtype), case.dtype
                )
            assert np.count_nonzero(low_precision) == 0
