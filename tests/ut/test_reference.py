from __future__ import annotations

import numpy as np
import pytest

from tests.common.cases import CASES
from tests.common.reference import backward_golden, make_input, make_io
from tests.common.tensor_io import decode_logical, encode_logical


CPU_CASES = tuple(case for case in CASES if case.largest_tensor_numel <= 1_000_000)


@pytest.mark.parametrize("case", CPU_CASES, ids=lambda case: case.name)
def test_cpu_reference_semantics(case) -> None:
    tensors = make_io(case)
    value = tensors["input"]
    golden = tensors["golden"]
    expected_dtype = np.dtype(np.float16 if case.dtype == "float16" else np.uint16)
    assert value.dtype == golden.dtype == expected_dtype

    if case.mode == "forward":
        assert value.shape == (case.s, case.d)
        assert golden.shape == (case.s, case.mhc_mult, case.d)
        for lane in range(case.mhc_mult):
            np.testing.assert_array_equal(golden[:, lane, :], value)
    else:
        assert value.shape == (case.s, case.mhc_mult, case.d)
        assert golden.shape == (case.s, case.d)
        accumulator = np.zeros((case.s, case.d), dtype=np.float32)
        decoded = decode_logical(value, case.dtype)
        for lane in range(case.mhc_mult):
            accumulator += decoded[:, lane, :]
        np.testing.assert_array_equal(golden, encode_logical(accumulator, case.dtype))


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_precision_triplet_distinguishes_reduced_precision_order(dtype: str) -> None:
    case = next(case for case in CASES if case.name == f"backward_fp32_triplet_{dtype}")
    value = make_input(case)
    fp32_result = decode_logical(backward_golden(value, case.mhc_mult, dtype), dtype)
    reduced = np.zeros((case.s, case.d), dtype=np.float32)
    decoded = decode_logical(value, dtype)
    for lane in range(case.mhc_mult):
        reduced = decode_logical(encode_logical(reduced + decoded[:, lane, :], dtype), dtype)
    np.testing.assert_array_equal(fp32_result, np.ones_like(fp32_result))
    np.testing.assert_array_equal(reduced, np.zeros_like(reduced))
