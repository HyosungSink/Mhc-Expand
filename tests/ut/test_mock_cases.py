"""Fixed Mock catalog, fixture, and comparison checks."""

from __future__ import annotations

import numpy as np

from tests.common.mock_cases import (
    MOCK_CASES,
    compare_mock_output,
    load_mock_suite,
    materialize_mock_fixture,
)
from tests.common.tensor_io import decode_logical, float32_to_bfloat16_bits
from tests.st.scripts.profile_mock_workloads import select_cases
from tests.st.scripts.profile_mock_workloads import timing_settings


def test_mock_suite_contains_eight_executable_cases() -> None:
    assert load_mock_suite() == MOCK_CASES
    assert {case["test_point"] for case in MOCK_CASES} == set(range(1, 9))
    assert select_cases([1, 8]) == (MOCK_CASES[0], MOCK_CASES[7])


def test_mock_fixture_is_deterministic(tmp_path) -> None:
    first = materialize_mock_fixture(MOCK_CASES[0], tmp_path / "first")
    second = materialize_mock_fixture(MOCK_CASES[0], tmp_path / "second")
    for name in ("case.json", "input.bin", "golden.bin"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    raw = np.fromfile(first / "input.bin", dtype="<u2")
    assert raw[0] == int(MOCK_CASES[0]["observed_input_bits"]["0"], 16)
    assert raw[17] == int(MOCK_CASES[0]["observed_input_bits"]["17"], 16)
    assert compare_mock_output(
        first / "golden.bin", first, atol=0, rtol=0, mismatch_ratio=0
    )["status"] == "Pass"


def test_mock_comparator_rejects_corruption(tmp_path) -> None:
    fixture = materialize_mock_fixture(MOCK_CASES[0], tmp_path / "fixture")
    actual = tmp_path / "actual.bin"
    raw = bytearray((fixture / "golden.bin").read_bytes())
    raw[0:2] = (0x7C00).to_bytes(2, byteorder="little")
    actual.write_bytes(raw)
    result = compare_mock_output(
        actual,
        fixture,
        atol=MOCK_CASES[0]["atol"],
        rtol=MOCK_CASES[0]["rtol"],
        mismatch_ratio=0,
    )
    assert result["status"] == "Wrong Answer"
    assert result["mismatches"] >= 1


def test_mock_comparator_interprets_float16_values(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "case.json").write_text('{"dtype": "float16"}')
    np.array([-0.0], dtype="<f2").tofile(fixture / "golden.bin")
    actual = tmp_path / "actual.bin"
    np.array([0.0], dtype="<f2").tofile(actual)
    result = compare_mock_output(actual, fixture, atol=0, rtol=0,
                                 mismatch_ratio=0)
    assert result["status"] == "Pass"
    assert result["bit_mismatches"] == 1


def test_bfloat16_encoding_rounds_ties_to_even() -> None:
    values = np.array(
        [1 + 1 / 256, 1 + 3 / 256, np.nan, np.inf, -np.inf], dtype=np.float32
    )
    bits = float32_to_bfloat16_bits(values)
    assert bits.tolist() == [0x3F80, 0x3F82, 0x7FC0, 0x7F80, 0xFF80]
    assert np.allclose(
        decode_logical(bits[:2], "bfloat16"),
        np.array([1.0, 1.015625], dtype=np.float32),
    )


def test_mock_timing_selects_calibrated_large_case_protocol() -> None:
    small = timing_settings(MOCK_CASES[0], {})
    large = timing_settings(MOCK_CASES[2], {})
    assert (small["profiler_backend"], small["vector_cores"], small["repeat"]) == (
        "operator", 0, 1)
    assert (large["profiler_backend"], large["vector_cores"], large["repeat"]) == (
        "task", 24, 10)
    assert timing_settings(MOCK_CASES[6], {})["vector_cores"] == 32
