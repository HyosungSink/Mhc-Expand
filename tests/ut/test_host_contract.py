"""Exercise production Host Tiling and inference through CANN metadata contexts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from tests.common.cases import COVERAGE_CASES
from tests.common.mock_cases import MOCK_CASES
from tests.st.scripts.run_tests import SDK, runtime_environment


@pytest.fixture(scope="module")
def host_contract(tmp_path_factory):
    header = SDK / "include/base/context_builder/op_tiling_context_builder.h"
    if not header.is_file():
        pytest.skip("CANN context builders are required for Host contract tests")
    binary = tmp_path_factory.mktemp("host-contract") / "host_contract"
    source = Path(__file__).with_name("host_contract.cpp")
    command = [
        "c++", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DOP_TILING_LIB",
        "-I" + str(SDK / "include"), str(source), "-L" + str(SDK / "lib64"),
        "-Wl,-rpath," + str(SDK / "lib64"), "-Wl,--copy-dt-needed-entries",
        "-lnnopbase", "-lexe_graph", "-lregister", "-ltiling_api", "-lplatform",
        "-lrt2_registry", "-lunified_dlog", "-lgraph_base", "-lc_sec", "-lmetadef",
        "-ldl", "-pthread", "-o", str(binary),
    ]
    result = subprocess.run(
        command,
        env=runtime_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


def _run(host_contract: Path, specifications: list[tuple[str, str, bool, int, int, int, str]]) -> None:
    rows = [(*row[:2], int(row[2]), *row[3:]) for row in specifications]
    payload = "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
    result = subprocess.run(
        [str(host_contract), os.environ.get("MHC_TEST_SOC", "Ascend910B3")],
        input=payload,
        env=runtime_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert sum(line.startswith("PASS ") for line in result.stdout.splitlines()) == len(specifications)


def test_host_accepts_coverage_and_mock_contracts(host_contract: Path) -> None:
    cases = [
        (case.name, case.dtype, case.backward, case.s, case.d, case.mhc_mult, "accept")
        for case in COVERAGE_CASES
    ]
    cases.extend(
        (f"mock_{case['test_point']}", case["dtype"], case["backward"],
         case["s"], case["d"], case["m"], "accept")
        for case in MOCK_CASES
    )
    _run(host_contract, cases)


def test_host_accepts_flat_backward_and_rejects_invalid_contracts(host_contract: Path) -> None:
    cases = [
        ("flat_backward", "float16", True, 3, 17, 3, "flat"),
        ("bad_forward_rank", "float16", False, 3, 17, 2, "rank"),
        ("bad_dtype", "float16", False, 3, 17, 2, "dtype"),
        ("missing_input", "float16", False, 3, 17, 2, "missing"),
        ("zero_multiplier", "float16", False, 3, 17, 2, "zero_multiplier"),
        ("lane_mismatch", "float16", True, 3, 17, 3, "lane_mismatch"),
    ]
    _run(host_contract, cases)
