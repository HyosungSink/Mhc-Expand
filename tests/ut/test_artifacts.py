"""Fixture generation, runner plans, dtype filtering, and Kernel cache checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.common.cases import case_by_name
from tests.common.reference import verify_case
from tests.st.scripts.gen_data import materialize_case
from tests.st.scripts.kernel_cache import CacheValidationError, KernelArtifactCache
from tests.st.scripts.run_tests import (
    _evaluate_calls,
    _plan,
    _single_dtype_config,
    resolve_jobs,
    select_cases,
)


def test_submission_snapshot_excludes_extra_header_and_detects_changes(tmp_path, monkeypatch):
    from tests.st.scripts import run_tests as pipeline

    source = tmp_path / "code"
    for name in pipeline.TEMPLATE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    extra = source / "op_kernel/extra.h"
    extra.write_text("header outside the submission")
    monkeypatch.setattr(pipeline, "CODE_DIR", source)
    snapshot, identity = pipeline.snapshot_submission(tmp_path / "build")
    assert not (snapshot / "op_kernel/extra.h").exists()
    assert len(pipeline.verify_submission_snapshot(tmp_path / "build", identity)) == 7
    (snapshot / "op_kernel/mhc_expand.cpp").write_text("changed snapshot")
    with pytest.raises(pipeline.StepFailure, match="snapshot"):
        pipeline.verify_submission_snapshot(tmp_path / "build", identity)


def test_fixture_cache_identity_includes_tensor_codec(tmp_path, monkeypatch):
    from tests.st.scripts import run_tests as pipeline

    for name in ("st/scripts/gen_data.py", "common/reference.py", "common/tensor_io.py", "common/cases.py"):
        path = tmp_path / "tests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("first version")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    case = case_by_name("backward_fp32_triplet_bfloat16")
    before = pipeline._fixture_key(case)
    (tmp_path / "tests/common/tensor_io.py").write_text("different rounding")
    assert pipeline._fixture_key(case) != before


def _completed_calls(tmp_path, *, wrong=False, later_re=False):
    run = tmp_path / "run"
    run.mkdir()
    fixture = materialize_case(case_by_name("forward_scalar_float16"), tmp_path / "fixtures")
    fixtures = [fixture]
    if later_re:
        fixtures.append(materialize_case(case_by_name("forward_scalar_bfloat16"), tmp_path / "fixtures"))
    _, calls = _plan(fixtures, run, False)
    library = tmp_path / "lib.so"
    library.write_bytes(b"library identity")
    actual = calls[0]["output"]
    data = b"\x00\x7e" * 2 if wrong else (fixture / "golden.bin").read_bytes()
    actual.write_bytes(data)
    Path(str(actual) + ".final.bin").write_bytes(data)
    native = [{"case": calls[0]["label"], "status": "executed", "guards_ok": True, "library": str(library)}]
    if later_re:
        native.append({"case": calls[1]["label"], "status": "Runtime Error", "diagnostic": "161002"})
    log = run / "runner.log"
    log.write_text("\n".join(json.dumps(row) for row in native) + "\n")
    return calls, log, library


def test_completed_wa_survives_later_runtime_error(tmp_path):
    calls, log, library = _completed_calls(tmp_path, wrong=True, later_re=True)
    result = _evaluate_calls(calls, log, library, 1, False)
    assert result["status"] == "Wrong Answer"
    assert result["calls"][0]["first_error_ratio"] == "100.000000%"
    assert result["calls"][1]["status"] == "INCONCLUSIVE"


def test_later_runtime_error_does_not_turn_partial_pass_into_pass(tmp_path):
    calls, log, library = _completed_calls(tmp_path, later_re=True)
    assert _evaluate_calls(calls, log, library, 1, False)["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("suffix", ["", ".final.bin"])
def test_completed_call_requires_both_outputs(tmp_path, suffix):
    calls, log, library = _completed_calls(tmp_path)
    assert _evaluate_calls(calls, log, library, 0, False)["status"] == "Pass"
    Path(str(calls[0]["output"]) + suffix).unlink()
    assert _evaluate_calls(calls, log, library, 0, False)["status"] == "INCONCLUSIVE"


def test_layered_case_selection_and_jobs() -> None:
    selected = select_cases("custom", ["forward_scalar_float16"], [])
    assert [case.name for case in selected] == ["forward_scalar_float16"]
    assert len(select_cases("coverage", [], [])) == 16
    assert len(select_cases("correctness", [], [])) == 10
    assert resolve_jobs(None, {}) == 16
    assert resolve_jobs(None, {"CMAKE_BUILD_PARALLEL_LEVEL": "24"}) == 24
    with pytest.raises(ValueError):
        resolve_jobs(0, {})


def test_dtype_filter_keeps_one_schema_column() -> None:
    source = "input0.dtype=float16,bfloat16\ninput0.format=ND,ND\nattr.list=x\n"
    assert _single_dtype_config(source, 1) == (
        "input0.dtype=bfloat16\ninput0.format=ND\nattr.list=x\n"
    )


@pytest.mark.parametrize(
    "case_name",
    ("forward_scalar_float16", "backward_fp32_triplet_bfloat16"),
)
def test_binary_fixture_generation(tmp_path: Path, case_name: str) -> None:
    case = case_by_name(case_name)
    case_dir = materialize_case(case, tmp_path)
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert metadata["name"] == case.name
    assert metadata["suite"] == case.suite
    assert metadata["calls"][0]["attrs"] == {
        "mhc_mult": case.mhc_mult,
        "backward": case.backward,
    }
    assert metadata["calls"][0]["comparison"]["exact"] == (not case.backward)
    for tensor in metadata["tensors"].values():
        assert (case_dir / tensor["file"]).stat().st_size == tensor["bytes"]


def test_multi_case_plan_contains_every_call(tmp_path: Path) -> None:
    fixtures = [
        materialize_case(case_by_name("forward_scalar_float16"), tmp_path / "fixtures"),
        materialize_case(case_by_name("backward_scalar_float16"), tmp_path / "fixtures"),
    ]
    run = tmp_path / "run"
    run.mkdir()
    plan, calls = _plan(fixtures, run, workspace_only=False)
    assert len(calls) == 2
    assert len(plan.read_text().splitlines()) == 2
    assert {row["label"] for row in calls} == {
        "forward_scalar_float16#0",
        "backward_scalar_float16#0",
    }


@pytest.mark.parametrize(
    "case_name",
    ("forward_scalar_float16", "backward_fp32_triplet_bfloat16"),
)
def test_verifier_accepts_matching_result(tmp_path: Path, case_name: str) -> None:
    directory = materialize_case(case_by_name(case_name), tmp_path)
    actual = directory / "actual.bin"
    actual.write_bytes((directory / "golden.bin").read_bytes())
    result = verify_case(directory, actual)
    assert result.passed and result.mismatch_count == 0


def test_verifier_rejects_corrupted_exact_result(tmp_path: Path) -> None:
    directory = materialize_case(case_by_name("forward_small_float16"), tmp_path)
    raw = bytearray((directory / "golden.bin").read_bytes())
    raw[-1] ^= 1
    actual = directory / "actual.bin"
    actual.write_bytes(raw)
    result = verify_case(directory, actual)
    assert not result.passed and result.message == "exact comparison failed"


def test_verifier_rejects_non_finite_backward_result(tmp_path: Path) -> None:
    directory = materialize_case(case_by_name("backward_fp32_triplet_bfloat16"), tmp_path)
    raw = bytearray((directory / "golden.bin").read_bytes())
    raw[0:2] = (0x7FC0).to_bytes(2, byteorder="little")
    actual = directory / "actual.bin"
    actual.write_bytes(raw)
    result = verify_case(directory, actual)
    assert not result.passed and result.mismatch_count == 1


def test_exact_comparison_counts_every_wrong_element(tmp_path: Path) -> None:
    directory = materialize_case(case_by_name("forward_small_float16"), tmp_path)
    raw = bytearray((directory / "golden.bin").read_bytes())
    for element in (0, 1, len(raw) // 2 - 1):
        raw[2 * element] ^= 1
    actual = directory / "actual.bin"
    actual.write_bytes(raw)
    result = verify_case(directory, actual)
    assert not result.passed and result.mismatch_count == 3


def test_kernel_cache_rejects_tampered_artifact(tmp_path: Path) -> None:
    binary = tmp_path / "binary"
    binary.mkdir()
    (binary / "kernel.o").write_bytes(b"object")
    (binary / "kernel.json").write_text("{}\n")
    cache = KernelArtifactCache(tmp_path / "cache")
    identity = {"operator": "MhcExpand", "dtype": "float16"}
    entry, _ = cache.publish(identity, binary, {"test": True})
    assert cache.lookup(identity) is not None
    (entry / "artifacts/kernel.o").write_bytes(b"changed")
    with pytest.raises(Exception):
        cache.lookup(identity)
