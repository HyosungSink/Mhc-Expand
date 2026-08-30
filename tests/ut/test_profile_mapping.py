"""Independent mapping from msprof CSV rows to mHC Expand samples."""

import csv

import pytest

from tests.st.scripts.profile_mock_workloads import duration_samples, kernel_rows


def test_profiler_mapping_selects_only_operator_rows(tmp_path) -> None:
    path = tmp_path / "op_summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Op Name", "Task Duration(us)"])
        writer.writeheader()
        writer.writerow({"Op Name": "MhcExpand", "Task Duration(us)": "3.5"})
        writer.writerow({"Op Name": "Other", "Task Duration(us)": "99"})
    rows = kernel_rows([path])
    assert len(rows) == 1
    assert duration_samples(rows) == [3.5]


def test_profiler_mapping_rejects_missing_duration(tmp_path) -> None:
    path = tmp_path / "op_summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Op Name", "Duration"])
        writer.writeheader()
        writer.writerow({"Op Name": "MhcExpand", "Duration": "3.5"})
    with pytest.raises(ValueError, match="Task Duration"):
        duration_samples(kernel_rows([path]))


def test_mock_default_runtime_uses_independent_calibrated_processes(tmp_path, monkeypatch):
    from tests.st.scripts import profile_mock_workloads as profiler
    from tests.common.mock_cases import MOCK_RUNTIME

    assert {k: MOCK_RUNTIME[k] for k in ("allocation_policy", "guard_bytes", "profiler_backend", "processes")} == {
        "allocation_policy": "normal-only", "guard_bytes": 512, "profiler_backend": "operator", "processes": 3}
    build, fixture = tmp_path / "build", tmp_path / "fixture"
    build.mkdir(); fixture.mkdir()
    (build / "libcust_opapi.so").write_bytes(b"library")
    calls = []
    times = iter([10., 50., 20.])
    def process(*args, **kwargs):
        calls.append(kwargs)
        sample = next(times)
        return {"status": "Pass", "passed": True, "median_us": sample, "kernel_samples_us": [sample]}
    monkeypatch.setattr(profiler, "_run_process", process)
    result = profiler.run_case(build, fixture, tmp_path / "output", tmp_path / "runner")
    assert len(calls) == len(result["observations"]) == 3
    assert all(c["allocation_policy"] == "normal-only" and c["guard_bytes"] == 512 and c["profiler_backend"] == "operator" for c in calls)
    assert result["median_us"] == 20.
    assert result["kernel_samples_us"] == [10., 50., 20.]


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_profiler_mapping_rejects_invalid_duration(value):
    with pytest.raises(ValueError):
        duration_samples([{"Task Duration(us)": value}])
