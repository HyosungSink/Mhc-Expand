#!/usr/bin/env python3
"""Run the fixed Mock workloads and optionally collect msprof task samples."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

from tests.common.mock_cases import (
    MOCK_CASES,
    MOCK_RUNTIME,
    compare_mock_output,
    materialize_mock_fixture,
)
from tests.st.scripts.kernel_cache import sha256_file
from tests.st.scripts.run_tests import SDK, build_runner, runtime_environment, verify_final_build


def select_cases(points: list[int]) -> tuple[dict, ...]:
    known = {case["test_point"] for case in MOCK_CASES}
    if set(points) - known:
        raise ValueError("unknown Mock test point")
    selected = tuple(case for case in MOCK_CASES if not points or case["test_point"] in points)
    if not selected:
        raise ValueError("Mock selection is empty")
    return selected


def kernel_rows(paths: list[Path], operator: str = "mhc") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if operator.lower() in " ".join(str(value) for value in row.values()).lower():
                    rows.append(row)
    return rows


def duration_samples(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    key = next((name for name in rows[0] if "Task Duration" in name), None)
    if key is None or any(key not in row for row in rows):
        raise ValueError("Profiler rows have no consistent Task Duration column")
    samples = [float(row[key]) for row in rows]
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Profiler duration must be finite and positive")
    return samples


def _run_process(
    build_dir: Path,
    fixture: Path,
    output: Path,
    runner: Path,
    *,
    device: int,
    warmup: int,
    repeat: int,
    profile: bool,
    profiler_backend: str,
    timeout: int,
    allocation_policy: str,
    guard_bytes: int,
    vector_cores: int,
    resource_mode: str,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    case = json.loads((fixture / "case.json").read_text())
    library = build_dir / "libcust_opapi.so"
    command = [
        str(runner),
        "--library", str(library),
        "--input", str(fixture / "input.bin"),
        "--output", str(output / "actual.bin"),
        "--s", str(case["s"]),
        "--d", str(case["d"]),
        "--m", str(case["m"]),
        "--device", str(device),
        "--guard-bytes", str(guard_bytes),
        "--poison-bits", "0",
        "--allocation-policy", allocation_policy,
        "--resource-mode", resource_mode,
    ]
    if vector_cores:
        command.extend(("--vector-cores", str(vector_cores)))
    native_warmup = 0 if profile and profiler_backend == "operator" else warmup
    native_repeat = 1 if profile and profiler_backend == "operator" else repeat
    command.extend(("--warmup", str(native_warmup), "--iterations", str(native_repeat)))
    if case["backward"]:
        command.append("--backward")
    if case["dtype"] == "bfloat16":
        command.append("--bf16")

    if profile and profiler_backend == "task":
        launcher = [
            "msprof",
            "--application=" + shlex.join(command),
            "--output=" + str(output / "profile"),
            "--task-time=on",
            "--ai-core=off",
            "--aicpu=off",
            "--ascendcl=off",
            "--runtime-api=off",
        ]
    elif profile:
        launcher = [
            "msprof", "op",
            "--application=" + shlex.join(command),
            "--output=" + str(output / "profile"),
            "--aic-metrics=BasicInfo",
            "--launch-count=1",
            "--warm-up=" + str(warmup),
        ]
    else:
        launcher = command

    environment = runtime_environment(build_dir)
    built_opp = build_dir / "tmp/vendors/custom"
    if built_opp.is_dir():
        environment["ASCEND_CUSTOM_OPP_PATH"] = str(built_opp)
    environment["ASCEND_HOME_PATH"] = str(SDK)
    (output / "command.json").write_text(json.dumps({
        "runner": command,
        "launcher": launcher,
    }, indent=2) + "\n")

    lock_path = Path("/tmp/cannjudge/mhcexpand/device.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock, (output / "runner.log").open("w") as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            process = subprocess.run(
                launcher, cwd=build_dir, env=environment, stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            returncode = None

    runner_result = None
    for line in (output / "runner.log").read_text(errors="replace").splitlines():
        if line.startswith("{"):
            try:
                runner_result = json.loads(line)
            except json.JSONDecodeError:
                continue

    result = {
        "returncode": returncode,
        "build_dir": str(build_dir.resolve()),
        "library_sha256": sha256_file(library),
        "fixture": str(fixture.resolve()),
        "warmup": warmup,
        "repeat": repeat,
        "profile": profile,
        "profiler_backend": profiler_backend,
        "runner_result": runner_result,
        "allocation_policy": allocation_policy,
        "guard_bytes": guard_bytes,
        "vector_cores": vector_cores,
        "resource_mode": resource_mode,
    }
    for name, key in (("actual.bin", "first"), ("actual.bin.final.bin", "final")):
        path = output / name
        if path.is_file():
            result[key] = compare_mock_output(
                path,
                fixture,
                atol=float(case["atol"]),
                rtol=float(case["rtol"]),
                mismatch_ratio=float(case["tol"]),
            )

    pattern = "OpBasicInfo.csv" if profiler_backend == "operator" else "op_summary*.csv"
    summaries = list((output / "profile").rglob(pattern)) if profile else []
    rows = kernel_rows(summaries)
    samples = duration_samples(rows)
    result["profile_files"] = [str(path) for path in summaries]
    result["kernel_samples_us"] = samples
    if profile:
        expected = 1 if profiler_backend == "operator" else warmup + repeat
        if len(samples) != expected:
            result["timing_error"] = f"expected {expected} kernel samples, found {len(samples)}"
        else:
            measured = samples if profiler_backend == "operator" else samples[warmup:]
            result["median_us"] = statistics.median(measured)
            result["mean_us"] = statistics.mean(measured)

    numerical_valid = (
        returncode == 0
        and bool(runner_result)
        and runner_result.get("status") == "executed"
        and runner_result.get("guards_ok") is True
        and Path(runner_result.get("library", "")).resolve() == library.resolve()
        and "first" in result and "final" in result
    )
    result["status"] = "INCONCLUSIVE"
    if numerical_valid:
        result["status"] = "Pass" if result["first"]["status"] == result["final"]["status"] == "Pass" else "Wrong Answer"
    result["passed"] = result["status"] == "Pass" and "timing_error" not in result
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_case(build_dir: Path, fixture: Path, output: Path, runner: Path, *,
             device: int = 0, warmup: int = MOCK_RUNTIME["warmup"],
             repeat: int = MOCK_RUNTIME["repeat"], profile: bool = True,
             profiler_backend: str = MOCK_RUNTIME["profiler_backend"], timeout: int = 240,
             processes: int = MOCK_RUNTIME["processes"],
             allocation_policy: str = MOCK_RUNTIME["allocation_policy"],
             guard_bytes: int = MOCK_RUNTIME["guard_bytes"],
             vector_cores: int = MOCK_RUNTIME["vector_cores"],
             resource_mode: str = MOCK_RUNTIME["resource_mode"]) -> dict:
    if warmup < 0 or repeat < 1 or processes < 1 or guard_bytes < 1:
        raise ValueError("invalid Mock execution counts or guards")
    output.mkdir(parents=True, exist_ok=False)
    library = build_dir / "libcust_opapi.so"
    library_hash = sha256_file(library)
    fixture_hashes = {p.name: sha256_file(p) for p in fixture.iterdir() if p.is_file()}
    observations = []
    for index in range(processes):
        result = _run_process(build_dir, fixture, output / f"process-{index}", runner,
            device=device, warmup=warmup, repeat=repeat, profile=profile,
            profiler_backend=profiler_backend, timeout=timeout, allocation_policy=allocation_policy,
            guard_bytes=guard_bytes, vector_cores=vector_cores, resource_mode=resource_mode)
        observations.append(result)
    if library_hash != sha256_file(library) or fixture_hashes != {p.name: sha256_file(p) for p in fixture.iterdir() if p.is_file()}:
        raise ValueError("Mock library or fixture changed during execution")
    status = "Wrong Answer" if any(o["status"] == "Wrong Answer" for o in observations) else (
        "Pass" if all(o["passed"] for o in observations) else "INCONCLUSIVE")
    result = {"status": status, "passed": status == "Pass", "observations": observations,
              "processes": processes, "fixture_sha256": fixture_hashes, "library_sha256": library_hash,
              "runtime": {"warmup": warmup, "repeat": repeat, "profiler_backend": profiler_backend,
                          "allocation_policy": allocation_policy, "guard_bytes": guard_bytes,
                          "vector_cores": vector_cores, "resource_mode": resource_mode},
              "kernel_samples_us": [v for o in observations for v in o["kernel_samples_us"]]}
    # Failed observations remain visible and never become a successful timing.
    medians = [o["median_us"] for o in observations if o["passed"] and "median_us" in o]
    result["pass_process_medians_us"] = medians
    if profile and status == "Pass" and len(medians) == processes:
        result.update(median_us=statistics.median(medians), mean_us=statistics.mean(medians))
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=MOCK_RUNTIME["warmup"])
    parser.add_argument("--repeat", type=int, default=MOCK_RUNTIME["repeat"])
    parser.add_argument("--processes", type=int, default=MOCK_RUNTIME["processes"])
    parser.add_argument("--allocation-policy", choices=("normal-only", "huge-first"), default=MOCK_RUNTIME["allocation_policy"])
    parser.add_argument("--guard-bytes", type=int, default=MOCK_RUNTIME["guard_bytes"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profiler-backend", choices=("task", "operator"), default=MOCK_RUNTIME["profiler_backend"])
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)
    if args.output.exists() or args.warmup < 0 or args.repeat < 1 or args.processes < 1 or args.guard_bytes < 1:
        parser.error("output must be absent; warmup must be nonnegative and repeat positive")

    verify_final_build(args.build_dir)
    runner, runner_manifest, _ = build_runner(args.build_dir, args.jobs)
    args.output.mkdir(parents=True)
    results = []
    for case in select_cases(args.case):
        point = case["test_point"]
        fixture = materialize_mock_fixture(case, args.output / "fixtures" / str(point))
        result = run_case(
            args.build_dir,
            fixture,
            args.output / "runs" / str(point),
            runner,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            profile=args.profile,
            profiler_backend=args.profiler_backend,
            timeout=args.timeout,
            processes=args.processes,
            allocation_policy=args.allocation_policy,
            guard_bytes=args.guard_bytes,
        )
        results.append({"test_point": point, **result})
        print(f"mock_case_{point}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)

    summary = {
        "status": "PASS" if all(item["passed"] for item in results) else "FAIL",
        "runner": runner_manifest,
        "cases": results,
    }
    verify_final_build(args.build_dir)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
