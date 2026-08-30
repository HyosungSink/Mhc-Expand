#!/usr/bin/env python3
"""Layered build and NPU validation pipeline for mHC Expand.

The incremental tree supports Host-only work and representative dtype kernels.
The final tree builds all templates from the submission file set.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence

from tests.common.cases import Case, accepted_cases, case_by_name
from tests.common.reference import verify_case
from tests.st.scripts.audit_coverage import audit
from tests.st.scripts.gen_data import DEFAULT_MAX_ELEMENTS, materialize_case
from tests.st.scripts.kernel_cache import (
    CacheValidationError,
    KernelArtifactCache,
    content_record,
    identity_files,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = ROOT / "code"
SDK = Path(os.environ.get("MHC_CANN_PATH", "/usr/local/Ascend/cann-8.5.0"))
ASC_DIR = SDK / "aarch64-linux/tikcpp/ascendc_kernel_cmake"
RUNNER_SOURCE = ROOT / "tests/st/runner/mhc_expand_runner.cpp"
DEFAULT_WORK_ROOT = Path(os.environ.get("MHC_WORK_ROOT", "/tmp/cannjudge/mhcexpand/layered"))
DEFAULT_INCREMENTAL_BUILD = DEFAULT_WORK_ROOT / "build/incremental"
DEFAULT_FINAL_BUILD = DEFAULT_WORK_ROOT / "build/final"
DEFAULT_KERNEL_CACHE = DEFAULT_WORK_ROOT / "kernel-cache"
DEFAULT_FIXTURE_CACHE = DEFAULT_WORK_ROOT / "fixtures"
DEFAULT_RESULTS = DEFAULT_WORK_ROOT / "results"
TEMPLATE_FILES = (
    "CMakeLists.txt",
    "op_host/CMakeLists.txt",
    "op_host/mhc_expand.cpp",
    "op_kernel/CMakeLists.txt",
    "op_kernel/mhc_expand.cpp",
    "op_kernel/mhc_expand_tiling.h",
    "op_kernel/tiling_key_mhc_expand.h",
)
DTYPE_REPRESENTATIVES = {
    "float16": "forward_scalar_float16",
    "bfloat16": "forward_scalar_bfloat16",
}
SMOKE_CASES = ("forward_scalar_float16", "backward_fp32_triplet_float16")
LAYERS = (
    "static", "ut", "generate", "host", "kernel", "workspace", "smoke",
    "correctness", "validate", "full-build", "full-test", "full", "pipeline",
)


class StepFailure(RuntimeError):
    def __init__(self, message: str, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode


def resolve_jobs(requested: int | None, environment: Mapping[str, str]) -> int:
    raw = requested
    if raw is None:
        raw = int(environment.get("CMAKE_BUILD_PARALLEL_LEVEL", "16"))
    if not 1 <= raw <= 64:
        raise ValueError("build parallelism must be between 1 and 64")
    return raw


def runtime_environment(*library_roots: Path, jobs: int | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment["ASCEND_HOME_PATH"] = str(SDK)
    environment["ASCEND_AICPU_PATH"] = str(SDK)
    environment["PATH"] = ":".join(
        (str(SDK / "bin"), str(SDK / "aarch64-linux/bin"), environment.get("PATH", ""))
    )
    libraries = [str(path) for path in library_roots]
    libraries.extend((str(SDK / "aarch64-linux/lib64"), environment.get("LD_LIBRARY_PATH", "")))
    environment["LD_LIBRARY_PATH"] = ":".join(value for value in libraries if value)
    environment["PYTHONPATH"] = ":".join(
        (str(ROOT), str(SDK / "python/site-packages"), environment.get("PYTHONPATH", ""))
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    if jobs is not None:
        environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    return environment


def _render(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def _run(command: Sequence[str], *, cwd: Path = ROOT, env: Mapping[str, str] | None = None) -> float:
    rendered = _render(command)
    print("+", rendered, flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, env=None if env is None else dict(env), check=False)
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise StepFailure(f"command failed ({completed.returncode}): {rendered}", completed.returncode)
    return elapsed


def _run_build(command: Sequence[str], build_dir: Path, label: str, jobs: int,
               *, cwd: Path = ROOT) -> float:
    log_dir = build_dir / "tests/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    print(f"+ {_render(command)} [log: {log_path}]", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=cwd, env=runtime_environment(jobs=jobs),
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode:
        lines = log_path.read_text(errors="replace").splitlines()
        pattern = re.compile(r"error:|fatal:|failed|undefined reference", re.IGNORECASE)
        hits = [index for index, line in enumerate(lines) if pattern.search(line)]
        if hits:
            start = max(0, hits[0] - 4)
            print("\n".join(lines[start:start + 20]), flush=True)
        else:
            print("\n".join(lines[-20:]), flush=True)
        raise StepFailure(f"{label} failed; full log: {log_path}", completed.returncode)
    print(f"PASS {label} elapsed={elapsed:.3f}s", flush=True)
    return elapsed


def _content_digest(paths: Iterable[Path], prefix: str) -> str:
    digest = hashlib.sha256(prefix.encode())
    for path in sorted(paths):
        label = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        digest.update(label.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_identity() -> str:
    return projected_source_identity(CODE_DIR)


def projected_source_identity(directory: Path) -> str:
    digest = hashlib.sha256(b"mhc-expand-seven-files-v1")
    for name in sorted(TEMPLATE_FILES):
        digest.update(("code/" + name).encode())
        digest.update((directory / name).read_bytes())
    return "source-sha256:" + digest.hexdigest()


def snapshot_submission(build_dir: Path) -> tuple[Path, str]:
    """Materialize exactly the source files transported by the official runner."""
    before = source_identity()
    destination = build_dir / "tests/submission/code"
    destination.mkdir(parents=True, exist_ok=False)
    for name in TEMPLATE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((CODE_DIR / name).read_bytes())
    if projected_source_identity(destination) != before or source_identity() != before:
        raise StepFailure("operator sources changed while snapshotting the submission")
    return destination, before


def verify_submission_snapshot(build_dir: Path, expected_source: str) -> list[dict]:
    directory = build_dir / "tests/submission/code"
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(TEMPLATE_FILES) or projected_source_identity(directory) != expected_source:
        raise StepFailure("final submission snapshot inventory or source identity differs")
    return [content_record(directory / name, "tests/submission/code/" + name)
            for name in sorted(TEMPLATE_FILES)]


def sdk_identity() -> dict:
    required = (
        (ASC_DIR / "ASCConfig.cmake").resolve(),
        (SDK / "compiler/version.info").resolve(),
    )
    if not all(path.is_file() for path in required):
        raise StepFailure(f"CANN SDK is incomplete: {SDK}")
    compiler = Path(shutil.which("bisheng", path=runtime_environment()["PATH"]) or SDK / "bin/bisheng").resolve()
    host_compiler = Path(shutil.which("c++") or "/usr/bin/c++").resolve()
    return {
        "root": str(SDK.resolve()),
        "asc_config": content_record(required[0], "ASCConfig.cmake"),
        "version": content_record(required[1], "compiler/version.info"),
        "bisheng": content_record(compiler, str(compiler)),
        "host_compiler": content_record(host_compiler, str(host_compiler)),
    }


def static_checks() -> dict[str, float]:
    started = time.monotonic()
    baseline = json.loads((ROOT / "tests/common/source_baseline.json").read_text())
    current = {"code/" + name for name in TEMPLATE_FILES}
    actual = {path.relative_to(ROOT).as_posix() for path in CODE_DIR.rglob("*") if path.is_file()}
    if actual != current or set(baseline) != current:
        raise StepFailure("code/ inventory differs from the seven official template files")
    for relative, expected in baseline.items():
        if Path(relative).name == "CMakeLists.txt" and sha256_file(ROOT / relative) != expected:
            raise StepFailure(f"CMakeLists.txt differs from the recorded template: {relative}")
    if (ROOT / ".git").is_dir():
        _run(("git", "diff", "--check"))
    report = audit()
    if report["status"] != "PASS":
        raise StepFailure(f"coverage audit failed: {report['issues']}")
    _run((sys.executable, "-m", "compileall", "-q", "tests"))
    return {"static": time.monotonic() - started}


def python_checks() -> float:
    return _run((sys.executable, "-m", "pytest", "-q", "tests/ut"), env=runtime_environment())


def configure(build_dir: Path, jobs: int, *, source_dir: Path | None = None) -> float:
    build_dir.mkdir(parents=True, exist_ok=True)
    return _run_build(
        (
            "cmake", "-S", str(source_dir or CODE_DIR), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release", f"-DASC_DIR={ASC_DIR}",
        ), build_dir, "configure", jobs,
    )


def _kernel_snapshot(build_dir: Path) -> tuple[tuple[str, int, int], ...]:
    root = build_dir / "op_kernel/ascendc_kernels/binary/ascend910b/mhc_expand"
    if not root.is_dir():
        return ()
    return tuple(sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns)
                        for path in root.iterdir() if path.is_file()))


def _host_only_opapi(build_dir: Path, jobs: int) -> tuple[Path, float]:
    generated = build_dir / "autogen/aclnn_mhc_expand.cpp"
    host_object = build_dir / "op_host/CMakeFiles/cust_optiling.dir/mhc_expand.cpp.o"
    missing = [str(path) for path in (generated, host_object) if not path.is_file()]
    if missing:
        raise StepFailure("Host-only ACLNN inputs are missing: " + ", ".join(missing))
    output = build_dir / "tests/host_only_opapi"
    output.mkdir(parents=True, exist_ok=True)
    generated_object = output / "aclnn_mhc_expand.cpp.o"
    library = output / "libcust_opapi.so"
    state = output / ".inputs.sha256"
    digest = _content_digest((generated, host_object), "mhc-host-only-opapi-v1")
    if library.is_file() and state.is_file() and state.read_text().strip() == digest:
        return output, 0.0
    elapsed = _run_build(
        (
            "c++", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DLOG_CPP",
            "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden",
            "-fstack-protector-strong", "-I", str(SDK / "include"),
            "-c", str(generated), "-o", str(generated_object),
        ), build_dir, "host-only-opapi-compile", jobs,
    )
    elapsed += _run_build(
        (
            "c++", "-fPIC", "-shared", "-Wl,-z,relro", "-Wl,-z,now",
            "-Wl,-z,noexecstack", "-Wl,-soname,libcust_opapi.so",
            "-o", str(library), str(host_object), str(generated_object),
            "-L", str(SDK / "lib64"), "-L", str(SDK / "aarch64-linux/lib64"),
            "-lascendcl", "-lnnopbase", "-lexe_graph", "-lregister", "-ltiling_api",
        ), build_dir, "host-only-opapi-link", jobs,
    )
    state.write_text(digest + "\n")
    return output, elapsed


def _host_state(build_dir: Path) -> dict:
    generated = (
        build_dir / "autogen/aic-ascend910b-ops-info.ini",
        build_dir / "autogen/aclnn_mhc_expand.h",
        build_dir / "autogen/aclnn_mhc_expand.cpp",
        build_dir / "op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json",
    )
    tiling = build_dir / "op_host/libcustom_ascendc_cust_optiling.so"
    opapi = build_dir / "tests/host_only_opapi/libcust_opapi.so"
    paths = [*(CODE_DIR / name for name in TEMPLATE_FILES if "op_kernel/mhc_expand.cpp" not in name), *generated]
    return {
        "source": source_identity(),
        "sdk": sdk_identity(),
        "inputs": [content_record(path, str(path)) for path in paths],
        "tiling": content_record(tiling, str(tiling)),
        "opapi": content_record(opapi, str(opapi)),
    }


def build_host(build_dir: Path, jobs: int) -> float:
    if not (build_dir / "CMakeCache.txt").is_file():
        configure(build_dir, jobs)
    before = _kernel_snapshot(build_dir)
    elapsed = _run_build(
        ("cmake", "--build", str(build_dir), "--target", "cust_optiling", "--parallel", str(jobs)),
        build_dir, "host-tiling", jobs,
    )
    elapsed += _run_build(
        ("cmake", "--build", str(build_dir), "--target", "custom_ascendc_cust_optiling",
         "--parallel", str(jobs)), build_dir, "host-tiling-link", jobs,
    )
    elapsed += _run_build(
        ("cmake", "--build", str(build_dir), "--target", "ascendc_kernels_ops_info_gen_ascend910b",
         "--parallel", str(jobs)), build_dir, "host-op-info", jobs,
    )
    _, opapi_elapsed = _host_only_opapi(build_dir, jobs)
    elapsed += opapi_elapsed
    after = _kernel_snapshot(build_dir)
    if before != after:
        raise StepFailure("Host-only build modified Kernel binaries")
    state = build_dir / "tests/host_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(_host_state(build_dir), indent=2) + "\n")
    print(f"PASS host-only kernel_files_unchanged={len(after)}", flush=True)
    return elapsed


def verify_host(build_dir: Path) -> None:
    state = build_dir / "tests/host_state.json"
    if not state.is_file() or json.loads(state.read_text()) != _host_state(build_dir):
        raise StepFailure("Host state source/toolchain/artifact identity differs")


def _dtype_column(text: str, dtype: str) -> int:
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in ("input0.dtype", "output0.dtype"):
                values[key] = value.split(",")
    if len(values) != 2:
        raise StepFailure("generated config has no input/output dtype columns")
    for index, value in enumerate(values["input0.dtype"]):
        if value == dtype and values["output0.dtype"][index] == dtype:
            return index
    raise StepFailure(f"generated config has no {dtype} column")


def _single_dtype_config(text: str, column: int) -> str:
    lines = []
    for line in text.splitlines():
        if "=" not in line:
            lines.append(line)
            continue
        key, value = line.split("=", 1)
        values = value.split(",")
        if len(values) > 1 and (key.endswith(".dtype") or key.endswith(".format")):
            value = values[column]
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def _representative_overlay(binary_root: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    source_kernel = binary_root / "ascend910b/mhc_expand"
    source_config = binary_root / "config/ascend910b"
    if not source_kernel.is_dir() or not source_config.is_dir():
        raise StepFailure("representative compiler did not produce a runtime package")
    target_kernel = destination / "op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand"
    target_config = destination / "op_impl/ai_core/tbe/kernel/config/ascend910b"
    target_kernel.mkdir(parents=True)
    target_config.mkdir(parents=True)
    for path in source_kernel.iterdir():
        if path.suffix in (".o", ".json"):
            shutil.copy2(path, target_kernel / path.name)
    for path in source_config.glob("*.json"):
        shutil.copy2(path, target_config / path.name)
    return destination


def _kernel_identity(build_dir: Path, dtype: str, quick_config: Path,
                     generated_json: Path, options: Sequence[Path]) -> dict:
    compile_script = ASC_DIR / "fwk_modules/util/ascendc_compile_kernel.py"
    tiling = build_dir / "op_host/libcustom_ascendc_cust_optiling.so"
    return {
        "schema_version": 1,
        "operator": "MhcExpand",
        "soc": "ascend910b",
        "dtype": dtype,
        "kernel": identity_files(((CODE_DIR / "op_kernel/mhc_expand.cpp", "code/op_kernel/mhc_expand.cpp"),)),
        "host_abi": {
            "sources": identity_files(
                (CODE_DIR / name, "code/" + name) for name in (
                    "op_host/mhc_expand.cpp", "op_kernel/mhc_expand_tiling.h",
                    "op_kernel/tiling_key_mhc_expand.h",
                )
            ),
            "tiling": content_record(tiling, "libcustom_ascendc_cust_optiling.so"),
        },
        "generated": identity_files((
            (build_dir / "autogen/aic-ascend910b-ops-info.ini", "autogen/aic-ascend910b-ops-info.ini"),
            (quick_config, "filtered/aic-ascend910b-ops-info.ini"),
            (generated_json, "op_info/aic-ascend910b-ops-info.json"),
            *((path, "options/" + path.name) for path in options),
        )),
        "compiler": {
            "sdk": sdk_identity(),
            "driver": content_record(compile_script, "ascendc_compile_kernel.py"),
            "python": sys.version,
        },
    }


def build_representative_kernel(build_dir: Path, jobs: int, case_name: str,
                                cache_dir: Path = DEFAULT_KERNEL_CACHE) -> tuple[float, Path]:
    started = time.monotonic()
    if not (build_dir / "tests/host_state.json").is_file():
        build_host(build_dir, jobs)
    verify_host(build_dir)
    case = case_by_name(case_name)
    generated_config = build_dir / "autogen/aic-ascend910b-ops-info.ini"
    generated_json = (
        build_dir / "op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend910b"
        / "aic-ascend910b-ops-info.json"
    )
    root = build_dir / "tests/representative" / case.dtype
    root.mkdir(parents=True, exist_ok=True)
    quick_config = root / "aic-ascend910b-ops-info.ini"
    quick_config.write_text(_single_dtype_config(
        generated_config.read_text(), _dtype_column(generated_config.read_text(), case.dtype)
    ))
    dtype_code = {"float16": 1, "bfloat16": 27}[case.dtype]
    keys = [dtype_code | (backward << 8) | (aligned << 9)
            for backward in (0, 1) for aligned in (0, 1)]
    opc_options = root / "custom_opc_options.ini"
    compile_options = root / "custom_compile_options.ini"
    opc_options.write_text("MhcExpand@@--tiling_key=" + ";".join(map(str, keys)) + "\n")
    compile_options.touch()
    identity = _kernel_identity(build_dir, case.dtype, quick_config, generated_json,
                                (opc_options, compile_options))
    cache = KernelArtifactCache(cache_dir)
    key = cache.key_for(identity)
    binary_root = root / "binary"
    overlay_root = root / "opp/vendors/custom"
    manifest_copy = root / "kernel-cache-manifest.json"
    completion = root / ".complete"
    with cache.lock(key):
        try:
            cached = cache.lookup(identity)
        except CacheValidationError as error:
            rejected = cache.quarantine(cache.entry_path(key), str(error))
            print(f"kernel cache rejected {rejected}: {error}", flush=True)
            cached = None
        if cached is None:
            print(f"kernel cache MISS dtype={case.dtype} key={key[:16]}", flush=True)
            for path in (binary_root, root / "dynamic", overlay_root):
                if path.is_dir():
                    shutil.rmtree(path)
            (root / "dynamic").mkdir()
            compile_script = ASC_DIR / "fwk_modules/util/ascendc_compile_kernel.py"
            command = (
                "python3", str(compile_script), "--op-type=MhcExpand",
                f"--src-file={CODE_DIR / 'op_kernel/mhc_expand.cpp'}",
                "--compute-unit=ascend910b", "--compile-options=", "--debug-config=",
                f"--config-ini={quick_config}",
                f"--tiling-lib={build_dir / 'op_host/libcustom_ascendc_cust_optiling.so'}",
                f"--output-path={binary_root}", f"--dynamic-dir={root / 'dynamic'}",
                "--enable-binary=TRUE", f"--json-file={generated_json}",
                "--target-name=ascendc_kernels", f"--auto-gen-path={build_dir / 'autogen'}",
                "--build-tool=make",
            )
            _run_build(command, build_dir, f"kernel-{case.dtype}", jobs, cwd=root)
            util = ASC_DIR / "fwk_modules/util"
            soc_root = binary_root / "ascend910b"
            _run(("python3", str(util / "insert_simplified_keys.py"), "-p", str(soc_root)), cwd=root)
            _run(("python3", str(util / "ascendc_ops_config.py"), "-p", str(soc_root),
                  "-s", "ascend910b"), cwd=root)
            config = binary_root / "config/ascend910b"
            config.mkdir(parents=True, exist_ok=True)
            for path in soc_root.glob("*.json"):
                shutil.move(str(path), config / path.name)
            cached = cache.publish(identity, binary_root, {
                "source": source_identity(), "representative_case": case_name,
            })
        entry, manifest = cached
        cache.materialize(entry, binary_root)
        overlay = _representative_overlay(binary_root, overlay_root)
        manifest_copy.write_text(json.dumps(manifest, indent=2) + "\n")
        completion.write_text(key + "\n")
    elapsed = time.monotonic() - started
    print(f"PASS representative dtype={case.dtype} key={key[:16]} elapsed={elapsed:.3f}s", flush=True)
    return elapsed, overlay


def build_runner(build_dir: Path, jobs: int) -> tuple[Path, dict, float]:
    identity = {
        "source": content_record(RUNNER_SOURCE, "tests/st/runner/mhc_expand_runner.cpp"),
        "sdk": sdk_identity(),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = build_dir / "tests/bin" / key[:16]
    binary = directory / "mhc_expand_runner"
    manifest_path = directory / "runner.json"
    if binary.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] == identity and manifest["binary"] == content_record(binary, str(binary)):
            return binary, manifest, 0.0
    directory.mkdir(parents=True, exist_ok=True)
    command = (
        "c++", "-std=c++17", "-O2", str(RUNNER_SOURCE), f"-I{SDK / 'include'}",
        f"-L{SDK / 'aarch64-linux/lib64'}", "-lascendcl", "-lnnopbase", "-ldl", "-o", str(binary),
    )
    elapsed = _run_build(command, build_dir, "runner", jobs)
    manifest = {"identity": identity, "command": command,
                "binary": content_record(binary, str(binary))}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return binary, manifest, elapsed


def _final_artifacts(build_dir: Path) -> list[dict]:
    roots = (
        build_dir / "libcust_opapi.so",
        build_dir / "op_host/libcustom_ascendc_cust_optiling.so",
    )
    paths = [path for path in roots if path.is_file()]
    paths.extend(path for base in (
        build_dir / "op_kernel/ascendc_kernels/binary",
        build_dir / "tmp/vendors/custom",
    ) if base.is_dir() for path in base.rglob("*")
                 if path.is_file() and path.suffix in (".o", ".json", ".so"))
    if len(paths) < 4 or len(paths) > 256:
        raise StepFailure("final package artifact inventory is missing or unexpectedly large")
    return [content_record(path, path.relative_to(build_dir).as_posix()) for path in sorted(set(paths))]


def final_identity(build_dir: Path) -> dict:
    source = source_identity()
    return {"schema_version": 2, "source": source, "sdk": sdk_identity(),
            "submission": verify_submission_snapshot(build_dir, source),
            "artifacts": _final_artifacts(build_dir)}


def verify_final_build(build_dir: Path) -> dict:
    receipt = build_dir / "tests/final_build_state.json"
    if not receipt.is_file():
        raise StepFailure(f"final build receipt is missing: {receipt}")
    expected = json.loads(receipt.read_text())
    actual = final_identity(build_dir)
    if actual != expected:
        raise StepFailure("final build source/toolchain/artifact identity differs")
    return actual


def clean_final_build_dir(path: Path | None) -> Path:
    selected = path or DEFAULT_FINAL_BUILD
    if selected.exists() and any(selected.iterdir()):
        raise StepFailure(f"final build directory must be absent or empty: {selected}")
    selected.mkdir(parents=True, exist_ok=True)
    return selected


def build_full(build_dir: Path, jobs: int) -> float:
    snapshot, source_before = snapshot_submission(build_dir)
    build_result = build_dir / "tests/build_result.json"
    try:
        elapsed = configure(build_dir, jobs, source_dir=snapshot)
        elapsed += _run_build(
            ("cmake", "--build", str(build_dir), "--target", "custom", "--parallel", str(jobs)),
            build_dir, "final-all-templates", jobs,
        )
    except StepFailure as error:
        build_result.write_text(json.dumps({"status": "Compile Error", "source": source_before,
            "returncode": error.returncode, "diagnostic": str(error)}, indent=2) + "\n")
        raise
    if source_identity() != source_before:
        raise StepFailure("operator sources changed during final build")
    receipt = build_dir / "tests/final_build_state.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(final_identity(build_dir), indent=2) + "\n")
    verify_final_build(build_dir)
    build_result.write_text(json.dumps({"status": "Pass", "source": source_before}) + "\n")
    return elapsed


def select_cases(suite: str, names: Sequence[str], tags: Sequence[str]) -> tuple[Case, ...]:
    available = () if suite == "custom" else accepted_cases(suite)
    selected_names = list(dict.fromkeys((*[case.name for case in available], *names)))
    if suite in ("coverage", "correctness"):
        outside = set(names) - {case.name for case in available}
        if outside:
            raise ValueError(f"cases outside {suite}: {sorted(outside)}")
    cases = tuple(case_by_name(name) for name in selected_names)
    wanted_tags = set(tags)
    cases = tuple(case for case in cases if wanted_tags.issubset(case.tags))
    if not cases:
        raise ValueError("no matching cases")
    return cases


def _fixture_key(case: Case) -> str:
    import numpy as np

    identity = {
        "case": asdict(case),
        "dependencies": {name: sha256_file(ROOT / "tests" / name) for name in (
            "st/scripts/gen_data.py", "common/reference.py", "common/tensor_io.py", "common/cases.py",
        )},
        "numpy_version": np.__version__,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def materialize_fixtures(cases: Sequence[Case], root: Path, allow_large: bool) -> list[Path]:
    fixtures = []
    for case in cases:
        if not allow_large and case.largest_tensor_numel > DEFAULT_MAX_ELEMENTS:
            raise StepFailure(f"large fixture requires --allow-large: {case.name}")
        directory = root / case.name / _fixture_key(case)[:20]
        completion = directory / ".complete"
        if not completion.is_file():
            if directory.exists():
                shutil.rmtree(directory)
            materialize_case(case, directory.parent)
            generated = directory.parent / case.name
            if generated != directory:
                directory.parent.mkdir(parents=True, exist_ok=True)
                generated.rename(directory)
            records = [content_record(path, path.name) for path in sorted(directory.iterdir()) if path.is_file()]
            completion.write_text(json.dumps(records, sort_keys=True) + "\n")
            print("generated fixture:", case.name, flush=True)
        else:
            records = json.loads(completion.read_text())
            actual = [content_record(path, path.name) for path in sorted(directory.iterdir())
                      if path.is_file() and path.name != ".complete"]
            if records != actual:
                raise StepFailure(f"fixture cache changed: {directory}")
            print("reused fixture:", case.name, flush=True)
        fixtures.append(directory)
    return fixtures


def _json_quote(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def _plan(fixtures: Sequence[Path], run_dir: Path, workspace_only: bool) -> tuple[Path, list[dict]]:
    plan = run_dir / "runner.plan"
    calls = []
    lines = []
    for fixture in fixtures:
        metadata = json.loads((fixture / "case.json").read_text())
        for index, call in enumerate(metadata["calls"]):
            label = f"{metadata['name']}#{index}"
            output = run_dir / "outputs" / metadata["name"] / f"call{index}" / "actual.bin"
            output.parent.mkdir(parents=True, exist_ok=True)
            guard = metadata["runner_checks"]["guard_regions"]["bytes_before"]
            poison = "0x7fc0" if metadata["dtype"] == "bfloat16" else "0x7e00"
            fields = (
                _json_quote(label), _json_quote(fixture / call["input"]), _json_quote(output),
                str(metadata["S"]), str(metadata["D"]), str(call["attrs"]["mhc_mult"]),
                str(int(call["attrs"]["backward"])), str(int(metadata["dtype"] == "bfloat16")),
                "0", "0", str(metadata["runner_checks"]["repeat_count"]), str(guard), poison,
            )
            lines.append(" ".join(fields))
            calls.append({"label": label, "fixture": fixture, "call_index": index,
                          "output": output, "metadata": metadata, "workspace_only": workspace_only})
    plan.write_text("\n".join(lines) + "\n")
    return plan, calls


def _evaluate_calls(calls: Sequence[dict], log_path: Path, library: Path,
                    returncode: int | None, workspace_only: bool) -> dict:
    """Compare this execution's outputs, including calls completed before an RE."""
    native = {}
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = record.get("case")
        if label in native:
            raise StepFailure(f"duplicate native case result: {label}")
        native[label] = record
    labels = [call["label"] for call in calls]
    if len(labels) != len(set(labels)) or set(native) - set(labels):
        raise StepFailure("native results and planned calls disagree")
    results = []
    for call in calls:
        record = native.get(call["label"])
        item = {"label": call["label"], "runner": record, "status": "INCONCLUSIVE"}
        loaded = record and record.get("library")
        valid_library = loaded and Path(loaded).resolve() == library.resolve()
        if workspace_only:
            if record and record.get("status") == "workspace" and valid_library:
                item["status"] = "Pass"
        elif record and record.get("status") == "executed" and record.get("guards_ok") is True and valid_library:
            outputs = (call["output"], Path(str(call["output"]) + ".final.bin"))
            if not all(path.is_file() for path in outputs):
                item["diagnostic"] = "completed call is missing first or final output"
            else:
                first, final = (verify_case(call["fixture"], path, call["call_index"]) for path in outputs)
                metadata = call["metadata"]
                spec = metadata["calls"][call["call_index"]]
                elements = next(t["numel"] for t in metadata["tensors"].values() if t["file"] == spec["output"])
                item.update(status="Pass" if first.passed and final.passed else "Wrong Answer",
                            first=asdict(first), final=asdict(final),
                            first_error_ratio=f"{100 * first.mismatch_count / elements:.6f}%",
                            final_error_ratio=f"{100 * final.mismatch_count / elements:.6f}%")
        results.append(item)
    status = "Wrong Answer" if any(item["status"] == "Wrong Answer" for item in results) else (
        "Pass" if returncode == 0 and results and all(item["status"] == "Pass" for item in results)
        else "INCONCLUSIVE"
    )
    return {"status": status, "passed": status == "Pass", "calls": results}


def _run_npu(build_dir: Path, opapi_dir: Path, overlays: Sequence[Path], fixtures: Sequence[Path],
             results_root: Path, jobs: int, device: int, workspace_only: bool,
             timeout: int) -> Path:
    runner, runner_manifest, _ = build_runner(build_dir, jobs)
    run_dir = results_root / (time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    run_dir.mkdir(parents=True, exist_ok=False)
    plan, calls = _plan(fixtures, run_dir, workspace_only)
    library = opapi_dir / "libcust_opapi.so"
    command = [str(runner), "--library", str(library), "--plan", str(plan), "--device", str(device)]
    if workspace_only:
        command.append("--workspace-only")
    environment = runtime_environment(opapi_dir)
    opp = [str(path) for path in overlays]
    built_opp = build_dir / "tmp/vendors/custom"
    if built_opp.is_dir():
        opp.append(str(built_opp))
    if not workspace_only and not opp:
        raise StepFailure("no representative or final runtime operator package")
    if opp:
        environment["ASCEND_CUSTOM_OPP_PATH"] = ":".join(opp)
    artifacts = [runner, library, build_dir / "op_host/libcustom_ascendc_cust_optiling.so"]
    artifacts.extend(path for root in overlays for path in root.rglob("*") if path.is_file())
    if built_opp.is_dir():
        artifacts.extend(path for path in built_opp.rglob("*") if path.is_file())
    fixture_files = [path for fixture in fixtures for path in fixture.iterdir() if path.is_file()]
    manifest = {
        "schema_version": 2,
        "command": command,
        "source": source_identity(),
        "workspace_only": workspace_only,
        "library": str(library.resolve()),
        "fixtures": [content_record(path, str(path.resolve())) for path in fixture_files],
        "artifacts": [content_record(path, str(path.resolve())) for path in sorted(set(artifacts))],
        "runner": runner_manifest,
        "plan": content_record(plan, "runner.plan"),
        "calls": [{"label": call["label"], "fixture": str(call["fixture"].resolve()),
                   "call_index": call["call_index"],
                   "output": call["output"].relative_to(run_dir).as_posix()} for call in calls],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    lock_path = Path("/tmp/cannjudge/mhcexpand/device.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (run_dir / "runner.log").open("w") as log:
            try:
                process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=timeout, check=False)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = None
    manifest["returncode"] = returncode
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["runner_log"] = content_record(run_dir / "runner.log", "runner.log")
    manifest["outputs"] = [content_record(path, path.relative_to(run_dir).as_posix())
                           for path in sorted((run_dir / "outputs").rglob("*")) if path.is_file()]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for record in manifest["artifacts"] + manifest["fixtures"]:
        if content_record(Path(record["path"]), record["path"]) != record:
            raise StepFailure(f"test dependency changed during execution: {record['path']}")
    result = _evaluate_calls(calls, run_dir / "runner.log", library, returncode, workspace_only)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise StepFailure(f"NPU validation failed ({result['status']}): {run_dir}",
                          1 if result["status"] == "Wrong Answer" else 2)
    print(f"PASS {'workspace' if workspace_only else 'NPU'} calls={len(calls)} results={run_dir}")
    return run_dir


def _overlays_for(cases: Sequence[Case], build_dir: Path, jobs: int,
                  cache_dir: Path) -> tuple[list[Path], float]:
    overlays = []
    elapsed = 0.0
    for dtype in sorted({case.dtype for case in cases}):
        duration, overlay = build_representative_kernel(
            build_dir, jobs, DTYPE_REPRESENTATIVES[dtype], cache_dir
        )
        elapsed += duration
        overlays.append(overlay)
    return overlays, elapsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=LAYERS, default="pipeline")
    parser.add_argument("--stage", choices=("quick", "final"), default="quick")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_INCREMENTAL_BUILD)
    parser.add_argument("--final-build-dir", type=Path, default=DEFAULT_FINAL_BUILD)
    parser.add_argument("--kernel-cache", type=Path, default=DEFAULT_KERNEL_CACHE)
    parser.add_argument("--fixture-cache", type=Path, default=DEFAULT_FIXTURE_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--suite", choices=("auto", "custom", "coverage", "correctness", "full"),
                        default="auto")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--representative-case", default="forward_scalar_float16")
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = resolve_jobs(args.jobs, os.environ)
    timings: dict[str, float] = {}

    def chosen(layer: str, names: Sequence[str] | None = None) -> tuple[Case, ...]:
        suite = args.suite
        if suite == "auto":
            suite = "correctness" if layer in ("correctness", "generate", "validate") else "custom"
        directed = tuple(args.case if names is None else names)
        return select_cases(suite, directed, args.tag)

    def execute_incremental(cases: Sequence[Case], workspace: bool = False) -> None:
        overlays, elapsed = _overlays_for(cases, args.build_dir, jobs, args.kernel_cache)
        timings["kernel"] = timings.get("kernel", 0.0) + elapsed
        verify_host(args.build_dir)
        opapi = args.build_dir / "tests/host_only_opapi"
        fixtures = materialize_fixtures(cases, args.fixture_cache, args.allow_large)
        _run_npu(args.build_dir, opapi, overlays, fixtures, args.results_dir, jobs,
                 args.device, workspace, args.timeout)

    if args.layer == "static":
        timings.update(static_checks())
    elif args.layer == "ut":
        timings["python_ut"] = python_checks()
    elif args.layer == "generate":
        cases = chosen("generate")
        materialize_fixtures(cases, args.fixture_cache, args.allow_large)
    elif args.layer == "host":
        timings["host"] = build_host(args.build_dir, jobs)
    elif args.layer == "kernel":
        timings["kernel"], _ = build_representative_kernel(
            args.build_dir, jobs, args.representative_case, args.kernel_cache
        )
    elif args.layer == "workspace":
        execute_incremental(tuple(case_by_name(name) for name in DTYPE_REPRESENTATIVES.values()), True)
    elif args.layer == "smoke":
        execute_incremental(chosen("smoke", SMOKE_CASES))
    elif args.layer == "correctness":
        execute_incremental(chosen("correctness"))
    elif args.layer == "validate":
        timings.update(static_checks())
        timings["python_ut"] = python_checks()
        timings["host"] = build_host(args.build_dir, jobs)
        execute_incremental(chosen("correctness"))
    elif args.layer == "full-build":
        timings["full_build"] = build_full(clean_final_build_dir(args.final_build_dir), jobs)
    elif args.layer == "full-test":
        verify_final_build(args.final_build_dir)
        cases = select_cases("full" if args.suite == "auto" else args.suite, args.case, args.tag)
        fixtures = materialize_fixtures(cases, args.fixture_cache, args.allow_large)
        _run_npu(args.final_build_dir, args.final_build_dir, (), fixtures, args.results_dir,
                 jobs, args.device, False, args.timeout)
    elif args.layer == "full":
        final = clean_final_build_dir(args.final_build_dir)
        timings["full_build"] = build_full(final, jobs)
        cases = select_cases("full", args.case, args.tag)
        fixtures = materialize_fixtures(cases, args.fixture_cache, args.allow_large)
        _run_npu(final, final, (), fixtures, args.results_dir, jobs,
                 args.device, False, args.timeout)
    else:
        timings.update(static_checks())
        timings["python_ut"] = python_checks()
        timings["host"] = build_host(args.build_dir, jobs)
        execute_incremental(tuple(case_by_name(name) for name in DTYPE_REPRESENTATIVES.values()), True)
        execute_incremental(chosen("smoke", SMOKE_CASES))
        execute_incremental(select_cases("correctness", args.case, args.tag))
        if args.stage == "final":
            if not args.allow_large:
                raise StepFailure("final pipeline requires --allow-large for the complete matrix")
            final = clean_final_build_dir(args.final_build_dir)
            timings["full_build"] = build_full(final, jobs)
            cases = select_cases("full", (), ())
            fixtures = materialize_fixtures(cases, args.fixture_cache, True)
            _run_npu(final, final, (), fixtures, args.results_dir, jobs,
                     args.device, False, args.timeout)
    print(json.dumps({"status": "PASS", "layer": args.layer, "timings": timings}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, StepFailure) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(error.returncode if isinstance(error, StepFailure) and error.returncode else 2)
