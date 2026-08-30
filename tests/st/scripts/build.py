#!/usr/bin/env python3
"""Focused layered build entry point for mHC Expand."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from tests.st.scripts.run_tests import (
    DEFAULT_FINAL_BUILD,
    DEFAULT_INCREMENTAL_BUILD,
    DEFAULT_KERNEL_CACHE,
    StepFailure,
    build_full,
    build_host,
    build_representative_kernel,
    clean_final_build_dir,
    configure,
    resolve_jobs,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("bootstrap", "host", "kernel", "final"))
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_INCREMENTAL_BUILD)
    parser.add_argument("--final-build-dir", type=Path, default=DEFAULT_FINAL_BUILD)
    parser.add_argument("--kernel-cache", type=Path, default=DEFAULT_KERNEL_CACHE)
    parser.add_argument("--representative-case", default="forward_scalar_float16")
    args = parser.parse_args(argv)
    jobs = resolve_jobs(args.jobs, os.environ)
    if args.target == "bootstrap":
        elapsed = configure(args.build_dir, jobs)
        active = args.build_dir
    elif args.target == "host":
        elapsed = build_host(args.build_dir, jobs)
        active = args.build_dir
    elif args.target == "kernel":
        elapsed, _ = build_representative_kernel(
            args.build_dir, jobs, args.representative_case, args.kernel_cache
        )
        active = args.build_dir
    else:
        active = clean_final_build_dir(args.final_build_dir)
        elapsed = build_full(active, jobs)
    print(f"PASS target={args.target} jobs={jobs} build_dir={active} elapsed={elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, StepFailure) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(error.returncode if isinstance(error, StepFailure) and error.returncode else 2)
