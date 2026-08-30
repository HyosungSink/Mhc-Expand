#!/usr/bin/env python3
"""Archive one verified final package before reusing the fixed final directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from tests.st.scripts.kernel_cache import content_record
from tests.st.scripts.run_tests import (
    DEFAULT_FINAL_BUILD,
    DEFAULT_WORK_ROOT,
    ROOT,
    TEMPLATE_FILES,
    source_identity,
)


CATALOG = DEFAULT_WORK_ROOT / "artifacts/relocations.json"


def commit_source(commit: str) -> str:
    digest = hashlib.sha256(b"mhc-expand-seven-files-v1")
    for name in sorted(TEMPLATE_FILES):
        label = "code/" + name
        digest.update(label.encode())
        digest.update(subprocess.check_output(("git", "show", f"{commit}:{label}"), cwd=ROOT))
    return "source-sha256:" + digest.hexdigest()


def verify_receipt(build_dir: Path) -> dict:
    path = build_dir / "tests/final_build_state.json"
    receipt = json.loads(path.read_text())
    for record in receipt["artifacts"]:
        artifact = build_dir / record["path"]
        if content_record(artifact, record["path"]) != record:
            raise ValueError(f"final artifact changed: {artifact}")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--commit")
    identity.add_argument("--candidate", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_FINAL_BUILD)
    parser.add_argument("--archive-root", type=Path, default=CATALOG.parent)
    args = parser.parse_args(argv)
    receipt = verify_receipt(args.build_dir)
    expected = source_identity() if args.candidate else commit_source(args.commit)
    if receipt["source"] != expected:
        raise ValueError("final package source does not match the requested identity")
    label = ("candidate-" + expected.removeprefix("source-sha256:")) if args.candidate else args.commit
    destination = args.archive_root / label / "final_package"
    if destination.exists():
        raise ValueError(f"archive destination exists: {destination}")
    catalog_path = args.archive_root / "relocations.json"
    relocations = json.loads(catalog_path.read_text()) if catalog_path.is_file() else []
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "label": label,
        "source": expected,
        "original": str(args.build_dir.resolve()),
        "archive": str(destination.resolve()),
        "receipt_sha256": hashlib.sha256(
            (args.build_dir / "tests/final_build_state.json").read_bytes()
        ).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.build_dir.rename(destination)
    catalog_path.write_text(json.dumps([*relocations, record], indent=2) + "\n")
    print("archived:", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
