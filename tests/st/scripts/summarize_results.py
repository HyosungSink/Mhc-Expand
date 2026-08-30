#!/usr/bin/env python3
"""Verify one layered run manifest and emit an independent summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.st.scripts.kernel_cache import content_record


def summarize(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = manifest["artifacts"] + manifest["fixtures"]
    for record in records:
        path = Path(record["path"])
        if content_record(path, record["path"]) != record:
            raise ValueError(f"recorded artifact changed: {path}")
    results = json.loads((run_dir / "results.json").read_text())
    passed = (
        manifest.get("returncode") == 0
        and results.get("passed") is True
        and results.get("status") == "Pass"
        and bool(results.get("calls"))
        and all(row.get("status") == "Pass" for row in results["calls"])
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "source": manifest["source"],
        "workspace_only": manifest["workspace_only"],
        "call_count": len(results.get("calls", [])),
        "artifact_hashes_verified": len(records),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "manifest": str(run_dir / "manifest.json"),
        "results": str(run_dir / "results.json"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    report = summarize(args.run_dir.resolve())
    output = args.run_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], "calls=", report["call_count"], "summary=", output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
