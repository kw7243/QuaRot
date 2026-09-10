#!/usr/bin/env python3
"""Verify a successful CPU CUDA-build preflight before E0 GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "refusing to overwrite verification output")
    value = json.loads(arguments.manifest.read_text())
    require(value["status"] == "PASS", "CUDA build preflight did not pass")
    stage_commit = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(arguments.stage), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    require(stage_commit == value["source"]["commit"], "preflight/stage commit mismatch")
    require(
        arguments.stage.resolve() == Path(value["source"]["path"]).resolve(),
        "GPU stage differs from the CPU preflight stage",
    )
    require(value["cuda"]["release"] == "12.1", "unexpected CUDA release")
    require(value["python"]["torch"] == "2.4.1+cu121", "unexpected PyTorch build")
    for name, artifact in value["artifacts"].items():
        path = Path(artifact["path"])
        require(path.is_file(), f"preflight artifact is absent: {name}: {path}")
        require(sha256_file(path) == artifact["sha256"], f"preflight artifact changed: {name}")
    environment = Path(value["environment"])
    build_source = Path(value["build_source"])
    require(environment.is_dir(), "preflight environment is absent")
    require(build_source.is_dir(), "preflight build source is absent")
    report = {
        "status": "PASS",
        "manifest": str(arguments.manifest.resolve()),
        "manifest_sha256": sha256_file(arguments.manifest),
        "stage": str(arguments.stage.resolve()),
        "stage_commit": stage_commit,
        "environment": str(environment),
        "build_source": str(build_source),
        "verified_artifacts": sorted(value["artifacts"]),
    }
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{environment}|{build_source}|{report['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
