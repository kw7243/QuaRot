#!/usr/bin/env python3
"""Validate a full reproducibility stage without comparing mutable Git indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXCLUDES = [
    "/staging",
    "/out",
    "/outputs",
    "/eval_outputs",
    "/slurm_outputs",
    "/wandb",
    "/.cache",
    "/__pycache__",
    "/.venv",
    "/venv",
    "/.git/index",
    "/.git/modules/**/index",
]


class ValidationError(RuntimeError):
    pass


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode != 0:
        rendered = " ".join(command)
        raise ValidationError(
            f"command failed ({completed.returncode}): {rendered}\n{completed.stdout}"
        )
    return completed.stdout


def git(repo: Path, *arguments: str, check: bool = True) -> str:
    return run(["git", "--no-optional-locks", "-C", str(repo), *arguments], check=check)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_index_sha256(repo: Path) -> str:
    output = run(
        ["git", "--no-optional-locks", "-C", str(repo), "ls-files", "--stage", "-z"]
    ).encode()
    return hashlib.sha256(output).hexdigest()


def submodule_paths(repo: Path) -> list[str]:
    gitmodules = repo / ".gitmodules"
    if not gitmodules.is_file():
        return []
    output = git(
        repo,
        "config",
        "--file",
        str(gitmodules),
        "--get-regexp",
        r"^submodule\..*\.path$",
        check=False,
    )
    paths = []
    for line in output.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            paths.append(fields[1])
    return sorted(paths)


def common_git_dir(repo: Path) -> Path:
    value = git(repo, "rev-parse", "--git-common-dir").strip()
    path = Path(value)
    return path if path.is_absolute() else (repo / path).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def clean_status(repo: Path, *, allow_metadata: bool) -> str:
    status = git(repo, "status", "--porcelain", "--untracked-files=all")
    lines = [line for line in status.splitlines() if line]
    if allow_metadata:
        lines = [line for line in lines if line != "?? REPRODUCIBILITY_METADATA.json"]
    require(not lines, f"unexpected Git status in {repo}: {lines}")
    return "clean"


def validate_repository(repo: Path) -> dict[str, Any]:
    common = common_git_dir(repo)
    alternates = common / "objects" / "info" / "alternates"
    require(not alternates.exists(), f"Git object alternate present: {alternates}")
    git(repo, "fsck", "--full", "--no-dangling")
    return {
        "path": str(repo),
        "head": git(repo, "rev-parse", "HEAD").strip(),
        "tree": git(repo, "rev-parse", "HEAD^{tree}").strip(),
        "semantic_index_sha256": semantic_index_sha256(repo),
        "object_alternates": "absent",
        "fsck_full_no_dangling": "PASS",
    }


def validate(
    source: Path,
    stage: Path,
    expected_commit: str,
    delta_path: Path,
) -> dict[str, Any]:
    require(source.is_dir() and (source / ".git").exists(), "source is not a Git worktree")
    require(stage.is_dir() and (stage / ".git").exists(), "stage is not a full Git worktree")
    metadata = stage / "REPRODUCIBILITY_METADATA.json"
    require(metadata.is_file(), "stage reproducibility metadata is absent")
    json.loads(metadata.read_text())

    source_status = clean_status(source, allow_metadata=False)
    stage_status = clean_status(stage, allow_metadata=True)
    source_head = git(source, "rev-parse", "HEAD").strip()
    stage_head = git(stage, "rev-parse", "HEAD").strip()
    require(source_head == expected_commit, "source HEAD differs from expected commit")
    require(stage_head == expected_commit, "stage HEAD differs from expected commit")
    source_tree = git(source, "rev-parse", "HEAD^{tree}").strip()
    stage_tree = git(stage, "rev-parse", "HEAD^{tree}").strip()
    require(source_tree == stage_tree, "source and stage trees differ")

    source_submodules = git(source, "submodule", "status", "--recursive")
    stage_submodules = git(stage, "submodule", "status", "--recursive")
    for label, value in (("source", source_submodules), ("stage", stage_submodules)):
        bad = [line for line in value.splitlines() if line.startswith(("-", "+", "U"))]
        require(not bad, f"{label} has uninitialized or mismatched submodules: {bad}")
    require(source_submodules == stage_submodules, "source/stage submodule status differs")

    roots = {
        "parent": {
            "source": validate_repository(source),
            "stage": validate_repository(stage),
        }
    }
    require(
        roots["parent"]["source"]["semantic_index_sha256"]
        == roots["parent"]["stage"]["semantic_index_sha256"],
        "parent semantic index differs",
    )
    source_paths = submodule_paths(source)
    require(source_paths == submodule_paths(stage), "source/stage submodule path set differs")
    for relative in source_paths:
        source_submodule = source / relative
        stage_submodule = stage / relative
        roots[relative] = {
            "source": validate_repository(source_submodule),
            "stage": validate_repository(stage_submodule),
        }
        require(
            roots[relative]["source"]["head"] == roots[relative]["stage"]["head"],
            f"submodule HEAD differs: {relative}",
        )
        require(
            roots[relative]["source"]["semantic_index_sha256"]
            == roots[relative]["stage"]["semantic_index_sha256"],
            f"submodule semantic index differs: {relative}",
        )

    command = ["rsync", "-rlnic"]
    for pattern in EXCLUDES:
        command.extend(["--exclude", pattern])
    command.extend([f"{source}/", f"{stage}/"])
    delta = run(command)
    delta_path.write_text(delta)
    require(delta == "", f"semantic checksum delta is nonempty; see {delta_path}")

    return {
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "stage": str(stage),
        "expected_commit": expected_commit,
        "source_status": source_status,
        "stage_status_except_metadata": stage_status,
        "metadata": {
            "path": str(metadata),
            "sha256": sha256_file(metadata),
            "json": "PASS",
        },
        "submodule_status": stage_submodules.splitlines(),
        "repositories": roots,
        "semantic_checksum": {
            "command": command,
            "excluded_mutable_implementation_files": [
                "/.git/index",
                "/.git/modules/**/index",
            ],
            "delta_path": str(delta_path),
            "delta": "empty",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("expected_commit")
    parser.add_argument("report", type=Path)
    parser.add_argument("delta", type=Path)
    arguments = parser.parse_args()
    require(not arguments.report.exists(), f"refusing to overwrite {arguments.report}")
    require(not arguments.delta.exists(), f"refusing to overwrite {arguments.delta}")
    try:
        report = validate(
            arguments.source.resolve(),
            arguments.stage.resolve(),
            arguments.expected_commit,
            arguments.delta.resolve(),
        )
        return_code = 0
    except Exception as error:
        report = {
            "status": "FAIL",
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
            "source": str(arguments.source.resolve()),
            "stage": str(arguments.stage.resolve()),
            "expected_commit": arguments.expected_commit,
            "delta_path": str(arguments.delta.resolve()),
        }
        return_code = 1
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if return_code:
        print(report["error"], file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
