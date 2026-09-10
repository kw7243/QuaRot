#!/usr/bin/env python3
"""No-job counterfactual tests for the E0 semantic stage validator."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_stage.py"
SPEC = importlib.util.spec_from_file_location("validate_stage", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def command(*arguments: str, cwd: Path) -> str:
    return subprocess.check_output(arguments, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def initialize_repo(path: Path, filename: str = "tracked.txt") -> None:
    path.mkdir()
    command("git", "init", "-q", cwd=path)
    command("git", "config", "user.name", "E0 Test", cwd=path)
    command("git", "config", "user.email", "e0-test@example.invalid", cwd=path)
    (path / filename).write_text("abcdefgh\n")
    command("git", "add", filename, cwd=path)
    command("git", "commit", "-qm", "fixture", cwd=path)


def clone_as_stage(source: Path, stage: Path) -> str:
    shutil.copytree(source, stage, symlinks=True)
    metadata = {
        "source_repo": str(source),
        "staged_repo": str(stage),
        "git_commit_full": command("git", "rev-parse", "HEAD", cwd=source).strip(),
    }
    (stage / "REPRODUCIBILITY_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata["git_commit_full"]


class StageValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="absym-e0-validator-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.root / "source",
            self.root / "stage",
            self.root / "delta.txt",
            self.root / "report.json",
        )

    def test_post_copy_index_refresh_is_accepted(self) -> None:
        source, stage, delta, _ = self.paths()
        initialize_repo(source)
        commit = clone_as_stage(source, stage)
        before = (stage / ".git" / "index").read_bytes()
        command("git", "status", "--short", cwd=stage)
        after = (stage / ".git" / "index").read_bytes()
        self.assertNotEqual(before, after, "fixture did not exercise index stat-cache refresh")
        report = VALIDATOR.validate(source, stage, commit, delta)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(delta.read_text(), "")

    def test_same_size_same_mtime_tracked_tamper_is_rejected(self) -> None:
        source, stage, delta, _ = self.paths()
        initialize_repo(source)
        commit = clone_as_stage(source, stage)
        target = stage / "tracked.txt"
        timestamp = target.stat().st_mtime_ns
        target.write_text("abcdEfgh\n")
        os.utime(target, ns=(timestamp, timestamp))
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate(source, stage, commit, delta)

    def test_uninitialized_submodule_is_rejected(self) -> None:
        source, stage, delta, _ = self.paths()
        dependency = self.root / "dependency"
        initialize_repo(dependency, "dependency.txt")
        initialize_repo(source)
        command(
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(dependency),
            "deps/dependency",
            cwd=source,
        )
        command("git", "commit", "-qam", "add submodule", cwd=source)
        commit = clone_as_stage(source, stage)
        shutil.rmtree(stage / "deps" / "dependency")
        shutil.rmtree(stage / ".git" / "modules" / "deps" / "dependency")
        (stage / "deps" / "dependency").mkdir()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate(source, stage, commit, delta)

    def test_missing_git_object_is_rejected(self) -> None:
        source, stage, delta, _ = self.paths()
        initialize_repo(source)
        commit = clone_as_stage(source, stage)
        blob = command("git", "rev-parse", "HEAD:tracked.txt", cwd=stage).strip()
        object_path = stage / ".git" / "objects" / blob[:2] / blob[2:]
        self.assertTrue(object_path.is_file())
        object_path.unlink()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate(source, stage, commit, delta)


if __name__ == "__main__":
    unittest.main()
