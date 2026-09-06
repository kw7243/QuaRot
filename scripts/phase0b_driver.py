#!/usr/bin/env python3
"""Single-owner Phase 0B runner around QuaRot's official fake-quant harness."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase0b_audit import load_and_validate_config, quantization_site_audit


RESULT_FIELDS = [
    "run_id",
    "model",
    "model_revision",
    "tokenizer_revision",
    "dataset_revision",
    "commit",
    "source_repo",
    "quantization_setting",
    "calibration_setting",
    "transform_family",
    "transform_optimization_steps",
    "full_precision_equivalence_error",
    "block_nmse",
    "wikitext_ppl",
    "gpu_type",
    "gpu_hours",
    "stage_path",
    "durable_output_path",
    "notes_or_failure_reason",
]
PPL_RE = re.compile(r"WIKITEXT2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new_json(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_logged(label: str, command: list[str], cwd: Path, env: dict[str, str], logs: Path) -> tuple[str, float]:
    log_path = logs / f"{label}.log"
    started = time.monotonic()
    with log_path.open("x") as stream:
        stream.write("command=" + shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        chunks = []
        for line in process.stdout:
            stream.write(line)
            chunks.append(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    elapsed = time.monotonic() - started
    output = "".join(chunks)
    require(returncode == 0, f"{label} exited {returncode}; see {log_path}")
    return output, elapsed


def get_gpu_name() -> str:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    require(len(result) == 1, f"expected exactly one visible GPU, got {result}")
    return result[0]


def base_row(config: dict[str, Any], run_id: str, setting: str, transform: str, gpu: str, stage: Path, output: Path) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "dataset_revision": config["dataset"]["revision"],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_repo": config["source"]["fork_url"],
        "quantization_setting": setting,
        "calibration_setting": "128x2048 seed=0 WikiText2 train windows (RTN does not consume calibration)",
        "transform_family": transform,
        "transform_optimization_steps": 0,
        "full_precision_equivalence_error": "",
        "block_nmse": "",
        "wikitext_ppl": "",
        "gpu_type": gpu,
        "gpu_hours": "",
        "stage_path": str(stage),
        "durable_output_path": str(output),
        "notes_or_failure_reason": "",
    }


def condition_command(config: dict[str, Any], condition: dict[str, Any]) -> list[str]:
    python = Path(config["storage"]["environment"]) / "bin" / "python"
    command = [
        str(python),
        "main.py",
        "--model", config["model"]["id"],
        "--seed", str(config["evaluation"]["seed"]),
        "--eval_dataset", "wikitext2",
        "--bsz", str(config["evaluation"]["batch_size"]),
    ]
    if condition["weight_bits"] < 16:
        quant = config["quantization"]
        command.extend(
            [
                "--w_bits", str(condition["weight_bits"]),
                "--a_bits", str(condition["activation_bits"]),
                "--k_bits", str(condition["key_bits"]),
                "--v_bits", str(condition["value_bits"]),
                "--w_groupsize", str(quant["weight_group_size"]),
                "--a_groupsize", str(quant["activation_group_size"]),
                "--k_groupsize", str(quant["key_group_size"]),
                "--v_groupsize", str(quant["value_group_size"]),
                "--a_clip_ratio", str(quant["activation_clip_ratio"]),
                "--k_clip_ratio", str(quant["key_clip_ratio"]),
                "--v_clip_ratio", str(quant["value_clip_ratio"]),
                "--w_rtn",
            ]
        )
    if condition["rotate"]:
        command.extend(["--rotate", "--rotate_mode", "hadamard", "--fp32_had"])
    return command


def parse_ppl(output: str, label: str) -> float:
    matches = PPL_RE.findall(output)
    require(matches, f"{label}: PPL marker not found")
    value = float(matches[-1])
    require(math.isfinite(value), f"{label}: non-finite PPL")
    return value


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: phase0b_driver.py CONFIG DURABLE_OUTPUT_DIR")
    config_path = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    config = load_and_validate_config(config_path)
    research_root = Path(config["storage"]["research_root"]).resolve()
    scratch_root = Path(config["storage"]["scratch_root"]).resolve()
    stage = Path(os.environ["RESEARCH_REPRO_STAGED_DIR"]).resolve()
    credential = Path(config["storage"]["credential_path"])
    environment = Path(config["storage"]["environment"])

    require(str(output).startswith(str(research_root) + os.sep), "output outside RESEARCH_ROOT")
    require(str(stage).startswith(str(scratch_root) + os.sep), "stage outside SCRATCH_ROOT")
    require(str(environment.resolve()).startswith(str(scratch_root) + os.sep), "environment outside SCRATCH_ROOT")
    require(not output.exists() and not output.is_symlink(), "durable output already exists")
    require((stage / "REPRODUCIBILITY_METADATA.json").is_file(), "stage metadata absent")
    require(credential.is_file() and os.access(credential, os.R_OK), "authorized Hugging Face credential absent")
    require(os.environ.get("SLURM_JOB_ID"), "Slurm job id absent")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible and "," not in visible, f"expected exactly one CUDA_VISIBLE_DEVICES entry, got {visible!r}")
    require(Path(os.environ["TMPDIR"]).is_dir(), "allocation-local TMPDIR absent")

    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(stage / "REPRODUCIBILITY_METADATA.json", output / "REPRODUCIBILITY_METADATA.json")
    lock = environment / "phase0b-freeze.txt"
    require(lock.is_file(), f"environment lock absent: {lock}")
    shutil.copy2(lock, output / "environment-freeze.txt")

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": config["storage"]["hf_home"],
            "HF_HUB_CACHE": str(Path(config["storage"]["hf_home"]) / "hub"),
            "HF_DATASETS_CACHE": str(Path(config["storage"]["hf_home"]) / "datasets"),
            "HF_TOKEN_PATH": str(credential),
            "PIP_CACHE_DIR": config["storage"]["pip_cache"],
            "PHASE0B_MODEL_REVISION": config["model"]["revision"],
            "PHASE0B_TOKENIZER_REVISION": config["model"]["tokenizer_revision"],
            "PHASE0B_DATASET_REVISION": config["dataset"]["revision"],
            "PHASE0B_TORCH_DTYPE": config["model"]["dtype"],
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": str(config["evaluation"]["seed"]),
        }
    )
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    gpu = get_gpu_name()
    attempt_id = f"phase0b-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.environ['SLURM_JOB_ID']}"
    preflight = {
        "attempt_id": attempt_id,
        "created_at_utc": utc_now(),
        "status": "RUNNING",
        "single_job_owner": True,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "gpu": gpu,
        "roots": {"research": str(research_root), "scratch": str(scratch_root), "tmpdir": env["TMPDIR"]},
        "stage": str(stage),
        "stage_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": config,
        "quantization_site_audit": quantization_site_audit(),
    }
    write_new_json(output / "manifest.preflight.json", preflight)

    rows: list[dict[str, Any]] = []
    current = "preflight"
    started_all = time.monotonic()
    failure = ""
    try:
        audit_command = [
            str(environment / "bin" / "python"),
            "scripts/phase0b_audit.py",
            str(config_path),
            "--source-root", str(stage),
            "--require-clean",
        ]
        audit_output, _ = run_logged("source-and-quantizer-audit", audit_command, stage, env, logs)
        (output / "quantization-audit.json").write_text(audit_output)

        fp_values = []
        for rerun in range(1, config["evaluation"]["deterministic_reruns"] + 1):
            current = f"fp16-r{rerun}"
            condition = config["conditions"][0]
            text, elapsed = run_logged(current, condition_command(config, condition), stage / "fake_quant", env, logs)
            ppl = parse_ppl(text, current)
            row = base_row(config, f"{attempt_id}-{current}", "FP16", "none", gpu, stage, output)
            row["wikitext_ppl"] = ppl
            row["gpu_hours"] = elapsed / 3600.0
            rows.append(row)
            fp_values.append(ppl)
        require(max(fp_values) - min(fp_values) <= config["evaluation"]["ppl_rerun_tolerance"], "FP16 rerun drift")
        reported = config["model"]["reported_fp16_wikitext2_ppl"]
        allowed = max(
            config["evaluation"]["fp16_source_tolerance_absolute"],
            config["evaluation"]["fp16_source_tolerance_relative"] * reported,
        )
        require(abs(sum(fp_values) / len(fp_values) - reported) <= allowed, "FP16 source-result drift")

        current = "quarot-fp32-equivalence"
        equivalence_path = output / "quarot-fp32-equivalence.json"
        equivalence_command = [
            str(environment / "bin" / "python"),
            "scripts/check_quarot_equivalence.py",
            str(config_path),
            str(equivalence_path),
        ]
        _, equivalence_elapsed = run_logged(current, equivalence_command, stage, env, logs)
        equivalence = json.loads(equivalence_path.read_text())

        for condition in config["conditions"][1:]:
            values = []
            condition_rows = []
            for rerun in range(1, config["evaluation"]["deterministic_reruns"] + 1):
                current = f"{condition['name']}-r{rerun}"
                text, elapsed = run_logged(current, condition_command(config, condition), stage / "fake_quant", env, logs)
                ppl = parse_ppl(text, current)
                transform = "QuaRot randomized Hadamard" if condition["rotate"] else "none"
                row = base_row(config, f"{attempt_id}-{current}", condition["name"], transform, gpu, stage, output)
                row["wikitext_ppl"] = ppl
                row["gpu_hours"] = elapsed / 3600.0
                if condition["rotate"]:
                    row["full_precision_equivalence_error"] = equivalence["relative_l2"]
                    row["notes_or_failure_reason"] = f"equivalence max_abs={equivalence['max_absolute']}"
                rows.append(row)
                condition_rows.append(row)
                values.append(ppl)
            require(max(values) - min(values) <= config["evaluation"]["ppl_rerun_tolerance"], f"{condition['name']} rerun drift")
        equivalence["gpu_hours"] = equivalence_elapsed / 3600.0
    except Exception as exc:
        failure = f"{current}: {type(exc).__name__}: {exc}"
        row = base_row(config, f"{attempt_id}-failure", current, "not_completed", gpu, stage, output)
        row["notes_or_failure_reason"] = failure
        rows.append(row)
    finally:
        write_results(output / "baseline_results.csv", rows)
        final = {
            "attempt_id": attempt_id,
            "completed_at_utc": utc_now(),
            "status": "FAILED" if failure else "PASS",
            "failure_reason": failure,
            "elapsed_gpu_hours": (time.monotonic() - started_all) / 3600.0,
            "result_rows": len(rows),
            "phase_1_opened": False,
        }
        write_new_json(output / "manifest.final.json", final)
    if failure:
        print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
