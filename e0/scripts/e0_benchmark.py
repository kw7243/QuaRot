#!/usr/bin/env python3
"""Paired v2 E0 timing for QuaRot's existing packed MLP path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty data")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median_interval(values: list[float], seed: int) -> list[float]:
    generator = random.Random(seed)
    medians = []
    for _ in range(4000):
        sample = [values[generator.randrange(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def summarize_pairs(reference: list[float], counterfactual: list[float], seed: int) -> dict[str, Any]:
    require(len(reference) == len(counterfactual) and len(reference) >= 30, "fewer than 30 timing pairs")
    require(
        all(math.isfinite(value) and value > 0 for value in reference + counterfactual),
        "timing samples must be finite and positive",
    )
    savings = [(r - c) / r for r, c in zip(reference, counterfactual)]
    speedups = [r / c for r, c in zip(reference, counterfactual)]
    return {
        "pairs": len(reference),
        "reference_median_ms": statistics.median(reference),
        "reference_mean_ms": statistics.fmean(reference),
        "reference_stdev_ms": statistics.stdev(reference),
        "counterfactual_median_ms": statistics.median(counterfactual),
        "counterfactual_mean_ms": statistics.fmean(counterfactual),
        "counterfactual_stdev_ms": statistics.stdev(counterfactual),
        "paired_median_removable_fraction": statistics.median(savings),
        "paired_mean_removable_fraction": statistics.fmean(savings),
        "paired_median_removable_fraction_bootstrap_95": bootstrap_median_interval(savings, seed),
        "paired_median_speedup": statistics.median(speedups),
        "paired_median_speedup_bootstrap_95": bootstrap_median_interval(speedups, seed + 1),
    }


@torch.inference_mode()
def elapsed_ms(function: Callable[[], Any], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = None
    for _ in range(iterations):
        result = function()
    end.record()
    end.synchronize()
    del result
    return float(start.elapsed_time(end))


def calibrated_iterations(
    set_mode: Callable[[str], None],
    function: Callable[[], Any],
    target_ms: float,
) -> tuple[int, dict[str, float]]:
    iterations = 1
    observations: dict[str, float] = {}
    while True:
        for mode in ("R", "C"):
            set_mode(mode)
            torch.cuda.synchronize()
            observations[mode] = elapsed_ms(function, iterations)
        if min(observations.values()) >= target_ms or iterations >= 1024:
            break
        iterations *= 2
    require(min(observations.values()) >= target_ms, "could not exceed target timing duration")
    return iterations, observations


def paired_measurement(
    set_mode: Callable[[str], None],
    function: Callable[[], Any],
    *,
    pairs: int,
    warmup_pairs: int,
    target_ms: float,
    seed: int,
) -> dict[str, Any]:
    iterations, calibration = calibrated_iterations(set_mode, function, target_ms)
    generator = random.Random(seed)
    for _ in range(warmup_pairs):
        order = ["R", "C"]
        generator.shuffle(order)
        for mode in order:
            set_mode(mode)
            elapsed_ms(function, iterations)
    reference: list[float] = []
    counterfactual: list[float] = []
    orders: list[list[str]] = []
    for _ in range(pairs):
        order = ["R", "C"]
        generator.shuffle(order)
        orders.append(order.copy())
        values: dict[str, float] = {}
        for mode in order:
            set_mode(mode)
            values[mode] = elapsed_ms(function, iterations) / iterations
        reference.append(values["R"])
        counterfactual.append(values["C"])
    return {
        "inner_iterations": iterations,
        "calibration_total_ms": calibration,
        "warmup_pairs": warmup_pairs,
        "orders": orders,
        "reference_ms": reference,
        "counterfactual_ms": counterfactual,
        "summary": summarize_pairs(reference, counterfactual, seed),
    }


def initialize_packed_modules(module: torch.nn.Module) -> None:
    import quarot

    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, quarot.nn.Linear4bit):
                child.weight_scales.fill_(2.0**-10)


class MlpTransformSwitch:
    def __init__(self, modules: list[torch.nn.Module]):
        self.entries = []
        for module in modules:
            down = module.down_proj
            require(isinstance(down, torch.nn.Sequential), "expected sequential packed down projection")
            hadamard = down[0]
            require(hadamard.__class__.__name__ == "OnlineHadamard", "reference Hadamard is absent")
            self.entries.append((down, hadamard, torch.nn.Identity()))

    def set_mode(self, mode: str) -> None:
        require(mode in ("R", "C"), f"unexpected mode {mode}")
        for down, hadamard, identity in self.entries:
            down[0] = hadamard if mode == "R" else identity


def build_mlp(config: dict[str, Any]) -> tuple[torch.nn.Module, MlpTransformSwitch]:
    from e2e.quantized_llama.modeling_llama import QuarotLlamaConfig, QuarotLlamaMLP

    model = config["model"]
    llama_config = QuarotLlamaConfig(
        hidden_size=model["mlp"]["hidden_size"],
        intermediate_size=model["mlp"]["intermediate_size"],
        hidden_act="silu",
        mlp_bias=False,
    )
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        mlp = QuarotLlamaMLP(llama_config)
    finally:
        torch.set_default_dtype(old_dtype)
    mlp = mlp.cuda().eval()
    initialize_packed_modules(mlp)
    return mlp, MlpTransformSwitch([mlp])


def benchmark_complete_mlp(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mlp, switch = build_mlp(config)
    timing = config["timing"]
    raw: dict[str, Any] = {}
    rows = []
    for index, workload in enumerate(config["workloads"]["points"]):
        tokens = workload["mlp_tokens"]
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260910 + index)
        inputs = torch.randn(
            (tokens, config["model"]["mlp"]["hidden_size"]),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        torch.cuda.reset_peak_memory_stats()
        measured = paired_measurement(
            switch.set_mode,
            lambda: mlp(inputs),
            pairs=timing["paired_samples"],
            warmup_pairs=timing["warmup_pairs"],
            target_ms=timing["minimum_target_duration_per_configuration_sample_ms"],
            seed=20260910 + index * 100,
        )
        peak = torch.cuda.max_memory_allocated()
        raw[workload["id"]] = measured
        rows.append(
            {
                "workload": workload,
                "status": "PASS",
                "scope": "complete_packed_mlp",
                "kernel_id": "QuaRot Linear4bit/Quantizer/OnlineHadamard",
                "extra_online_ops_R": ["generalized_hadamard_11008"],
                "extra_online_ops_C": [],
                "latency": measured["summary"],
                "latency_samples": timing["paired_samples"],
                "inner_iterations": measured["inner_iterations"],
                "peak_memory_bytes": peak,
            }
        )
        del inputs
        torch.cuda.empty_cache()
    switch.set_mode("R")
    del mlp
    gc.collect()
    torch.cuda.empty_cache()
    return rows, raw


def exact_llama2_7b_config(modeling: Any) -> Any:
    return modeling.QuarotLlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_implementation="flash_attention_2",
    )


def build_execution_model() -> tuple[torch.nn.Module, MlpTransformSwitch]:
    import transformers
    from e2e.quantized_llama import modeling_llama as modeling

    model_config = exact_llama2_7b_config(modeling)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with transformers.modeling_utils.no_init_weights():
            model = modeling.QuarotLlamaForCausalLM(config=model_config)
    finally:
        torch.set_default_dtype(old_dtype)
    with torch.no_grad():
        model.model.embed_tokens.weight.uniform_(-0.01, 0.01)
        model.model.norm.weight.fill_(1.0)
        model.lm_head.weight.zero_()
    initialize_packed_modules(model)
    model = model.cuda().eval()
    switch = MlpTransformSwitch([layer.mlp for layer in model.model.layers])
    return model, switch


def make_decode_function(
    model: torch.nn.Module,
    batch: int,
    context: int,
) -> tuple[Callable[[], Any], Any, torch.Tensor]:
    cache = model.build_cache(batch, page_size=context + 1, max_length=context + 1)
    cache.pages.zero_()
    cache.scales.zero_()
    cache._needs_init = [False] * len(model.model.layers)
    cache.length = context
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260910 + batch)
    input_ids = torch.randint(
        100,
        30000,
        (batch, 1),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )

    def run_decode() -> Any:
        cache.length = context
        return model(input_ids, past_key_values=cache, use_cache=True)

    return run_decode, cache, input_ids


def benchmark_short_end_to_end(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timing = config["timing"]
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    model = None
    try:
        model, switch = build_execution_model()
        for index, workload in enumerate(config["workloads"]["points"][:2]):
            batch = workload["batch"]
            context = workload["sequence_or_context"]
            try:
                function, cache, input_ids = make_decode_function(model, batch, context)
                torch.cuda.reset_peak_memory_stats()
                measured = paired_measurement(
                    switch.set_mode,
                    function,
                    pairs=timing["paired_samples"],
                    warmup_pairs=timing["warmup_pairs"],
                    target_ms=timing["minimum_target_duration_per_configuration_sample_ms"],
                    seed=20261910 + index * 100,
                )
                peak = torch.cuda.max_memory_allocated()
                raw[workload["id"]] = measured
                rows.append(
                    {
                        "workload": workload,
                        "status": "PASS",
                        "scope": "short_end_to_end_packed_decode",
                        "kernel_id": "QuaRotLlamaForCausalLM existing packed decode path",
                        "extra_online_ops_R": ["32x_generalized_hadamard_11008"],
                        "extra_online_ops_C": [],
                        "latency": measured["summary"],
                        "latency_samples": timing["paired_samples"],
                        "inner_iterations": measured["inner_iterations"],
                        "peak_memory_bytes": peak,
                    }
                )
                del function, cache, input_ids
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                rows.append(
                    {
                        "workload": workload,
                        "status": "SKIPPED_MEMORY_BOUNDARY",
                        "scope": "short_end_to_end_packed_decode",
                        "failure_reason": f"{type(error).__name__}: {error}",
                    }
                )
        rows.append(
            {
                "workload": config["workloads"]["points"][2],
                "status": "NOT_RUN",
                "scope": "short_end_to_end_packed_prefill",
                "failure_reason": "E0 short end-to-end baseline is decode-only; the complete MLP prefill point is measured above.",
            }
        )
        switch.set_mode("R")
    except Exception as error:
        rows = [
            {
                "workload": workload,
                "status": "IMPLEMENTATION_UNAVAILABLE",
                "scope": "short_end_to_end_packed_decode" if "decode" in workload["id"] else "short_end_to_end_packed_prefill",
                "failure_reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            for workload in config["workloads"]["points"]
        ]
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    output = arguments.output.resolve()
    contract = json.loads(contract_path.read_text())
    stage = Path(os.environ["RESEARCH_REPRO_STAGED_DIR"]).resolve()
    research_root = Path(contract["storage"]["research_root"]).resolve()
    scratch_root = Path(contract["storage"]["scratch_root"]).resolve()
    require(str(stage).startswith(str(scratch_root) + os.sep), "stage is outside SCRATCH_ROOT")
    require(str(output).startswith(str(research_root) + os.sep), "output is outside RESEARCH_ROOT")
    require(not output.exists() and not output.is_symlink(), "refusing to overwrite output")
    require((stage / "REPRODUCIBILITY_METADATA.json").is_file(), "stage metadata is absent")
    require(Path.cwd().resolve() == stage, "benchmark must run from stage root")
    require(os.environ.get("SLURM_JOB_ID"), "Slurm job is required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") and "," not in os.environ["CUDA_VISIBLE_DEVICES"], "exactly one visible GPU is required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one CUDA device is required")
    require(len(contract["operating_points"]) == 2, "contract must contain exactly R and C")
    require([point["id"] for point in contract["operating_points"]] == ["R", "C"], "unexpected operating points")
    require(contract["workloads"]["primary"] == "decode_b1_ctx2048_step1", "primary workload drift")
    require(contract["timing"]["paired_samples"] >= 30, "insufficient paired samples")

    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
        "experiment_id": "v2-e0-packed-mlp-headroom",
        "run_id": f"e0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.environ['SLURM_JOB_ID']}",
        "parent_run_id": None,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "model_id": contract["model"]["architecture_id"],
        "model_revision": contract["model"]["checkpoint_revision"],
        "tokenizer_revision": None,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment_id": os.environ.get("E0_ENVIRONMENT_ID"),
        "device": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(contract_path),
        "transform_id": "online_mlp_hadamard_present_vs_absent",
        "rounding_config": contract["quantization"],
        "calibration_ids": [],
        "validation_ids": [],
        "seed": 20260910,
        "optimization_steps": 0,
        "search_evaluations": 0,
        "float_equivalence_error": None,
        "local_nmse": None,
        "nll": None,
        "ppl": None,
        "stage_path": str(stage),
        "stage_metadata_sha256": sha256_file(stage / "REPRODUCIBILITY_METADATA.json"),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_job_descriptor": subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]], text=True).strip(),
        "tmpdir": os.environ["TMPDIR"],
        "quality_claim": False,
        "c_is_execution_counterfactual": True,
    }
    (output / "manifest.preflight.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    mlp_rows, mlp_raw = benchmark_complete_mlp(contract)
    e2e_rows, e2e_raw = benchmark_short_end_to_end(contract)
    rows = []
    for mlp_row in mlp_rows:
        workload_id = mlp_row["workload"]["id"]
        e2e_row = next(row for row in e2e_rows if row["workload"]["id"] == workload_id)
        rows.append(
            {
                "run_id": manifest["run_id"],
                "parent_run_id": None,
                "experiment_id": manifest["experiment_id"],
                "status": "PASS" if mlp_row["status"] == "PASS" else "FAILED",
                "model_id": manifest["model_id"],
                "model_revision": None,
                "tokenizer_revision": None,
                "code_commit": manifest["code_commit"],
                "environment_id": manifest["environment_id"],
                "device": manifest["device"],
                "device_count": 1,
                "contract_id": manifest["contract_id"],
                "transform_id": manifest["transform_id"],
                "rounding_config": "contract.json#quantization",
                "calibration_ids": [],
                "validation_ids": [],
                "seed": manifest["seed"],
                "optimization_steps": 0,
                "search_evaluations": 0,
                "float_equivalence_error": None,
                "local_nmse": None,
                "nll": None,
                "ppl": None,
                "kernel_id": mlp_row["kernel_id"],
                "extra_online_ops": {"R": mlp_row["extra_online_ops_R"], "C": []},
                "effective_bits": {"weights": 4, "activations": 4},
                "latency_workload": mlp_row["workload"],
                "latency_samples": mlp_row["latency_samples"],
                "complete_mlp": mlp_row,
                "short_end_to_end": e2e_row,
                "peak_memory": max(
                    mlp_row.get("peak_memory_bytes", 0),
                    e2e_row.get("peak_memory_bytes", 0),
                ),
                "elapsed_device_hours": None,
                "allocated_device_hours": None,
                "failure_reason": e2e_row.get("failure_reason"),
                "stage_path": str(stage),
            }
        )
    elapsed_hours = (time.monotonic() - started) / 3600.0
    for row in rows:
        row["elapsed_device_hours"] = elapsed_hours
    with (output / "results.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "raw_timings.json").write_text(
        json.dumps({"complete_mlp": mlp_raw, "short_end_to_end": e2e_raw}, indent=2, sort_keys=True) + "\n"
    )
    final = {
        **manifest,
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "elapsed_device_hours": elapsed_hours,
        "allocated_device_hours": None,
        "result_rows": len(rows),
        "complete_mlp_workloads_passed": sum(row["complete_mlp"]["status"] == "PASS" for row in rows),
        "short_end_to_end_workloads_passed": sum(row["short_end_to_end"]["status"] == "PASS" for row in rows),
        "phase_after_run": "E0_STOP",
        "phase_e1_opened": False,
    }
    (output / "manifest.final.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
