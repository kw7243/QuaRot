#!/usr/bin/env python3
"""Fail-closed source/config audit for the Phase 0B QuaRot baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
EXPECTED_MODEL = "meta-llama/Llama-3.2-3B"
EXPECTED_MODEL_REVISION = "13afe5124825b4f3751f836b40dafda64c1ed062"
EXPECTED_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_anchors(path: Path, anchors: list[str]) -> None:
    text = path.read_text()
    missing = [anchor for anchor in anchors if anchor not in text]
    _require(not missing, f"{path}: missing source anchors {missing}")


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    _require(config["source"]["commit"] == EXPECTED_COMMIT, "unexpected source commit")
    _require(config["model"]["id"] == EXPECTED_MODEL, "silent model substitution")
    _require(
        config["model"]["revision"] == EXPECTED_MODEL_REVISION,
        "unexpected model revision",
    )
    _require(
        config["model"]["tokenizer_revision"] == EXPECTED_MODEL_REVISION,
        "tokenizer revision must be pinned to the model revision",
    )
    _require(
        config["dataset"]["revision"] == EXPECTED_DATASET_REVISION,
        "unexpected evaluation dataset revision",
    )
    _require(
        config["calibration"]["dataset_revision"] == EXPECTED_DATASET_REVISION,
        "evaluation/calibration dataset revision mismatch",
    )
    _require(config["model"]["sequence_length"] == 2048, "sequence length drift")
    _require(config["calibration"]["samples"] == 128, "calibration-size drift")
    _require(config["calibration"]["sequence_length"] == 2048, "calibration length drift")
    _require(config["evaluation"]["deterministic_reruns"] >= 2, "need two reruns")
    quant = config["quantization"]
    _require(quant["weight_method"] == "RTN", "Phase 0B baseline must use RTN")
    _require(
        [quant[key] for key in ("weight_bits", "activation_bits", "key_bits", "value_bits")]
        == [4, 4, 4, 4],
        "baseline is not W4A4KV4",
    )
    _require(
        [
            quant[key]
            for key in (
                "weight_group_size",
                "activation_group_size",
                "key_group_size",
                "value_group_size",
            )
        ]
        == [128, 128, 128, 128],
        "baseline group definitions are not matched g128",
    )
    _require(quant["symmetric"] is True, "baseline must use matched symmetric quantization")
    _require(quant["weight_clipping"] is False, "naive RTN must not tune clipping")
    _require(
        all(quant[key] == 1.0 for key in ("activation_clip_ratio", "key_clip_ratio", "value_clip_ratio")),
        "activation/K/V clipping must be disabled",
    )
    names = [condition["name"] for condition in config["conditions"]]
    _require(
        names == ["fp16", "rtn_w4a4kv4_g128", "quarot_w4a4kv4_g128"],
        "condition set/order drift",
    )
    storage = config["storage"]
    _require(
        storage["research_root"] == "/data/vision/torralba/u/kwen1",
        "retained-data root drift",
    )
    _require(
        storage["scratch_root"] == "/data/scratch-fast/kwen1",
        "scratch root drift",
    )
    _require(storage["environment"].startswith(storage["scratch_root"] + "/"), "environment outside scratch")
    _require(storage["hf_home"].startswith(storage["scratch_root"] + "/"), "HF cache outside scratch")
    return config


def audit_source(source_root: Path, require_clean: bool) -> dict[str, Any]:
    head = _git(source_root, "rev-parse", "HEAD")
    status = _git(source_root, "status", "--porcelain", "--untracked-files=no")
    base_is_ancestor = subprocess.run(
        ["git", "-C", str(source_root), "merge-base", "--is-ancestor", EXPECTED_COMMIT, head],
        check=False,
    ).returncode == 0
    _require(base_is_ancestor, "pinned QuaRot commit is not an ancestor of the adapter")
    if require_clean:
        _require(status == "", "source has tracked changes")

    _require_anchors(
        source_root / "fake_quant" / "main.py",
        [
            "rotation_utils.fuse_layer_norms(model)",
            "gptq_utils.rtn_fwrd(model, utils.DEV, args)",
            "quant_utils.add_actquant(model)",
            "add_qk_rotation_wrapper_after_function_call_in_forward",
        ],
    )
    _require_anchors(
        source_root / "fake_quant" / "quant_utils.py",
        [
            "self.quantizer.find_params(x)",
            "self.out_quantizer.find_params(x)",
            "find_params_per_token_groupwise",
        ],
    )
    _require_anchors(
        source_root / "fake_quant" / "gptq_utils.py",
        ["def rtn_fwrd", "grouped = W.reshape(-1, args.w_groupsize)"],
    )
    _require_anchors(
        source_root / "fake_quant" / "data_utils.py",
        ["Salesforce/wikitext", "revision=dataset_revision", "wikitext-2-raw-v1"],
    )
    return {
        "adapter_head": head,
        "pinned_quarot_base": EXPECTED_COMMIT,
        "pinned_base_is_ancestor": base_is_ancestor,
        "tracked_status": status,
    }


def quantization_site_audit() -> dict[str, Any]:
    linear_sites = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    return {
        "weight": {
            "sites_per_decoder_layer": linear_sites,
            "bits": 4,
            "method": "RTN fake quantization",
            "axis": "contiguous groups on each output row's input-feature axis",
            "group_size": 128,
            "range": "dynamic per weight-row group, symmetric [-8, 7], no clipping",
        },
        "activation": {
            "linear_input_sites_per_decoder_layer": linear_sites,
            "bits": 4,
            "axis": "last feature dimension, independently for every token",
            "group_size": 128,
            "range": "dynamic per token/group, symmetric [-8, 7], clip ratio 1.0",
            "exclusion": "lm_head input remains 16-bit",
        },
        "value_cache": {
            "site": "output of self_attn.v_proj before attention",
            "bits": 4,
            "axis": "last projected KV feature dimension",
            "group_size": 128,
            "range": "dynamic per token/group, symmetric [-8, 7], clip ratio 1.0",
        },
        "key_cache": {
            "site": "K after RoPE and the matched Q/K Hadamard",
            "bits": 4,
            "axis": "each 128-wide KV head",
            "group_size": 128,
            "range": "dynamic per token/head, symmetric [-8, 7], clip ratio 1.0",
        },
        "fp16_exclusions": ["model.embed_tokens", "model.norm", "lm_head"],
        "baseline_runtime_note": "QuaRot's online Hadamards are baseline operations, not Candidate A operations.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    config = load_and_validate_config(args.config.resolve())
    report = {
        "schema_version": 1,
        "source": audit_source(args.source_root.resolve(), args.require_clean),
        "model": config["model"],
        "dataset": config["dataset"],
        "calibration": config["calibration"],
        "quantization_site_audit": quantization_site_audit(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        _require(not args.output.exists(), f"refusing to overwrite {args.output}")
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
