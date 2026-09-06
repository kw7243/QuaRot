#!/usr/bin/env python3
"""Full-model FP32 equivalence check for the exact QuaRot transform path."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
import transformers


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_quarot_equivalence.py CONFIG OUTPUT_JSON")
    config = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    require(not output.exists(), f"refusing to overwrite {output}")
    stage = Path(os.environ["RESEARCH_REPRO_STAGED_DIR"])
    sys.path.insert(0, str(stage / "fake_quant"))
    import hadamard_utils
    import model_utils
    import quant_utils
    import rotation_utils
    import utils

    transformers.set_seed(config["evaluation"]["seed"])
    os.environ["PHASE0B_TORCH_DTYPE"] = "float32"
    model = model_utils.get_model(config["model"]["id"], None)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["tokenizer_revision"],
        token=True,
        use_fast=False,
    )
    encoded = tokenizer(
        "QuaRot full precision invariance must hold before any quantization is enabled. " * 4,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    ).input_ids
    device = torch.device("cuda:0")
    require(torch.cuda.is_available(), "CUDA unavailable")
    model.to(device)
    with torch.inference_mode():
        reference = model(encoded.to(device), use_cache=False).logits.float().cpu()
    model.cpu()
    torch.cuda.empty_cache()

    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, type("Args", (), {"rotate_mode": "hadamard"})())
    quant_utils.add_actquant(model)
    qlayers = quant_utils.find_qlayers(model)
    for name, layer in qlayers.items():
        if "down_proj" in name:
            layer.had_K, layer.K = hadamard_utils.get_hadK(model.config.intermediate_size)
            layer.online_full_had = True
            layer.fp32_had = True
        if "o_proj" in name:
            layer.had_K, layer.K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            layer.online_partial_had = True
            layer.had_dim = model.config.hidden_size // model.config.num_attention_heads
            layer.fp32_had = True
    rope_name = model_utils.get_rope_function_name(model)
    for layer in model_utils.get_layers(model):
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            rope_name,
            config=model.config,
            k_bits=16,
            k_groupsize=128,
            k_sym=True,
            k_clip_ratio=1.0,
        )

    model.to(device)
    with torch.inference_mode():
        transformed = model(encoded.to(device), use_cache=False).logits.float().cpu()
    difference = transformed - reference
    relative_l2 = (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)).item()
    max_absolute = difference.abs().max().item()
    rel_tolerance = config["evaluation"]["quarot_equivalence_relative_l2_tolerance"]
    abs_tolerance = config["evaluation"]["quarot_equivalence_max_absolute_tolerance"]
    passed = math.isfinite(relative_l2) and relative_l2 <= rel_tolerance and max_absolute <= abs_tolerance
    result = {
        "relative_l2": relative_l2,
        "max_absolute": max_absolute,
        "relative_l2_tolerance": rel_tolerance,
        "max_absolute_tolerance": abs_tolerance,
        "input_tokens": int(encoded.numel()),
        "dtype": "float32",
        "passed": passed,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    require(passed, f"QuaRot FP32 equivalence failed: {result}")


if __name__ == "__main__":
    main()
