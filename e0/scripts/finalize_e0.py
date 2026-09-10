#!/usr/bin/env python3
"""Validate E0 timing, emit complete-path projections, checksums, and one gate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("preflight_manifest", type=Path)
    parser.add_argument("stage_metadata", type=Path)
    arguments = parser.parse_args()

    contract = json.loads(arguments.contract.read_text())
    rows_path = arguments.results / "results.jsonl"
    raw_path = arguments.results / "raw_timings.json"
    run_manifest_path = arguments.results / "manifest.final.json"
    decision_json_path = arguments.results / "decision.json"
    decision_md_path = arguments.results / "decision.md"
    checksums_path = arguments.results / "checksums.sha256"
    for path in (decision_json_path, decision_md_path, checksums_path):
        require(not path.exists(), f"refusing to overwrite {path}")

    rows = load_jsonl(rows_path)
    require(len(rows) == 3, "expected one row for each frozen E0 workload")
    require(
        [row["latency_workload"]["id"] for row in rows]
        == [point["id"] for point in contract["workloads"]["points"]],
        "result workloads differ from the frozen contract",
    )
    projections = []
    for row in rows:
        complete = row["complete_mlp"]
        require(complete["status"] == "PASS", "a complete packed MLP workload failed")
        require(complete["latency_samples"] >= 30, "fewer than 30 complete-path pairs")
        summary = complete["latency"]
        projection: dict[str, Any] = {
            "workload": row["latency_workload"]["id"],
            "complete_packed_mlp": {
                "paired_median_removable_fraction": summary[
                    "paired_median_removable_fraction"
                ],
                "bootstrap_95": summary[
                    "paired_median_removable_fraction_bootstrap_95"
                ],
                "paired_median_speedup": summary["paired_median_speedup"],
                "pairs": summary["pairs"],
            },
            "short_end_to_end": None,
        }
        short = row["short_end_to_end"]
        if short["status"] == "PASS":
            require(short["latency_samples"] >= 30, "fewer than 30 end-to-end pairs")
            short_summary = short["latency"]
            projection["short_end_to_end"] = {
                "paired_median_removable_fraction": short_summary[
                    "paired_median_removable_fraction"
                ],
                "bootstrap_95": short_summary[
                    "paired_median_removable_fraction_bootstrap_95"
                ],
                "paired_median_speedup": short_summary["paired_median_speedup"],
                "pairs": short_summary["pairs"],
            }
        projections.append(projection)

    primary_id = contract["workloads"]["primary"]
    primary = next(item for item in projections if item["workload"] == primary_id)
    require(
        primary["short_end_to_end"] is not None,
        "primary complete-path projection is unavailable",
    )
    primary_fraction = primary["short_end_to_end"][
        "paired_median_removable_fraction"
    ]
    attractive = contract["interpretation"][
        "attractive_projected_end_to_end_improvement"
    ]
    deprioritize = contract["interpretation"][
        "deprioritize_below_projected_end_to_end_improvement"
    ]
    if primary_fraction >= attractive:
        gate_decision = "ATTRACTIVE_HEADROOM_STOP_AFTER_E0"
        rationale = "primary projected end-to-end improvement is at least 5%"
    elif primary_fraction < deprioritize:
        gate_decision = "DEPRIORITIZE_OVERHEAD_REMOVAL_STOP_AFTER_E0"
        rationale = "primary projected end-to-end improvement is below 2%"
    else:
        gate_decision = "BORDERLINE_HEADROOM_STOP_AFTER_E0"
        rationale = "primary projected end-to-end improvement is between 2% and 5%"

    decision = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "gate": "v2_E0_execution_headroom",
        "gate_decision": gate_decision,
        "rationale": rationale,
        "primary_workload": primary_id,
        "primary_projected_end_to_end_removable_fraction": primary_fraction,
        "primary_projected_end_to_end_bootstrap_95": primary["short_end_to_end"][
            "bootstrap_95"
        ],
        "projections": projections,
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(arguments.contract),
        "preflight_manifest": str(arguments.preflight_manifest),
        "preflight_manifest_sha256": sha256_file(arguments.preflight_manifest),
        "stage_metadata": str(arguments.stage_metadata),
        "stage_metadata_sha256": sha256_file(arguments.stage_metadata),
        "quality_claim": False,
        "c_is_execution_counterfactual": True,
        "phase_after_decision": "E0_STOP",
        "phase_e1_opened": False,
        "scientific_fallback_invoked": False,
    }
    decision_json_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    decision_md_path.write_text(
        "# Absorbable Symmetry v2 E0 decision\n\n"
        f"Decision: **{gate_decision}**\n\n"
        f"The primary `{primary_id}` paired complete-path counterfactual projects "
        f"{primary_fraction:.3%} removable end-to-end time; the paired bootstrap "
        f"95% interval is [{primary['short_end_to_end']['bootstrap_95'][0]:.3%}, "
        f"{primary['short_end_to_end']['bootstrap_95'][1]:.3%}]. {rationale.capitalize()}.\n\n"
        "This is an execution-only headroom result. C is not an accurate checkpoint, "
        "no quality claim is made, no fallback was invoked, and E1 remains closed for this task.\n"
    )

    checksum_inputs = [
        arguments.contract,
        arguments.preflight_manifest,
        arguments.stage_metadata,
        rows_path,
        raw_path,
        run_manifest_path,
        decision_json_path,
        decision_md_path,
    ]
    checksums_path.write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in checksum_inputs)
    )
    print(gate_decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
