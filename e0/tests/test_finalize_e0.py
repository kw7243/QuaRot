#!/usr/bin/env python3
"""CPU-only tests for the E0 decision and checksum finalizer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FINALIZER = Path(__file__).resolve().parents[1] / "scripts" / "finalize_e0.py"


def summary(removable_fraction: float) -> dict[str, object]:
    return {
        "pairs": 40,
        "paired_median_removable_fraction": removable_fraction,
        "paired_median_removable_fraction_bootstrap_95": [
            removable_fraction - 0.005,
            removable_fraction + 0.005,
        ],
        "paired_median_speedup": 1.0 / (1.0 - removable_fraction),
    }


class FinalizeE0Test(unittest.TestCase):
    def test_primary_complete_path_selects_one_gate_decision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="absym-e0-finalize-") as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            contract = {
                "contract_id": "frozen-test-contract",
                "workloads": {
                    "primary": "decode_b1_ctx2048_step1",
                    "points": [
                        {"id": "decode_b1_ctx2048_step1"},
                        {"id": "decode_b16_ctx2048_step1"},
                        {"id": "prefill_b1_s2048"},
                    ],
                },
                "interpretation": {
                    "attractive_projected_end_to_end_improvement": 0.05,
                    "deprioritize_below_projected_end_to_end_improvement": 0.02,
                },
            }
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract))
            rows = []
            for index, point in enumerate(contract["workloads"]["points"]):
                short = (
                    {
                        "status": "PASS",
                        "latency_samples": 40,
                        "latency": summary(0.06 if index == 0 else 0.03),
                    }
                    if index < 2
                    else {"status": "NOT_RUN"}
                )
                rows.append(
                    {
                        "latency_workload": point,
                        "complete_mlp": {
                            "status": "PASS",
                            "latency_samples": 40,
                            "latency": summary(0.10),
                        },
                        "short_end_to_end": short,
                    }
                )
            (results / "results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (results / "raw_timings.json").write_text("{}\n")
            (results / "manifest.final.json").write_text("{}\n")
            preflight = root / "preflight.json"
            preflight.write_text('{"status":"PASS"}\n')
            stage = root / "REPRODUCIBILITY_METADATA.json"
            stage.write_text("{}\n")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(FINALIZER),
                    str(contract_path),
                    str(results),
                    str(preflight),
                    str(stage),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertEqual(completed.stdout.strip(), "ATTRACTIVE_HEADROOM_STOP_AFTER_E0")
            decision = json.loads((results / "decision.json").read_text())
            self.assertEqual(decision["gate_decision"], completed.stdout.strip())
            self.assertEqual(len(decision["projections"]), 3)
            self.assertEqual(len((results / "checksums.sha256").read_text().splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
