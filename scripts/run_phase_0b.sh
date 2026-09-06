#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG DURABLE_OUTPUT_DIR" >&2
  exit 2
fi

config=$1
output_dir=$2
repo_root=$(git rev-parse --show-toplevel)

if [[ -z "${RESEARCH_REPRO_STAGED_DIR:-}" || "$repo_root" != "$RESEARCH_REPRO_STAGED_DIR" ]]; then
  echo "refusing to run outside the reproducibility stage" >&2
  exit 2
fi
if [[ ! -f "$repo_root/REPRODUCIBILITY_METADATA.json" ]]; then
  echo "missing REPRODUCIBILITY_METADATA.json" >&2
  exit 2
fi
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
  echo "refusing to overwrite output: $output_dir" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Phase 0B evaluation must run inside one Slurm job" >&2
  exit 2
fi

environment=$(python3 - "$config" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["storage"]["environment"])
PY
)
if [[ ! -x "$environment/bin/python" ]]; then
  echo "missing pinned Phase 0B environment: $environment" >&2
  exit 2
fi

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR=$SLURM_TMPDIR
else
  export TMPDIR
  TMPDIR=$(mktemp -d -p /tmp "phase0b-${SLURM_JOB_ID}-XXXXXXXX")
  trap 'rm -rf -- "$TMPDIR"' EXIT
fi

exec "$environment/bin/python" scripts/phase0b_driver.py "$config" "$output_dir"
