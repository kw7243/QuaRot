#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -Eeuo pipefail

# This script is submitted with --export=NIL. Define every non-Slurm variable
# after the directives; positional arguments are recorded in the attempt state.
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LC_ALL=C
LANG=C
research_root=/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries
scratch_root=/data/scratch-fast/kwen1/absorbable-transformer-symmetries
source_repo=${1:?source repository required}
staging_parent=${2:?fresh staging parent required}
attempt_dir=${3:?durable attempt directory required}
manifest=${4:?new durable manifest path required}
expected_commit=${5:?expected source commit required}
helper=${6:?copied stock staging helper required}
validator=$source_repo/e0/scripts/validate_stage.py
running_state=$attempt_dir/state.running.json
final_state=$attempt_dir/state.final.json
validation_report=$attempt_dir/stage-validation.json
content_delta=$attempt_dir/content-delta.txt
stage_path=
failure_command=unknown
tmp_root=/tmp/absym-e0-stage-${SLURM_JOB_ID:?SLURM_JOB_ID required}
export PATH LC_ALL LANG

write_failure() {
  local return_code=$1
  trap - EXIT
  if [[ ! -e "$final_state" ]]; then
    python3 - "$final_state" "$return_code" "$failure_command" "$stage_path" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, return_code, command, stage = sys.argv[1:]
value = {
    "schema_version": 1,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "FAILED",
    "exit_code": int(return_code),
    "failure_command": command,
    "stage_path": stage or None,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "experimental_command_executed": False,
    "gpu_requested": False,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
  fi
  rm -rf -- "$tmp_root"
  exit "$return_code"
}

trap 'failure_command=$BASH_COMMAND' ERR
trap 'write_failure $?' EXIT

[[ "${SLURM_EXPORT_ENV:-}" == "NIL" ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
[[ -z "${SLURM_GPUS:-}" ]]
[[ "$source_repo" == "$research_root/"* ]]
[[ "$staging_parent" == "$scratch_root/"* ]]
[[ "$attempt_dir" == "$research_root/"* ]]
[[ "$manifest" == "$research_root/"* ]]
[[ -d "$source_repo/.git" ]]
[[ -x "$helper" ]]
[[ -f "$validator" ]]
[[ ! -e "$manifest" ]]
[[ -d "$staging_parent" ]]
[[ -z "$(find "$staging_parent" -mindepth 1 -maxdepth 1 -print -quit)" ]]

mkdir -m 700 "$tmp_root"
TMPDIR=$tmp_root/tmp
XDG_CACHE_HOME=$tmp_root/xdg-cache
XDG_CONFIG_HOME=$tmp_root/xdg-config
PYTHONUSERBASE=$tmp_root/python-user
mkdir -m 700 "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$PYTHONUSERBASE"
export TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME PYTHONUSERBASE

job_descriptor=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$job_descriptor" != *"gres/gpu"* ]]
source_commit=$(git --no-optional-locks -C "$source_repo" rev-parse HEAD)
source_status=$(git --no-optional-locks -C "$source_repo" status --porcelain --untracked-files=all)
source_submodules=$(git --no-optional-locks -C "$source_repo" submodule status --recursive)
[[ "$source_commit" == "$expected_commit" ]]
[[ -z "$source_status" ]]
if grep -Eq '^[-+U]' <<<"$source_submodules"; then
  echo "source submodule state is incomplete or mismatched" >&2
  exit 3
fi

python3 - "$running_state" "$source_repo" "$staging_parent" "$manifest" "$source_commit" "$TMPDIR" "$job_descriptor" "$*" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, source, staging_parent, manifest, commit, tmpdir, descriptor, argv = sys.argv[1:]
value = {
    "schema_version": 1,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "RUNNING",
    "source_repo": source,
    "staging_parent": staging_parent,
    "expected_manifest": manifest,
    "source_commit": commit,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "slurm_job_descriptor": descriptor,
    "script_argv": argv,
    "tmpdir": tmpdir,
    "experimental_command_executed": False,
    "gpu_requested": False,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "state=RUNNING job_id=$SLURM_JOB_ID source=$source_repo target=$staging_parent"

rsync() {
  command rsync "$@" --no-perms --no-owner --no-group
}
export -f rsync

"$helper" --repo-root "$source_repo" --staging-parent "$staging_parent" --stage-only

mapfile -t stage_candidates < <(find "$staging_parent" -mindepth 1 -maxdepth 1 -type d -print)
[[ ${#stage_candidates[@]} -eq 1 ]]
stage_path=${stage_candidates[0]}
python3 "$validator" "$source_repo" "$stage_path" "$expected_commit" "$validation_report" "$content_delta"

metadata=$stage_path/REPRODUCIBILITY_METADATA.json
metadata_sha256=$(sha256sum "$metadata" | awk '{print $1}')
helper_sha256=$(sha256sum "$helper" | awk '{print $1}')
validator_sha256=$(sha256sum "$validator" | awk '{print $1}')
job_script_sha256=$(sha256sum "$0" | awk '{print $1}')
validation_sha256=$(sha256sum "$validation_report" | awk '{print $1}')

python3 - "$manifest" "$source_repo" "$stage_path" "$expected_commit" "$metadata" \
  "$metadata_sha256" "$helper" "$helper_sha256" "$validator" "$validator_sha256" \
  "$0" "$job_script_sha256" "$validation_report" "$validation_sha256" \
  "$content_delta" "$TMPDIR" "$job_descriptor" "$attempt_dir/stdout.log" "$attempt_dir/stderr.log" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    path, source, stage, commit, metadata, metadata_sha, helper, helper_sha,
    validator, validator_sha, job_script, job_script_sha, validation,
    validation_sha, delta, tmpdir, descriptor, stdout, stderr,
) = sys.argv[1:]
value = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "source": {"path": source, "commit": commit, "status": "clean"},
    "stage": {
        "path": stage,
        "reproducibility_metadata": metadata,
        "reproducibility_metadata_sha256": metadata_sha,
        "validation_report": validation,
        "validation_report_sha256": validation_sha,
        "content_delta": delta,
        "semantic_validation": "PASS",
    },
    "storage": {
        "RESEARCH_ROOT": "/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries",
        "SCRATCH_ROOT": "/data/scratch-fast/kwen1/absorbable-transformer-symmetries",
        "TMPDIR": tmpdir,
    },
    "executor": {
        "owner": "single v2 E0 staging event",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
        "gpu_requested": False,
        "job_descriptor": descriptor,
        "stdout": stdout,
        "stderr": stderr,
    },
    "code": {
        "stock_staging_helper": helper,
        "stock_staging_helper_sha256": helper_sha,
        "semantic_validator": validator,
        "semantic_validator_sha256": validator_sha,
        "job_script": job_script,
        "job_script_sha256": job_script_sha,
    },
    "phase": {"plan": "v2", "gate": "E0_STAGE_READY", "experiment_executed": False},
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

python3 - "$final_state" "$stage_path" "$manifest" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, stage, manifest = sys.argv[1:]
value = {
    "schema_version": 1,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "stage_path": stage,
    "manifest": manifest,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "experimental_command_executed": False,
    "gpu_requested": False,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "state=PASS job_id=$SLURM_JOB_ID stage=$stage_path manifest=$manifest"
rm -rf -- "$tmp_root"
trap - EXIT
