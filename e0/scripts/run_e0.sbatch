#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --no-requeue

set -Eeuo pipefail

# Submitted with --export=NIL. All non-Slurm environment and storage paths are
# defined here after the directives; the stage/output are explicit argv.
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LC_ALL=C.UTF-8
LANG=C.UTF-8
research_root=/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries
scratch_root=/data/scratch-fast/kwen1/absorbable-transformer-symmetries
stage=${1:?validated stage required}
attempt_dir=${2:?durable attempt directory required}
contract=${3:?staged contract required}
preflight_manifest=${4:?successful CPU build preflight manifest required}
results_dir=$attempt_dir/results
cache_root=$scratch_root/cache/e0
running_state=$attempt_dir/state.running.json
final_state=$attempt_dir/state.final.json
tmp_root=${SLURM_TMPDIR:-/tmp/absym-e0-${SLURM_JOB_ID:?SLURM_JOB_ID required}}
failure_command=unknown
experiment_started=false
export PATH LC_ALL LANG

write_failure() {
  local return_code=$1
  trap - EXIT
  if [[ ! -e "$final_state" ]]; then
    python3 - "$final_state" "$return_code" "$failure_command" "$experiment_started" "$results_dir" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, return_code, command, experiment_started, results = sys.argv[1:]
value = {
    "schema_version": 1,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "FAILED",
    "exit_code": int(return_code),
    "failure_command": command,
    "experiment_started": experiment_started == "true",
    "results_dir": results,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "phase_after_run": "E0_STOP",
    "phase_e1_opened": False,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
  fi
  exit "$return_code"
}

trap 'failure_command=$BASH_COMMAND' ERR
trap 'write_failure $?' EXIT

[[ "${SLURM_EXPORT_ENV:-}" == "NIL" ]]
[[ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS:-}}" ]]
[[ "$stage" == "$scratch_root/staging/"* ]]
[[ "$attempt_dir" == "$research_root/"* ]]
[[ "$contract" == "$stage/"* ]]
[[ -d "$stage/.git" ]]
[[ -f "$stage/REPRODUCIBILITY_METADATA.json" ]]
[[ -f "$contract" ]]
[[ -f "$stage/e0/scripts/e0_benchmark.py" ]]
[[ -f "$stage/e0/scripts/finalize_e0.py" ]]
[[ -f "$stage/e0/scripts/verify_cuda_preflight.py" ]]
[[ ! -e "$results_dir" ]]
[[ "$preflight_manifest" == "$research_root/manifests/"* ]]
[[ -f "$preflight_manifest" ]]

mkdir -p "$tmp_root" "$cache_root/huggingface" "$cache_root/pip" "$cache_root/torch_extensions"
TMPDIR=$tmp_root/tmp
XDG_CACHE_HOME=$tmp_root/xdg-cache
XDG_CONFIG_HOME=$tmp_root/xdg-config
PYTHONUSERBASE=$tmp_root/python-user
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$PYTHONUSERBASE"
export TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME PYTHONUSERBASE

preflight_values=$(python3 "$stage/e0/scripts/verify_cuda_preflight.py" \
  "$preflight_manifest" "$stage" "$attempt_dir/preflight-verification.json")
IFS='|' read -r environment build_source preflight_manifest_sha256 <<<"$preflight_values"
[[ -x "$environment/bin/python" ]]
[[ -x "$environment/bin/nvcc" ]]
[[ -d "$build_source" ]]

job_descriptor=$(scontrol show job -o "$SLURM_JOB_ID")
gpu_inventory=$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader)
visible_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$visible_count" -eq 1 ]]

python3 - "$running_state" "$stage" "$attempt_dir" "$contract" "$environment" \
  "$build_source" "$preflight_manifest" "$preflight_manifest_sha256" "$TMPDIR" \
  "$job_descriptor" "$gpu_inventory" "$*" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    path, stage, attempt, contract, environment, build_source, preflight_manifest,
    preflight_manifest_sha256, tmpdir, descriptor, gpu, argv,
) = sys.argv[1:]
value = {
    "schema_version": 1,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "RUNNING",
    "stage": stage,
    "attempt_dir": attempt,
    "contract": contract,
    "environment": environment,
    "build_source": build_source,
    "cuda_build_preflight_manifest": preflight_manifest,
    "cuda_build_preflight_manifest_sha256": preflight_manifest_sha256,
    "tmpdir": tmpdir,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "slurm_job_descriptor": descriptor,
    "gpu_inventory": gpu,
    "script_argv": argv,
    "single_event_owner": True,
    "phase": "v2 E0",
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "state=RUNNING job_id=$SLURM_JOB_ID stage=$stage"
echo "gpu=$gpu_inventory"

PATH=$environment/bin:$PATH
export PATH
HF_HOME=$cache_root/huggingface
HF_HUB_CACHE=$cache_root/huggingface/hub
PIP_CACHE_DIR=$cache_root/pip
TORCH_EXTENSIONS_DIR=$cache_root/torch_extensions
TOKENIZERS_PARALLELISM=false
PYTHONHASHSEED=20260910
TORCH_CUDA_ARCH_LIST=8.6
MAX_JOBS=8
CUDA_HOME=$environment
cuda_target_lib=$environment/targets/x86_64-linux/lib
torch_cuda_lib=$environment/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
LD_LIBRARY_PATH=$environment/lib:$cuda_target_lib:$torch_cuda_lib:${LD_LIBRARY_PATH:-}
PYTHONPATH=$build_source:$build_source/third-party/fast-hadamard-transform
export HF_HOME HF_HUB_CACHE PIP_CACHE_DIR TORCH_EXTENSIONS_DIR TOKENIZERS_PARALLELISM
export PYTHONHASHSEED TORCH_CUDA_ARCH_LIST MAX_JOBS CUDA_HOME LD_LIBRARY_PATH PYTHONPATH

cd "$stage"
flash_attention_status=preflight_verified
export E0_FLASH_ATTENTION_STATUS=$flash_attention_status

"$environment/bin/python" -m pip freeze > "$attempt_dir/environment-freeze.txt"
"$environment/bin/python" - <<'PY' > "$attempt_dir/software-preflight.txt"
import os
import torch
import transformers
import quarot
import fast_hadamard_transform

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("flash_attention_status", os.environ["E0_FLASH_ATTENTION_STATUS"])
PY

export RESEARCH_REPRO_STAGED_DIR=$stage
export RESEARCH_REPRO_SOURCE_REPO=/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/baseline-src
export E0_ENVIRONMENT_ID=quarot-e0-py310-cu121-r3
experiment_started=true
"$environment/bin/python" "$stage/e0/scripts/e0_benchmark.py" "$contract" "$results_dir"
gate_decision=$("$environment/bin/python" "$stage/e0/scripts/finalize_e0.py" \
  "$contract" "$results_dir" "$preflight_manifest" \
  "$stage/REPRODUCIBILITY_METADATA.json")

python3 - "$final_state" "$results_dir" "$environment" "$flash_attention_status" \
  "$gate_decision" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, results, environment, flash_status, gate_decision = sys.argv[1:]
value = {
    "schema_version": 1,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "results_dir": results,
    "environment": environment,
    "flash_attention_status": flash_status,
    "gate_decision": gate_decision,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "phase_after_run": "E0_STOP",
    "phase_e1_opened": False,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "state=PASS job_id=$SLURM_JOB_ID results=$results_dir"
trap - EXIT
