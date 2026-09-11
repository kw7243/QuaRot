#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -Eeuo pipefail

# This is a scheduler-tracked CPU-only environment/build proof. It is submitted
# with --export=NIL from a fresh validated stage. Compilation happens in an
# attempt-local copy so the stage remains immutable for the later GPU command.
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LC_ALL=C.UTF-8
LANG=C.UTF-8
research_root=/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries
scratch_root=/data/scratch-fast/kwen1/absorbable-transformer-symmetries
source_repo=${1:?validated immutable QuaRot stage required}
expected_commit=${2:?expected source commit required}
attempt_dir=${3:?durable preflight attempt directory required}
manifest=${4:?new durable preflight manifest required}
environment=${5:?fresh scratch environment required}
build_root=${6:?fresh scratch build root required}
environment_source=$scratch_root/envs/quarot-phase0b-py310-cu121-r1
build_source=$build_root/source
micromamba=/data/scratch-fast/kwen1/micromamba/bin/micromamba
mamba_state=$scratch_root/mamba-state/${environment##*/}
mamba_root=$mamba_state/root
mamba_pkgs=$mamba_state/pkgs
mamba_envs=$scratch_root/envs
running_state=$attempt_dir/state.running.json
final_state=$attempt_dir/state.final.json
failure_command=unknown
tmp_root=/tmp/absym-e0-preflight-${SLURM_JOB_ID:?SLURM_JOB_ID required}
export PATH LC_ALL LANG
MAMBA_ROOT_PREFIX=$mamba_root
CONDA_PKGS_DIRS=$mamba_pkgs
CONDA_ENVS_PATH=$mamba_envs
export MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS CONDA_ENVS_PATH

write_failure() {
  local return_code=$1
  trap - EXIT
  if [[ ! -e "$final_state" ]]; then
    python3 - "$final_state" "$return_code" "$failure_command" "$environment" \
      "$build_root" "$mamba_root" "$mamba_pkgs" "$mamba_envs" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    path, return_code, command, environment, build_root,
    mamba_root, mamba_pkgs, mamba_envs,
) = sys.argv[1:]
value = {
    "schema_version": 1,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "FAILED",
    "exit_code": int(return_code),
    "failure_command": command,
    "environment": environment,
    "build_root": build_root,
    "mamba_state": {
        "root_prefix": mamba_root,
        "package_cache": mamba_pkgs,
        "envs_directory": mamba_envs,
    },
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "gpu_requested": False,
    "experiment_executed": False,
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
[[ -z "${SLURM_JOB_GPUS:-}" ]]
[[ -z "${SLURM_GPUS:-}" ]]
[[ "$source_repo" == "$scratch_root/staging/"* ]]
[[ "$attempt_dir" == "$research_root/preflight-attempts/"* ]]
[[ "$manifest" == "$research_root/manifests/"* ]]
[[ "$environment" == "$scratch_root/envs/"* ]]
[[ "$build_root" == "$scratch_root/preflight-builds/"* ]]
[[ -d "$source_repo/.git" ]]
[[ -f "$source_repo/REPRODUCIBILITY_METADATA.json" ]]
[[ -d "$attempt_dir" ]]
[[ ! -e "$manifest" ]]
[[ ! -e "$environment" ]]
[[ ! -e "$build_root" ]]
[[ ! -e "$mamba_state" ]]
[[ -x "$micromamba" ]]
[[ -d "$environment_source" ]]

mkdir -m 700 -p "$mamba_root" "$mamba_pkgs"
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
source_tree=$(git --no-optional-locks -C "$source_repo" rev-parse 'HEAD^{tree}')
source_status=$(git --no-optional-locks -C "$source_repo" status --porcelain --untracked-files=all)
source_submodules=$(git --no-optional-locks -C "$source_repo" submodule status --recursive)
[[ "$source_commit" == "$expected_commit" ]]
[[ -z "$source_status" || "$source_status" == "?? REPRODUCIBILITY_METADATA.json" ]]
[[ "$(pwd -P)" == "$(realpath "$source_repo")" ]]
if grep -Eq '^[-+U]' <<<"$source_submodules"; then
  echo "source submodule state is incomplete or mismatched" >&2
  exit 3
fi

python3 - "$running_state" "$source_repo" "$source_commit" "$source_tree" \
  "$attempt_dir" "$manifest" "$environment" "$build_root" "$mamba_root" \
  "$mamba_pkgs" "$mamba_envs" "$TMPDIR" "$job_descriptor" "$*" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    path, source, commit, tree, attempt, manifest, environment, build_root,
    mamba_root, mamba_pkgs, mamba_envs, tmpdir, descriptor, argv,
) = sys.argv[1:]
value = {
    "schema_version": 1,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "RUNNING",
    "source": {
        "path": source,
        "commit": commit,
        "tree": tree,
        "stage_status_except_metadata": "clean",
    },
    "attempt_dir": attempt,
    "expected_manifest": manifest,
    "environment": environment,
    "build_root": build_root,
    "mamba_state": {
        "root_prefix": mamba_root,
        "package_cache": mamba_pkgs,
        "envs_directory": mamba_envs,
    },
    "tmpdir": tmpdir,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "slurm_job_descriptor": descriptor,
    "script_argv": argv,
    "gpu_requested": False,
    "experiment_executed": False,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "state=RUNNING job_id=$SLURM_JOB_ID stage=$source_repo"

mkdir -p "$build_source"
rsync -a --no-perms --no-owner --no-group \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.cache' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='*.egg-info' \
  "$source_repo/" "$build_source/"

rsync -rlnic \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.cache' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='*.egg-info' \
  "$source_repo/" "$build_source/" > "$attempt_dir/source-content-before-build.delta"
[[ ! -s "$attempt_dir/source-content-before-build.delta" ]]

"$micromamba" --no-rc --root-prefix "$mamba_root" info --json \
  > "$attempt_dir/micromamba-info-before-create.json"
"$environment_source/bin/python" - "$attempt_dir/micromamba-info-before-create.json" \
  "$mamba_pkgs" "$mamba_envs" <<'PY'
import json
import sys
from pathlib import Path

path, expected_pkgs, expected_envs = sys.argv[1:]
value = json.loads(Path(path).read_text())
assert value["package cache"] == [expected_pkgs], value["package cache"]
assert value["envs directories"][0] == expected_envs, value["envs directories"]
assert all(not item.startswith("/afs/") for item in value["package cache"])
assert all(not item.startswith("/afs/") for item in value["envs directories"])
assert value.get("populated config files", []) == []
PY

"$micromamba" --no-rc --root-prefix "$mamba_root" create \
  -y -p "$environment" --clone "$environment_source"
"$micromamba" --no-rc --root-prefix "$mamba_root" install \
  -y -p "$environment" --override-channels \
  -c nvidia/label/cuda-12.1.1 -c conda-forge \
  cuda-nvcc=12.1.105=0 \
  cuda-cudart=12.1.105=0 \
  cuda-cudart-dev=12.1.105=0 \
  cuda-cudart-static=12.1.105=0 \
  cuda-cccl=12.1.109=0

PATH=$environment/bin:$PATH
CUDA_HOME=$environment
cuda_target_lib=$environment/targets/x86_64-linux/lib
torch_cuda_lib=$environment/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
LD_LIBRARY_PATH=$environment/lib:$cuda_target_lib:$torch_cuda_lib:${LD_LIBRARY_PATH:-}
PIP_CACHE_DIR=$scratch_root/cache/e0/pip
TORCH_EXTENSIONS_DIR=$scratch_root/cache/e0/torch_extensions
TORCH_CUDA_ARCH_LIST=8.6
MAX_JOBS=8
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
export PATH CUDA_HOME LD_LIBRARY_PATH PIP_CACHE_DIR TORCH_EXTENSIONS_DIR
export TORCH_CUDA_ARCH_LIST MAX_JOBS FAST_HADAMARD_TRANSFORM_FORCE_BUILD
mkdir -p "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

"$environment/bin/python" -m pip install \
  --disable-pip-version-check \
  -r "$build_source/requirements-phase0b.txt" \
  > "$attempt_dir/python-requirements.log" 2>&1
"$environment/bin/python" -m pip install \
  --disable-pip-version-check \
  ninja==1.11.1.4 packaging==24.1 \
  > "$attempt_dir/python-build-requirements.log" 2>&1

"$micromamba" --no-rc --root-prefix "$mamba_root" list \
  -p "$environment" --json > "$attempt_dir/conda-packages.json"
"$environment/bin/python" - "$attempt_dir/conda-packages.json" <<'PY'
import json
import sys

packages = {item["name"]: str(item["version"]) for item in json.load(open(sys.argv[1]))}
expected = {
    "cuda-cccl": "12.1.109",
    "cuda-cudart": "12.1.105",
    "cuda-cudart-dev": "12.1.105",
    "cuda-cudart-static": "12.1.105",
    "cuda-nvcc": "12.1.105",
}
for name, version in expected.items():
    assert packages.get(name) == version, (name, packages.get(name), version)
for name, version in packages.items():
    if name.startswith("cuda-"):
        assert not version.startswith("13."), (name, version)
PY

"$environment/bin/nvcc" --version > "$attempt_dir/nvcc-version.txt"
grep -Eq 'release 12\.1, V12\.1\.105' "$attempt_dir/nvcc-version.txt"

dynamic_cudart=$environment/targets/x86_64-linux/lib/libcudart.so.12.1.105
static_cudart=$environment/targets/x86_64-linux/lib/libcudart_static.a
device_runtime=$environment/targets/x86_64-linux/lib/libcudadevrt.a
[[ -f "$dynamic_cudart" ]]
[[ -f "$static_cudart" ]]
[[ -f "$device_runtime" ]]
[[ "$(readlink -f "$environment/lib/libcudart.so")" == "$dynamic_cudart" ]]

cat > "$TMPDIR/cuda-link-smoke.cu" <<'CUDA'
#include <cuda_runtime.h>
int main() { return 0; }
CUDA
"$environment/bin/nvcc" --cudart shared "$TMPDIR/cuda-link-smoke.cu" \
  -o "$attempt_dir/cuda-link-shared"
"$environment/bin/nvcc" --cudart static "$TMPDIR/cuda-link-smoke.cu" \
  -o "$attempt_dir/cuda-link-static"
ldd "$attempt_dir/cuda-link-shared" > "$attempt_dir/cuda-link-shared.ldd"
grep -F "$environment/lib/libcudart.so.12" "$attempt_dir/cuda-link-shared.ldd"
ldd "$attempt_dir/cuda-link-static" > "$attempt_dir/cuda-link-static.ldd"
if grep -Fq 'libcudart' "$attempt_dir/cuda-link-static.ldd"; then
  echo "static CUDA smoke unexpectedly depends on dynamic libcudart" >&2
  exit 1
fi

cd "$build_source"
"$environment/bin/python" setup.py --name \
  > "$attempt_dir/quarot-metadata.log" 2>&1
grep -Fx 'quarot' "$attempt_dir/quarot-metadata.log"
"$environment/bin/python" setup.py build_ext --inplace \
  > "$attempt_dir/quarot-build.log" 2>&1
"$environment/bin/python" -m pip install \
  --disable-pip-version-check --no-build-isolation --no-deps flash-attn==2.6.3 \
  > "$attempt_dir/flash-attn-build.log" 2>&1

mapfile -t quarot_extensions < <(find "$build_source/quarot" -maxdepth 1 -type f -name '_CUDA*.so' -print)
mapfile -t hadamard_extensions < <(find "$build_source/third-party/fast-hadamard-transform" \
  -maxdepth 1 -type f -name 'fast_hadamard_transform_cuda*.so' -print)
mapfile -t flash_attention_extensions < <(find "$environment/lib/python3.10/site-packages" \
  -maxdepth 1 -type f -name 'flash_attn_2_cuda*.so' -print)
[[ ${#quarot_extensions[@]} -eq 1 ]]
[[ ${#hadamard_extensions[@]} -eq 1 ]]
[[ ${#flash_attention_extensions[@]} -eq 1 ]]

"$environment/bin/python" -m pip freeze > "$attempt_dir/pip-freeze.txt"

PYTHONPATH=$build_source:$build_source/third-party/fast-hadamard-transform
export PYTHONPATH
"$environment/bin/python" - <<'PY' > "$attempt_dir/python-imports.txt"
import flash_attn
import fast_hadamard_transform
import fast_hadamard_transform_cuda
import quarot
import quarot._CUDA
import torch
import transformers

assert torch.__version__ == "2.4.1+cu121", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("quarot_extension", quarot._CUDA.__file__)
print("fast_hadamard_extension", fast_hadamard_transform_cuda.__file__)
print("flash_attn", flash_attn.__version__)
PY

ldd "${quarot_extensions[0]}" > "$attempt_dir/quarot-extension.ldd"
if grep -Fq 'not found' "$attempt_dir/quarot-extension.ldd"; then
  cat "$attempt_dir/quarot-extension.ldd" >&2
  exit 1
fi
ldd "${hadamard_extensions[0]}" > "$attempt_dir/hadamard-extension.ldd"
if grep -Fq 'not found' "$attempt_dir/hadamard-extension.ldd"; then
  cat "$attempt_dir/hadamard-extension.ldd" >&2
  exit 1
fi
ldd "${flash_attention_extensions[0]}" > "$attempt_dir/flash-attention-extension.ldd"
if grep -Fq 'not found' "$attempt_dir/flash-attention-extension.ldd"; then
  cat "$attempt_dir/flash-attention-extension.ldd" >&2
  exit 1
fi

python3 - "$manifest" "$source_repo" "$source_commit" "$source_tree" \
  "$build_source" "$environment" "$attempt_dir" "$dynamic_cudart" \
  "$static_cudart" "$device_runtime" "${quarot_extensions[0]}" \
  "${hadamard_extensions[0]}" "${flash_attention_extensions[0]}" \
  "$job_descriptor" "$0" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    path, source, commit, tree, build_source, environment, attempt,
    dynamic_cudart, static_cudart, device_runtime, quarot_extension,
    hadamard_extension, flash_attention_extension, descriptor, script,
) = sys.argv[1:]

def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = {
    "nvcc": str(Path(environment) / "bin/nvcc"),
    "dynamic_cudart": dynamic_cudart,
    "static_cudart": static_cudart,
    "device_runtime": device_runtime,
    "quarot_extension": quarot_extension,
    "hadamard_extension": hadamard_extension,
    "flash_attention_extension": flash_attention_extension,
    "conda_packages": str(Path(attempt) / "conda-packages.json"),
    "micromamba_info": str(Path(attempt) / "micromamba-info-before-create.json"),
    "pip_freeze": str(Path(attempt) / "pip-freeze.txt"),
    "python_imports": str(Path(attempt) / "python-imports.txt"),
}
value = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "source": {
        "path": source,
        "commit": commit,
        "tree": tree,
        "stage_status_except_metadata": "clean",
    },
    "build_source": build_source,
    "environment": environment,
    "cuda": {
        "release": "12.1",
        "nvcc": "12.1.105",
        "cudart": "12.1.105",
        "cudart_dev": "12.1.105",
        "cudart_static": "12.1.105",
        "cccl": "12.1.109",
    },
    "python": {"version": "3.10", "torch": "2.4.1+cu121", "torch_cuda": "12.1"},
    "proofs": {
        "dynamic_cuda_link": "PASS",
        "static_cuda_link": "PASS",
        "quarot_metadata": "PASS",
        "quarot_extension_build": "PASS",
        "fast_hadamard_extension_build": "PASS",
        "flash_attention_install": "PASS",
        "imports_without_gpu": "PASS",
    },
    "artifacts": {name: {"path": filename, "sha256": sha256(filename)} for name, filename in files.items()},
    "executor": {
        "owner": "single v2 E0 CUDA build preflight",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
        "gpu_requested": False,
        "job_descriptor": descriptor,
        "job_script": script,
        "job_script_sha256": sha256(script),
    },
    "experiment_executed": False,
    "phase_after_run": "E0_PREFLIGHT_PASS",
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

sha256sum "$manifest" > "$attempt_dir/manifest.sha256"
python3 - "$final_state" "$manifest" "$environment" "$build_source" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, manifest, environment, build_source = sys.argv[1:]
value = {
    "schema_version": 1,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "manifest": manifest,
    "environment": environment,
    "build_source": build_source,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_export_mode": os.environ.get("SLURM_EXPORT_ENV"),
    "gpu_requested": False,
    "experiment_executed": False,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

echo "state=PASS job_id=$SLURM_JOB_ID manifest=$manifest environment=$environment"
trap - EXIT
