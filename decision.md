# Current decision

State: **STOPPED — CORRECTED CPU PREFLIGHT FAILED**

The 2026-09-16 authorization resumed the preserved immutable stage at source commit `4bc8433528cec54e6b0c32ce94bd755f80303e33`. Corrected CPU preflight job `1977465` used that stage as `--chdir`, passed the stage guard, and reached fresh environment creation.

Micromamba 2.8.1 then failed while cloning the environment. Although the command used `--no-rc` and all configured root, package, and environment paths were scratch-local, libmamba tried to update `/afs/csail.mit.edu/u/k/kwen1/.conda/environments.txt` and received permission denied. This is a mechanical compute-environment failure before CUDA/build work or experiment execution, not scientific evidence.

The one corrected preflight allowance is consumed. No retry, GPU job, R/C change, fallback, measurement, or E1 work occurred. The expected PASS manifest remains absent. The separate terminal record is `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/e0-preflight-r3-corrected-20260916T192229Z-fb5cd9-terminal.json`, SHA-256 `2c7a6d508c33986c3a049bc68ea23b0e7f9b37b7b2c397abbd34e1b988b47da6`.
