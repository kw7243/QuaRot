# Current decision

State: **STOPPED — REGISTRY-RECOVERY CPU PREFLIGHT FAILED**

The 2026-09-16 registry-recovery authorization preserved failed jobs `1927348` and `1977465` and reused the immutable stage at source commit `4bc8433528cec54e6b0c32ce94bd755f80303e33`. No stage or earlier record changed.

The first unscheduled proof disproved base-prefix isolation: micromamba `create` appended its fresh scratch prefix to `/afs/csail.mit.edu/u/k/kwen1/.conda/environments.txt`. That failed proof is preserved and the AFS file was not repaired or rolled back. A second unscheduled proof passed by initializing an empty valid conda prefix and using micromamba `install`; a real pinned package transaction completed while the AFS registry content and metadata stayed unchanged. Recovery manifest `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/e0-mamba-registry-recovery-20260916T201036Z-f20ee9.json` has SHA-256 `661df5b135940ee077d3948fe638e2e9457e18a85c76cfdc3a65e4a8fd63c731`.

The one authorized CPU attempt, job `1978687`, then failed in one second at `[[ -r "$environment_registry" ]]`: the `--export=NIL` allocation could not read the AFS registry. It did not reach scratch environment initialization, micromamba, CUDA/build work, or experiment execution. This is an implementation failure, not scientific evidence.

The fresh CPU allowance is consumed. No retry, GPU job, R/C change, fallback, measurement, or E1 work occurred. The expected PASS manifest remains absent. The separate terminal record is `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/e0-preflight-r3-registry-20260916T201810Z-dc3fe6-terminal.json`, SHA-256 `947d6a05d22584b90453e5ccf4845e9082ff49983b10742394956c70d3c171c2`.
