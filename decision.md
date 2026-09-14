# Current decision

State: **STOPPED — CPU PREFLIGHT FAILED**

The fresh immutable stage passed at source commit `4bc8433528cec54e6b0c32ce94bd755f80303e33`. The sole authorized CPU CUDA/build preflight, Slurm job `1927348`, then failed before environment creation because its submission used the retained attempt directory as `--chdir`; the frozen script requires the current directory to equal the immutable stage. This is an implementation failure, not scientific evidence, and it occurred before the repaired micromamba `--no-rc` operations.

The exact-one preflight allowance is consumed. No retry, GPU job, R/C change, fallback, measurement, or E1 work occurred. A later attempt would require new authority and should change only the submission working directory to the exact immutable stage.

The expected PASS manifest remains absent. The separate terminal record is `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/e0-preflight-r3-20260914T182817Z-c0865a-terminal.json`, SHA-256 `1622c1c9af7e823044fc55c1ca62ea9a4a92b3ea346fb2aec8aab8ea765c8649`.
