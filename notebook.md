# Research notebook: absorbable Transformer symmetries

This is an append-only record. Add dated entries; do not rewrite prior entries.

## 2026-09-05 — Phase 0A / exact-invariance gate opened

- Isolation verified before branching: `pwd -P` and `git rev-parse --show-toplevel` both resolved to `/home/ubuntu/.treehouse/triton-ff92c5/16/triton`, the disposable task worktree.
- Research branch: `fm/absorbable-transformer-symmetries-r1`.
- Exact base commit: `f893845b9b91599ebd3b7a9c7f28164f39c7ed94` (`[AMD] Fix empty range inference for HistogramOp in RangeAnalysis (#11246)`).
- Authoritative plan: `/home/ubuntu/firstmate/data/research-plans/idea4_absorbable_symmetry_experimental_plan.md`.
- Plan-read attestation: read the complete 759-line plan in full before research or implementation.
- Authorized scope: Phase 0A, the exact-invariance gate for the first literature-approved candidate only, and a non-executed Phase 0B handoff. No GPU/model/baseline experiment is authorized in this task.

### Storage and reproducibility policy

- Canonical retained-data root: `RESEARCH_ROOT=/data/vision/torralba/u/kwen1`. Retained repositories, accepted results, plots, notebooks, and manifests belong there. This task-specific instruction overrides the generic CSAIL storage skill's preference to keep repositories on scratch.
- Ephemeral/reconstructable root: `SCRATCH_ROOT=/data/scratch-fast/kwen1`. Environments, caches, staging working copies, redownloadable weights, model downloads, and temporary data belong there.
- Allocation-local `TMPDIR` or `SLURM_TMPDIR` is disposable. Historical scratch paths must be preserved and never rewritten.
- Before any future experimental command, create and verify a full self-contained timestamped repository stage under `SCRATCH_ROOT` with `/home/ubuntu/.codex/skills/research-reproducibility/scripts/stage_and_run.sh`; run only from that stage. Before exit, write or verify durable accepted outputs under `RESEARCH_ROOT` and record the stage, commit, command, environment, inputs, outputs, and decision here.
- No environment, model, dataset, remote CSAIL access, Slurm allocation/job, repository clone/fork, or baseline/evaluation run is created in this task.

### Primary-source ledger

Sources will be appended as they are inspected. Secondary summaries are not evidence for the novelty decision.

## 2026-09-05 — Primary-source audit and novelty decision

### Primary sources inspected

- QuaRot, arXiv `2404.00456v2`: §§3.4 and 4, especially Eqs. (4), (8)–(15), plus §5.3 group-wise quantization. https://arxiv.org/html/2404.00456
  - Official repository `spcl/QuaRot`, inspected main commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a` (Apache-2.0), especially `fake_quant/rotation_utils.py` and `quarot/transformers/kv_cache.py`. https://github.com/spcl/QuaRot
- SpinQuant, arXiv `2405.16406v4`: §3.1 and Eqs. (2)–(4), including the explicit mergeable `R1/R2` versus online `R3/R4` split. https://arxiv.org/html/2405.16406
  - Official repository `facebookresearch/SpinQuant`, inspected main commit `8f47aa3f00e8662caf1a484153920a07e5281c3a` (repository `LICENSE`: CC BY-NC 4.0), especially `eval_utils/rotation_utils.py` and `train_utils/apply_r3_r4.py`. https://github.com/facebookresearch/SpinQuant
- FPTQuant, arXiv `2506.04985v2`: §3.1.1–3.1.4, Thm. 3.1/Eqs. (1)–(5), and Appendix A Tables 6–7. https://arxiv.org/html/2506.04985
  - The inspected primary paper does not link an author implementation repository; no code claim was inferred.
- GaugeQuant, arXiv `2607.20757v2`: §3.1/Eqs. (5)–(6), §4/Eq. (15), and §5's g128 setup. https://arxiv.org/html/2607.20757
  - Official repository `MPedraBento/gauge-quant`, inspected main commit `103bb4a0bc3e1b06c689098509121d868f444f19` (no repository license detected), especially `gauge_quant/fusion.py` and `gauge_quant/modules.py`. https://github.com/MPedraBento/gauge-quant
- ReSpinQuant, arXiv `2604.11080v2`: §3.1–3.2 and Eqs. (1), (3), including offline layer-wise fusion and online low-rank residual alignment. https://arxiv.org/html/2604.11080
  - The inspected primary paper does not link an author implementation repository; no code claim was inferred.
- DuQuant, arXiv `2406.01721v3`, added as the closest permutation precedent exposed by FPTQuant Appendix A: §3.2/Eq. (5) and §4.2's permutation-frequency overhead. https://arxiv.org/html/2406.01721
  - Official repository `Hsu1023/DuQuant`, inspected main commit `d56cfc6fe97c34c0eb100fec82fe439865905679` (MIT), especially `models/transformation.py`. https://github.com/Hsu1023/DuQuant

### Facts, derivation, and decision

- **Fact:** FPTQuant already uses a fully mergeable MLP up/down diagonal scaler. Scaling alone is occupied.
- **Fact:** DuQuant already uses channel permutation to balance activation outliers across blocks, but applies its combined activation transform online and reports 8.9–9.3% added W4A4 computation for the permutation-bearing setting. “Permutation helps outliers” is occupied.
- **Fact:** QuaRot, SpinQuant, and GaugeQuant retain an online transform at the post-SwiGLU/down-projection boundary. GaugeQuant explicitly shows why a general dense rotation cannot commute through SiLU and the Hadamard product, and its official `fuse_mlp_gauge` still requires online `H @ R`.
- **Derivation:** a shared permutation is a structured exception. For `H=SiLU(XWg)⊙XWu`, `Wg'=WgP`, `Wu'=WuPD`, and `Wd'=D^-1 P^T Wd` give `H'=HPD` and exactly cancel at `Wd'`. Biases transform in the same output coordinates. Every factor is stored in an existing weight/layout.
- **Decision:** Candidate A is `GO`, limited to the non-cosmetic claim of a fully absorbed shared SwiGLU hidden permutation plus scaling for order-sensitive fixed-group quantization. Implement/test A only.
- **Unresolved inference:** Candidate B's whole-RoPE-plane conjugation is algebraically exact, but compatibility with fused RoPE/cache kernels and a useful cross-plane group is not established by a maintained primary implementation; classify `UNCERTAIN`.
- **Unresolved inference:** Candidate C's compatible GQA group reindexing is algebraically exact, but the inspected QuaRot K-cache path uses token-wise or per-head (`head_dim`) grouping, where head order has no effect; no primary cross-head grouping target is established; classify `UNCERTAIN`.

## 2026-09-05 — Candidate A exact-invariance gate

- Implementation/test artifact: `symmetry_invariance_tests.py`. It uses only the Python standard library and explicitly rounds scalar arithmetic to IEEE binary32. The host Python lacked PyTorch, NumPy, and pytest; per task constraints, no environment or package was created or installed.
- This is local verification, not an experimental/model/evaluation command, so the research-reproducibility skill does not require a staged repository copy.
- Command: `python3 -m unittest -v symmetry_invariance_tests.py`.
- Result: 6/6 passed. Coverage includes biased standalone SwiGLU, ordinary shapes, a 257-unit hidden dimension with a g128 boundary plus one-element tail and cross-group swaps, exact transform/inverse round trip for binary scales, a pre-norm residual block, invalid/noninvertible inputs, and a dense-rotation negative case.
- Measured FP32 errors (acceptance thresholds: relative L2 `<3e-5`, max absolute `<2e-5`):
  - ordinary operation: relative L2 `1.156782114e-07`, max absolute `1.192092896e-07`;
  - g128 plus tail operation: relative L2 `1.986395943e-07`, max absolute `7.152557373e-07`;
  - one pre-norm residual block: relative L2 `1.676094190e-07`, max absolute `1.430511475e-06`.
- Negative boundary: attempting to pass a dense 2×2 rotation through SiLU and the elementwise product produced L2 error `1.514922205`, confirming that the monomial structure is essential rather than a general rotation claim.
- Round trip: exact equality for all weights and biases with binary diagonal scales.

## 2026-09-05 — Phase 0B handoff boundary

- `phase_0b_handoff.md` names model/tokenizer/dataset/calibration/quantizer inputs, primary baseline repositories and licenses, acceptance checks, result schema, durable output rules, and the future staged command boundary.
- No Phase 0B command was executed. No model, dataset, environment, cache, remote CSAIL session, GPU allocation, or Slurm job was created or accessed.

## 2026-09-05 — Project memory maintenance

- Ran `/home/ubuntu/firstmate/bin/fm-ensure-agents-md.sh .` because the repository already contains `AGENTS.md` and this task produced durable research artifacts.
- The helper added the standard `## Maintaining this file` section to `AGENTS.md` and the `CLAUDE.md` pointer to `AGENTS.md`; no task-specific research claim was added to the general project instructions.

## 2026-09-05 — Committed artifact provenance

- Phase 0A artifact commit: `9e56b0fa9e3dc528c43e8cf66d625861583cbc47` (`Add absorbable symmetry Phase 0A gate`).
- Exact parent/base: `f893845b9b91599ebd3b7a9c7f28164f39c7ed94`; the artifact commit is a direct child, so its ancestry is a one-commit fast-forward from the recorded local `main`.
- The no-mistakes test-quality rules informed behavioral coverage, but its push/PR/CI pipeline was deliberately not invoked because the delivery contract is local-only and expressly forbids a pipeline, push, or PR. No no-mistakes daemon lifecycle command was issued.

## 2026-09-06 — Phase 0B resumed; model-access preflight stop

### Scope and provenance

- The captain explicitly resumed the research plans and authorized exactly the Phase 0B baseline-reproduction gate in `phase_0b_handoff.md`; Phase 1 remains closed until every Phase 0B acceptance and stop condition passes.
- The complete authoritative plan at `/home/ubuntu/firstmate/data/research-plans/idea4_absorbable_symmetry_experimental_plan.md` was read in full before this research program began, as attested in the first entry of this append-only notebook. Before Phase 0B action, the checked-in handoff and this notebook were reread completely.
- Isolation and starting state were reverified: worktree `/home/ubuntu/.treehouse/triton-ff92c5/16/triton`, branch `fm/absorbable-transformer-symmetries-r1`, clean starting tip `651c1dfb5b4a474856e15b99f214dcc4704bd3be`.
- The `research-reproducibility`, `csail-storage`, and `csail-slurm` skills governed source staging, root placement, and cluster-resource inspection. The complete Slurm reference was read before constructing any Slurm command.

### Storage binding and retained baseline source

- Storage policy remains unchanged: `RESEARCH_ROOT=/data/vision/torralba/u/kwen1`, `SCRATCH_ROOT=/data/scratch-fast/kwen1`, and a future allocation must bind `TMPDIR` to allocation-local `SLURM_TMPDIR` (or an allocation-local per-job fallback). Historical stages and manifests were not rewritten.
- Via `gh-axi`, confirmed that the captain's existing public fork `https://github.com/kw7243/QuaRot` and upstream `https://github.com/spcl/QuaRot` both resolve to handoff commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a`. No new fork was created.
- Retained clone: `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/baseline-src`, local branch `fm/absorbable-transformer-symmetries-r1`, Apache-2.0, clean at the exact pinned commit. No push or remote mutation occurred.
- To make the retained source and stage self-contained, initialized the exact recorded submodules: CUTLASS `ffa34e70756b0bc744e1dfcc115b5a991a68f132`, fast-hadamard-transform `4ea722e434e3d4f2a14522341959ebdbe62be2de`, and NVBench `d8dced8a64d9ce305add92fa6d274fd49b569b7e`. Parent and submodule status are clean.

### Fresh stage and manifest

- The stock reproducibility staging tool's first attempt, `/data/scratch-fast/kwen1/absorbable-transformer-symmetries/staging/20260905_213522-021526-5008669-code`, could not set source permission metadata on the scratch filesystem. It was preserved without reuse.
- A second stage, `/data/scratch-fast/kwen1/absorbable-transformer-symmetries/staging/20260905_213605-d67b78-5008669-code`, was made before the pinned submodules were initialized and is therefore preserved but not accepted as the complete stage.
- Accepted preflight stage: `/data/scratch-fast/kwen1/absorbable-transformer-symmetries/staging/20260905_213929-df5a9c-5008669-code`. It was produced with the stock `stage_and_run.sh --stage-only` flow, adding only `rsync --no-perms --no-owner --no-group` to accommodate scratch metadata semantics. The supervising SSH session ended after the complete copy, so the stock metadata body was run separately and records that fact.
- Stage verification facts: `.git` is present; there is no object alternate; parent HEAD is the pinned QuaRot commit; tracked parent status is clean; all three submodule commits match and are clean; `REPRODUCIBILITY_METADATA.json` parses; its SHA-256 is `130445f7c00f5f1e5cb555fddc1c44eea3fcdb20f35118314fbd035f62f4145b`.
- Immutable durable attempt manifest: `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/phase0b-preflight-20260906T014549Z.json`, SHA-256 `3a62dfbf9230b5c0131c546779be168e5b2e66e072b17bb6f1f7c2d232660250`. A byte-identical committed copy is `phase_0b/phase0b-preflight-20260906T014549Z.json`.
- No experimental command was executed. No environment was created, and no model or dataset was downloaded. The complete stage therefore precedes any future experimental command as required.

### Primary metadata and model-access evidence

- Hugging Face primary API, `meta-llama/Llama-3.2-3B`: pinned model/tokenizer revision `13afe5124825b4f3751f836b40dafda64c1ed062`, license tag `llama3.2`, access state `gated: manual`. An unauthenticated HEAD request for `model.safetensors.index.json` at that revision returned HTTP 401.
- Hugging Face primary API, allowed fallback `meta-llama/Llama-3.2-3B-Instruct`: pinned model/tokenizer revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, license tag `llama3.2`, access state `gated: manual`. Its pinned artifact probe also returned HTTP 401.
- Hugging Face primary API, `Salesforce/wikitext`: pinned dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`, public and not gated. It was not downloaded because model access failed first.
- Credential/cache audit facts: no `HF_TOKEN` or equivalent environment name; no token or `stored_tokens` file in the permitted AFS/scratch Hugging Face locations; no cached exact base or Instruct 3B model found under `RESEARCH_ROOT` or `SCRATCH_ROOT` through depth seven. An older scratch environment was inspected read-only but not adopted (`Python 3.10.20`, `torch 2.8.0+cu128`, `transformers 5.12.1`, `datasets 4.0.0`).
- **Inference:** an authorized Hugging Face credential for an account that has already accepted the Llama 3.2 terms is required. Accepting those terms or fabricating/substituting credentials is outside this worker's authority. The allowed Instruct fallback does not resolve the stop because it is gated identically.

### Dynamic GPU inspection and gate outcome

- Slurm preflight at `2026-09-06T01:45:50Z`: account `vision-torralba-urops-meng` and QoS `vision-torralba-interactive` are eligible. `vision-torralba-rtx3090` exposed three nodes with seven RTX 3090 GPUs each; no GPU TRES was allocated on those nodes. The user's queue was empty.
- **Resource inference:** one RTX 3090 (24 GB), eight CPUs, and 32 GiB host memory is the smallest adequate listed Torralba allocation for serialized Llama-3.2-3B evaluation. No larger or multi-GPU resource was selected, and no allocation/job was submitted because the model-access preflight had already failed.
- Exactly one event owner was maintained: this Codex crewmate. Slurm job ID: none.
- **Phase 0B outcome: `BLOCKED_PRECHECK`.** Source/stage provenance passed, but the required model access grant is unresolved. FP16 source agreement, matched RTN/QuaRot graph audit, QuaRot unquantized equivalence, quantized baselines, and the two deterministic PPL reruns were not run; there are no accepted results or `baseline_results.csv`.
- **Stop decision:** do not open Phase 1. After authorized model access is made available outside the repository, create a new immutable manifest and a new fresh complete stage before the first experimental command; do not reuse this blocked attempt as a run.

## 2026-09-06 — Phase 0B preflight commit provenance

- Phase 0B blocked-preflight artifact commit: `bb4e3b986c0e7275c75483e8f88c1ab175f9e6c0` (`Record Phase 0B model-access preflight stop`).
- Exact parent/resume tip: `651c1dfb5b4a474856e15b99f214dcc4704bd3be`; local `main` remains `f893845b9b91599ebd3b7a9c7f28164f39c7ed94` and is an ancestor, so the research branch remains fast-forwardable.
- Local validation: `python3 -m unittest -v symmetry_invariance_tests.py` passed 6/6; `git diff --check` passed; the committed JSON parsed with `python3 -m json.tool`; the durable and committed manifest copies have identical SHA-256 `3a62dfbf9230b5c0131c546779be168e5b2e66e072b17bb6f1f7c2d232660250`.
- Ran `/home/ubuntu/firstmate/bin/fm-ensure-agents-md.sh .`; the existing maintained `AGENTS.md` and `CLAUDE.md` pointer were already compliant and unchanged.

## 2026-09-06 — Phase 0B TIG-safe staging recovery

### Scope, plan, and storage

- The complete authoritative plan at `/home/ubuntu/firstmate/data/research-plans/idea4_absorbable_symmetry_experimental_plan.md` was read in full before any work, as recorded at notebook creation. This recovery remained inside its Phase 0B baseline-reproduction gate and was staging-only.
- Recovery began from branch `fm/absorbable-transformer-symmetries-r1` at exact tip `42da1bb57ed48d134eb7ae390f129622c2b02491`. The canonical source was clean at `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/baseline-src`, commit `921b912bb8f6b7f16122d55a9e769e4e8b2c6f44`.
- Storage bindings were explicit: `RESEARCH_ROOT=/data/vision/torralba/u/kwen1`, `SCRATCH_ROOT=/data/scratch-fast/kwen1`, and allocation-local `TMPDIR=/tmp/absym-phase0b-stage-1716601/tmp`. Both incomplete earlier stages and this new stage were preserved; no historical manifest or path was rewritten.

### Job 1711616 exact launch and TIG diagnosis

- Before replacement, the exact job 1711616 argv was reconstructed from the submitting tool call, its durable submitted state, and its captured pre-purge `scontrol` record. The immutable record is `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-20260906T050706Z-cfb8a0/launch-argv-1711616.json`, SHA-256 `d9c4f3e2e4b2ff8dac44d7f7e55406d600ebe80dc0f4cddc557dd95ad9a810b4`.
- Exact launch argv, rendered as a shell command:

  ```text
  sbatch --parsable --account=csail --qos=tig-main --partition=tig-cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:20:00 --no-requeue --export=NONE --job-name=absym-s-cfb8a0 --comment=phase0b-stage-20260906T050706Z-cfb8a0 --chdir=/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-20260906T050706Z-cfb8a0 --output=/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-20260906T050706Z-cfb8a0/stdout.log --error=/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-20260906T050706Z-cfb8a0/stderr.log /data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-20260906T050706Z-cfb8a0/stage_phase0b_cpu_job.sh phase0b-stage-20260906T050706Z-cfb8a0 /data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/baseline-src /data/scratch-fast/kwen1/absorbable-transformer-symmetries/staging-jobs/phase0b-stage-20260906T050706Z-cfb8a0 /data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-20260906T050706Z-cfb8a0 /data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/phase0b-stage-20260906T050706Z-cfb8a0.json 921b912bb8f6b7f16122d55a9e769e4e8b2c6f44 /data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-20260906T050706Z-cfb8a0/stage_and_run.sh
  ```

- **Fact:** argv contains `--export=NONE`, one of TIG's retained known-bad selectors that enters its unsupported `_get_user_env` path. Accounting was `1711616|absym-s-cfb8a0|CANCELLED by 0|0:0|00:00:02|groenig-2|None`; there was no batch stdout, stderr, running state, stage child, metadata, or manifest.
- **Cause separation:** the initiating trigger was the unsupported export selector; the masking condition was cancellation before any user-script evidence could be written; the visible symptom was a two-second root cancellation that had looked like another transport loss. The earlier successful TIG scout job `1663997` using literal `--export=NIL` disconfirms a general `sbatch`, AFS, or `tig-cpu` failure.

### Predeclared recovery cells and dynamic resource gate

- Attempt: `phase0b-stage-nil-20260906T161229Z-2f9eaf`. Durable plan: `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-nil-20260906T161229Z-2f9eaf/recovery-plan.json`, SHA-256 `ed62864b96af6e0a6645a23cc6af1dfb30c5c3711843488abadd203f4a6f6eea`. Prepared state SHA-256: `ccbaf045f136f395de50e24b04f8a57ea0539f270ca39d73d4f51c92824ceb25`.
- Cell 1 allowed exactly one TIG-safe batch submission with literal `--export=NIL`. Every required non-Slurm variable was defined in the script after all `#SBATCH` directives. Cell 2 predeclared one durable terminal owner, exactly one `salloc`, and at most one `srun --pty`, but only if Cell 1 unambiguously terminated before script entry.
- Immediately before submission, association `csail` listed QoS `tig-main`; partition `tig-cpu` was `UP` and allowed `tig-main`; 56 CPUs were idle across eligible nodes; no replacement job, submitted/running/final state, stdout/stderr, stage child, or expected manifest existed. Requested resources were one node, one task, one CPU, 2 GiB, 20 minutes, no GPU, and no requeue—the smallest adequate staging resource.

### Single submission and terminal evidence

- Exact replacement command:

  ```text
  sbatch --parsable --export=NIL /data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-attempts/phase0b-stage-nil-20260906T161229Z-2f9eaf/stage_phase0b_cpu_nil_job.sh
  ```

- Job ID: `1716601`; one submission and one event owner. Executed script SHA-256 `a3005aede3b7ee1b09b31f3159e6fed199e661882e1f77990df8030b8cb67fd1`; copied stock staging helper SHA-256 `44e5dc6f1a958b1f4b32e8dceeb49814885ad8dcca59716090ca87b1033731fa`.
- Terminal accounting: `1716601|absym-nil-2f9eaf|csail|tig-main|tig-cpu|2026-09-06T12:20:42|2026-09-06T12:20:42|2026-09-06T12:20:50|2026-09-06T12:26:21|00:05:31|FAILED|1:0|groenig-2|None|billing=20,cpu=1,mem=2G,node=1`. Durable stdout/stderr and submitted/running/final states are in the attempt directory. The final-state SHA-256 is `8df7acfada51122cc5cc4f00ba20b317e13c95967c583fe97681eb1895cfc821`.
- The script entered, wrote durable state, and completed the canonical copy. Therefore Cell 2 became and remains closed; no `salloc`, `srun`, second job, or retry was issued.

### Stage result and exact validator failure

- Preserved stage: `/data/scratch-fast/kwen1/absorbable-transformer-symmetries/staging-jobs/phase0b-stage-nil-20260906T161229Z-2f9eaf/20260906_122058-d687ea-921b912-code`.
- `REPRODUCIBILITY_METADATA.json` exists and parses, SHA-256 `f7ed0629b8fc41ad9e041a64ce7e2b3d5b85ad63042245b27d144f8557ec7439`. Independent checks found exact HEAD `921b912bb8f6b7f16122d55a9e769e4e8b2c6f44`, clean tracked state, source-matching initialized submodules, and no Git object alternates. These facts disconfirm an SSH transport, AFS-token, source-path, shared-storage-copy, or staging-helper copy failure.
- **Fact:** the terminal assertion was `[[ -z "$content_delta" ]]`. Repeating the exact read-only dry run produced only directory timestamp records plus `.git/index` and each submodule index timestamp record. No working-tree or Git-object content path differed.
- **Cause:** the validator itself ran `git status`/submodule checks against the stage, refreshing Git index stat caches and directory mtimes, then compared those mutable implementation files and directory mtimes against the source. That masking condition turned a complete copy into a false content-delta failure. Separately, `grep -Eq '^[+\-U]'` emitted `Invalid range end`; it was not the terminal failure, but its submodule guard must be corrected before reuse.
- The expected success manifest `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/phase0b-stage-nil-20260906T161229Z-2f9eaf.json` is absent, as required after failed acceptance. Compact checked-in recovery evidence is `phase_0b/phase0b-staging-recovery-20260906T161229Z.json`.

### Stop decision

- **Phase 0B outcome: `BLOCKED_STAGE_VALIDATION`.** The stage reached valid reproducibility metadata, but the required success manifest and content-validation PASS do not exist. The already-known Hugging Face credential prerequisite also remains absent.
- Per the bounded recovery contract, no retry was submitted after the tracked job failed. No model, checkpoint, dataset, environment, experimental command, GPU job, license acceptance, account change, Phase 1 work, push, PR, merge, stable-branch promotion, or no-mistakes run occurred.
- Before any later authorized retry, make the content check semantic: ignore directory mtimes and mutable Git index stat caches while still checking working-tree bytes, exact commit, submodules, object self-containment, and clean Git state; also use a hyphen-safe submodule-status expression. A future retry must use another fresh timestamped stage and a new immutable manifest, never mutate or reuse this stage.

## 2026-09-06 — Post-copy validator diagnosis addendum

### Current state

- Diagnosis only; no new stage, job, allocation, fallback, or experiment was created. Job `1716601` remains the only recovery submission, and Cell 2 remains closed.
- Expected acceptance was: complete fresh copy, valid metadata, semantic source/stage equality, exact Git/submodule identity, then a new immutable success manifest. Observed acceptance reached every boundary except the raw rsync assertion and manifest.

### Evidence and cause

- The failed stage and source are both NFS with 131,072-byte blocks and nanosecond timestamps. The earlier accepted stage uses the same filesystem class. A general filesystem-type or timestamp-resolution boundary is disconfirmed.
- Failed-stage metadata was created at `2026-09-06T16:24:51.693775Z`; its parent/submodule indexes were rewritten at `16:25:56–16:25:59Z`. Thus the index mutation happened during post-copy validation, after helper completion, not during copy.
- The earlier accepted stage at commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a` has the same normal sequence: metadata at `01:43:33Z`, index rewrites at `01:44:45–01:44:48Z`. This comparison concerns sequencing only because the commits differ.
- For the failed stage, `git ls-files --stage` hashes match the source for the parent and all three submodules, while `git ls-files --debug` hashes differ. This isolates the byte difference to stat-cache fields, not paths, modes, stages, or blob IDs.
- The smallest no-job counterfactual used checksum-based `rsync -rlnic`, the normal cache/output exclusions, and exclusions only for `/.git/index` and `/.git/modules/**/index`. It ignores timestamp attributes while checking regular-file bytes and symlink targets. Result: empty. A separate `git --no-optional-locks status` check left every index hash and mtime unchanged.
- Bounded object verification also passed: `git fsck --full --no-dangling` succeeded for the parent and all three submodules, and no parent/submodule object database contains an alternates file.
- **Trigger:** post-copy stage `git status` refreshed Git stat caches, while metadata creation changed directory mtimes. **Mask:** a later archive-mode dry run treated those mutable bytes/times as content. **Symptom:** the empty-delta assertion failed after a complete copy, leaving valid metadata but no success manifest.
- **Conclusion:** this is a validator sequencing/representation bug. It is not an SSH transport, AFS token/path, shared-storage copy, staging-helper copy, Git semantic-index, or repository-content failure.

### Helper and code history

- The executed helper is byte-identical to the current stock helper, SHA-256 `44e5dc6f1a958b1f4b32e8dceeb49814885ad8dcca59716090ca87b1033731fa`; its directory has no available Git history. It archives the repo, then writes metadata. The wrapper adds stage Git inspection and the failing raw comparison afterward.
- Retained repo history provides two implementation precedents: `c53cf746ade4c615aa3890c8be1b8b152df02038` validates a semantic working-tree manifest and self-contained Git identity; `0d79adad95c5d699fb21c9bd5d9aa8965f7fa211` uses `git --no-optional-locks`, `git fsck`, and tracked-content hashing.

### Implementation-ready recommendation

- Use `git --no-optional-locks` for source/stage status, fix the submodule guard to `^[-+U]`, add staged parent/submodule `git fsck --full --no-dangling`, and compare semantic index entries.
- Replace `rsync -ani` with the proven-empty `rsync -rlnic` form, retaining normal output/cache exclusions and excluding only parent/submodule Git index files. Persist the itemized delta before asserting it is empty.
- Before retry authorization, add local tests that force a post-copy index refresh and require PASS; tamper with a tracked file while preserving its size/mtime and require FAIL; require FAIL for a dirty/uninitialized submodule and a missing Git object.
- Exact future retry prerequisites: explicit authorization; committed/tested validator; recorded script/helper hashes; no live or ambiguous allocation; fresh dynamic `csail/tig-main/tig-cpu` check; one new timestamped attempt, empty scratch stage parent, and absent durable manifest path; exactly one CPU-only literal-`NIL` job; metadata plus semantic/Git/submodule/object checks plus new manifest all PASS.
- Even after staging PASS, stop before model work while the authorized Hugging Face credential remains absent. Phase 0B is blocked, and Phase 1 stays closed.
- Detailed evidence and exact counterfactual: `phase_0b/stage_validation_diagnosis.md`; machine-readable record: `phase_0b/phase0b-staging-recovery-20260906T161229Z.json`.

### Retention and local validation

- Immutable durable diagnosis directory: `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/staging-diagnoses/phase0b-stage-nil-20260906T161229Z-2f9eaf`. `diagnosis.json` SHA-256 `1ddc05e5bb1ee06602b0645899bd216d56bf37d9b1b2a4637b700b07d63267ca`; `diagnosis.md` SHA-256 `76dbc994b750d71eca3072af26198db59c490aacbeafd5745881b92cf113145c`; notebook snapshot SHA-256 `58880bbb0bfad6e821f4fad0462ff093dc855eca54a4d072730a15b1384b08c8`. The failed stage and expected success-manifest path were untouched.
- Validation: the diagnosis JSON parsed; all Phase 0B shell scripts passed `bash -n`; all Phase 0B Python scripts passed bytecode compilation using a disposable `/tmp` cache; `python3 -m unittest -v symmetry_invariance_tests.py` passed 6/6; `git diff --check` passed; a credential-pattern scan found no secret material.
- `/home/ubuntu/firstmate/bin/fm-ensure-agents-md.sh .` reported the existing maintained `AGENTS.md` and `CLAUDE.md` pointer unchanged.
- Final diagnosis copies were added without overwriting the first snapshot: `diagnosis-v2.json` SHA-256 `bb8a1c25d877133ddb3ecb425e1eed234ff94cb3f17721a315f8633e266d2081`, `diagnosis-v2.md` SHA-256 `91a5a9edc6eec95c3a4a06f0ab258852bdbbde4a255162bc0675669f6c7cacd8`, and `notebook-v2.md` SHA-256 `18abd9afd9e6c5d160c058bfe520b72c85866806beb59b8ea9c2142284a66a21`, all in the same immutable diagnosis directory.

### Diagnosis commit provenance

- Diagnosis artifact commit: `16bfa71d2124ec9de5b362579141761f8f7b0bc6` (`Record Phase 0B staging validation diagnosis`), exact parent `42da1bb57ed48d134eb7ae390f129622c2b02491`. Local `main` at `f893845b9b91599ebd3b7a9c7f28164f39c7ed94` remains an ancestor.
- The retained QuaRot adapter patch passed `git apply --reverse --check` against canonical source commit `921b912bb8f6b7f16122d55a9e769e4e8b2c6f44`. Whitespace warnings are confined to byte-preserved upstream patch context and the exact pre-TIG provenance script; the authored diagnosis and active harness files passed the staged whitespace check.
- No push, PR, merge, stable-branch promotion, or no-mistakes invocation occurred.

## 2026-09-10 — v2 E0 supersession and reconciliation

- Inbox authority: read `/home/ubuntu/firstmate/state/absorbable-transformer-symmetries-r1.inbox/001.msg`; it directs this branch to the current v2 E0 Role C gate and forbids a secondmate.
- Full-read attestation: before v2 implementation or any experimental command, read in full `/home/ubuntu/firstmate/data/absorbable-transformer-symmetries-r1/brief.md`, `/home/ubuntu/firstmate/data/research-plans/CURRENT.md`, `/home/ubuntu/firstmate/data/research-plans/idea4_absorbable_symmetry_experimental_plan_v2.md`, `/home/ubuntu/firstmate/data/captain-shared.md`, this entire notebook, every checked-in v1 Phase 0A/0B artifact, the complete `research-reproducibility`, `csail-slurm`, and `csail-storage` skills, their required references, and the complete stock staging helper.
- Branch remains `fm/absorbable-transformer-symmetries-r1`. Exact clean v2 starting tip: `ffb99eca8dbea4929b0e368c9184df89c322efeb`; merge base with local `main`: `f893845b9b91599ebd3b7a9c7f28164f39c7ed94`.
- The v2 plan is authoritative. The unversioned v1 plan and prior Phase 0B baseline-PPL gate are historical evidence only. The old `BLOCKED_STAGE_VALIDATION` state is explicitly superseded for this lane; its incomplete stages, failed job evidence, diagnosis, and absent success manifest remain untouched.
- Captain decision recorded verbatim for the supersession boundary: “We have a new plan now ... the progress we had previously was from our old research plan ... you can still use the results we had from our old plan ... we’ll be pushing forth the new plan.” The matching unkeyed v1 status block was closed with one `resolved:` line before E0 continued.
- Reconciliation artifact: `novelty.md`. v1's exact MLP algebra and source audit remain background evidence. v2 E0 must newly price a concrete online transform with real packed execution; it makes no quality or novelty claim.
- Frozen E0 semantics: `contract.json`, ID `e0-quarot-packed-mlp-hadamard-removal-v1`. Exactly two points are allowed: R is the maintained QuaRot packed W4A4 MLP with its online generalized Hadamard before down-input quantization; C is identical except that operation is absent. Attention/KV stay fixed. C is an execution counterfactual, not an accurate checkpoint.
- Predeclared primary workload: single-sequence one-token decode at context 2048. Also measure batch-16 decode at context 2048 if memory permits and one 2048-token prefill for the complete MLP. Use 40 warmed randomized pairs, accepting no fewer than 30.
- CSAIL read-only inspection at `2026-09-10T00:00Z` found the retained QuaRot repo clean at `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/baseline-src`, adapter commit `921b912bb8f6b7f16122d55a9e769e4e8b2c6f44`, with pinned initialized submodules. The official packed MLP path is `e2e/quantized_llama/modeling_llama.py:QuarotLlamaMLP`; R's online transform is `quarot/nn/hadamard.py:OnlineHadamard`, followed by `quarot/nn/quantization.py:Quantizer` and `quarot/nn/linear.py:Linear4bit`.
- Existing scratch environment `/data/scratch-fast/kwen1/absorbable-transformer-symmetries/envs/quarot-phase0b-py310-cu121-r1` is preserved unchanged. It contains Python 3.10, PyTorch 2.4.1+cu121, and Transformers 4.44.2 but no installed QuaRot, fast-Hadamard, or FlashAttention package. E0 will use a new scratch environment and will not mutate the v1 environment.
- Storage remains exact: retained accepted outputs/manifests under `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries`; full timestamped repo stages, environment, cache, and redownloadable build inputs under `/data/scratch-fast/kwen1/absorbable-transformer-symmetries`; allocation-local temporary state under `SLURM_TMPDIR` or a per-job `/tmp` fallback. No historical path or manifest may be rewritten.

## 2026-09-10 — v2 E0 staging PASS and bounded setup retry

- Read the updated 139-line brief in full after Firstmate released the external login wait. Fleet smoke job `1815902` had completed `0:0` on `torralba-3090-3`; the canonical SSH endpoint resolved to `slurm-login-2`. The old Phase 0B staging and credential gates remain historical.
- Retained QuaRot source received the locally validated E0 overlay and was committed cleanly on the named research branch at `945e5432ab2081123971747c2d9517376156a4bd` (parent `921b912bb8f6b7f16122d55a9e769e4e8b2c6f44`). No remote push occurred.
- Fresh CPU-only staging attempt: `e0-stage-20260910T180231Z-4eee3b`, job `1816187`, `csail/tig-main/tig-cpu`, one CPU, 2 GiB, literal `--export=NIL`, no requeue, no GPU. Terminal accounting: `COMPLETED|0:0|00:07:27|groenig-1`.
- Accepted stage: `/data/scratch-fast/kwen1/absorbable-transformer-symmetries/staging/e0-stage-20260910T180231Z-4eee3b/20260910_140344-174beb-945e543-code`. Exact HEAD `945e5432ab2081123971747c2d9517376156a4bd`; full metadata, Git tree/index, submodules, object stores, and checksum validation passed.
- Immutable stage manifest: `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/e0-stage-20260910T180231Z-4eee3b.json`, SHA-256 `0c547b6c90c55e3f03aec994f6c039ff2c711538144a49662f2381959d204b02`. Metadata SHA-256 `b654a33c50c3b2daa801554638f5fb7097608b17a66d7b1dca86400bc6bdf13e`; validation SHA-256 `01734cffd8f4632b7cbd0e61245db076f211453631bcf756ed6086ffb90e43f6`; persisted semantic delta was empty.
- Dynamic device inspection selected one RTX 3090 as the smallest compatible adequate device. RTX 2080 Ti/3080 have insufficient memory for the packed 7B plus intended cache; V100 is not among QuaRot's compiled architecture targets; H100/H200 are larger. The authorized tuple passed `sbatch --test-only` on `torralba-3090-3`.
- First E0 GPU event: attempt `e0-run-20260910T181304Z-680fdc`, job `1816371`, one RTX 3090, 8 CPUs, 32 GiB, 2h30 ceiling, literal `--export=NIL`, no requeue, single owner. Terminal accounting: `FAILED|1:0|00:00:28|torralba-3090-3`; allocated GPU time was 0.0078 hours.
- Failure classification: **implementation failure before experiment**, not scientific evidence. `micromamba create --clone` reproduced conda records but omitted pip-installed PyTorch from the historical environment. QuaRot metadata generation then failed with `ModuleNotFoundError: No module named 'torch'`. Durable final state confirms `experiment_started=false` and no `results/` directory.
- Bounded fix: preserve failed scratch environment `quarot-e0-py310-cu121-r1`; create `quarot-e0-py310-cu121-r2`, install the already pinned `requirements-phase0b.txt`, then build QuaRot. The R/C contract, workloads, timing, GPU type, and scientific question are unchanged. This consumes the v2 plan's single targeted retry; no further automatic retry is allowed.

## 2026-09-14 — v2 E0 r3 immutable stage PASS; CPU preflight stop

### Recovery and source identity

- Resumed isolated branch `fm/absorbable-symmetry-v2-e0-preflight-r3` at local Triton base `f893845b9b91599ebd3b7a9c7f28164f39c7ed94`. The old recorded object `93e5c01e5bf6bc9985a7774361258c7aaa42e9c4` is absent from the local object store.
- Read-only reconciliation proved that the earlier denied transport copied no bytes and changed no retained Git state. The complete retained QuaRot source was clean on `fm/absorbable-symmetry-v2-e0-recovery-r2` at commit `4bc8433528cec54e6b0c32ce94bd755f80303e33`, tree `c6892dfc0662f8f74952b51566427d92fc9afe8e`.
- Durable records recover the missing object's exact repair: add `--no-rc` to the existing micromamba create and install calls. Preserved remote history carries it at `079d8737fe3e55abd0c9bcc0131b33dd4d7eb116`; later commit `4bc8433528cec54e6b0c32ce94bd755f80303e33` isolates all preflight state. The executed script SHA-256 is `5978246496f2e178e462226742a4eaec2f468a9825dcbde530a9056ffa7714e2`; frozen contract SHA-256 remains `433a7760df1fa5b05e1ac277ae2364deb03fba96be64a2de5ff467376f43156a`.

### Scheduler and immutable stage

- Dynamic inspection found `csail/tig-main/tig-cpu` and `vision-torralba-urops-meng/interactive/vision-torralba-rtx3090` available. No live matching absorbable-symmetry job existed before either submission.
- One CPU-only stage job was submitted with one terminal owner and literal `sbatch --parsable --wait --no-requeue --export=NIL`: job `1926606`, `csail/tig-main/tig-cpu`, one CPU, 2 GiB, 20-minute ceiling. It completed `0:0` in `00:07:25` on `groenig-2`; allocated TRES were `billing=20,cpu=1,mem=2G,node=1`.
- Accepted full stage: `/data/scratch-fast/kwen1/absorbable-transformer-symmetries/staging/e0-preflight-r3-20260914T182817Z-c0865a-stage/20260914_142941-dc0727-4bc8433-code`. Its parent and three pinned submodules are self-contained, the semantic content delta is empty, and validation passed. Manifest `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/e0-preflight-r3-20260914T182817Z-c0865a-stage.json` has SHA-256 `ee062c1b857565d19ceed659624d119f8b750f395f58e85b743fac0edf220311`; metadata SHA-256 is `ec9bc7055fa9be9ea8959e60416c2034a8ff6bbbe487f0dbb2d5ba035d377660`; validation SHA-256 is `9fb676e9415c2b5704bf468fdbb344bd900da3a7441f8c9d25924fb62f6eaebd`.

### Sole CPU preflight and stop

- One fresh scheduler-tracked CPU CUDA/build preflight was submitted with the required unattended form and one terminal owner: job `1927348`, `csail/tig-main/tig-cpu`, eight CPUs, 24 GiB, 90-minute ceiling, literal `--export=NIL`. The invocation passed the immutable stage as the first script argument but set `--chdir` to `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/preflight-attempts/e0-preflight-r3-20260914T182817Z-c0865a`.
- Job `1927348` failed `1:0` in `00:01:17` on `groenig-3`; allocated TRES were `billing=166,cpu=8,mem=24G,node=1`. The terminal assertion was `[[ "$(pwd -P)" == "$(realpath "$source_repo")" ]]`: the frozen script requires its current directory to equal the immutable stage. This is an implementation failure before environment creation, CUDA/build work, or experiment execution. It is not scientific evidence and is unrelated to whether the repaired micromamba operations work.
- Launch plan SHA-256: `f94d02f86e1018d92e1d45e17a5a773b3c5e94d75dc5de153f6f65e6ef6c324a`. Final state SHA-256: `1d0ce6cb452bb137749945963d26b8350418407772b60e97f9be142e62f8654b`. Standard output is empty; standard-error SHA-256 is `3964e4912d2aaa6d3f4fc8437dead635d06dcab418c2db7813d51c1ccad95fae`.
- The intended fresh environment and build root do not exist. The fresh mamba-state path contains only empty `root/` and `pkgs/` directories. No running-state, micromamba-info, source-copy delta, CUDA proof, or successful preflight manifest exists.
- The exact-one CPU preflight allowance is consumed. The gate stays closed: no retry, GPU submission, R/C change, fallback, cancellation, requeue, E1 work, or experiment occurred. The smallest unexecuted repair is to set `--chdir` to the exact stage, but it was not run.
- E0 never reached measurement. There are no retained R/C timing inputs for a valid table or plot. The frozen comparisons remain R versus C for `decode_b1_ctx2048_step1`, conditional `decode_b16_ctx2048_step1`, and `prefill_b1_s2048`, but this attempt produced none of those rows.
- The expected PASS manifest remains absent. Separate immutable terminal record: `/data/vision/torralba/u/kwen1/absorbable-transformer-symmetries/manifests/e0-preflight-r3-20260914T182817Z-c0865a-terminal.json`, SHA-256 `1622c1c9af7e823044fc55c1ca62ea9a4a92b3ea346fb2aec8aab8ea765c8649`.
