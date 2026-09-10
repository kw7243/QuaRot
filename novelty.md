# v2 E0 reconciliation

The active authority is `/home/ubuntu/firstmate/data/research-plans/idea4_absorbable_symmetry_experimental_plan_v2.md`, selected by `CURRENT.md` and the 2026-09-09 v2 brief. The v1 plan, Phase 0A matrix, exact-invariance tests, Phase 0B baseline harness, and staging diagnoses are retained evidence; they do not open a v2 gate.

What v1 established and v2 reuses:

- The shared SwiGLU hidden permutation and positive up/down diagonal map is algebraically exact in full precision. The checked-in dependency-free FP32 tests remain background evidence.
- FPTQuant occupies fully mergeable diagonal MLP scaling, and existing permutation/reordering work prevents a cosmetic novelty claim. A later contribution would have to cross the v2 quality/execution boundary against the listed cheap baselines.
- QuaRot's official packed MLP path retains an online generalized Hadamard immediately before dynamic activation quantization and the packed down projection. That concrete operation is the v2 E0 overhead target.

What must be redone under v2:

- v1's W4A4KV4 WikiText baseline-reproduction gate is not the v2 E0 decision. E0 compares exactly one strong execution reference R with one no-MLP-transform counterfactual C, while attention/KV and every quantizer field remain fixed.
- v1 did not produce paired, warmed, randomized packed-path timings. E0 must measure the complete MLP at decode batch 1, decode batch 16 if memory permits, and prefill 1x2048, with at least 30 pairs, plus a short end-to-end baseline when the maintained path runs.
- The previous `BLOCKED_STAGE_VALIDATION` outcome is superseded for this lane. Its preserved evidence identifies a wrapper-validator bug, not a repository-copy failure. A fresh v2 stage must use the proven semantic checksum boundary and create a new immutable manifest; no historical stage or manifest may be changed.

E0 makes no model-quality or novelty claim. C is explicitly an execution counterfactual until a separately generated, accurate checkpoint exists. E1 and every later gate remain closed.
