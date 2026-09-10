# Current decision

State: **ONE-RETRY**

The v2 plan supersedes the historical v1 Phase 0B stop. The retained QuaRot packed path exposes the exact online MLP operation E0 needs to price: generalized Hadamard before down-input activation quantization. Fresh staging passed, but GPU job `1816371` failed before the experiment because `micromamba --clone` omitted the source environment's pip-installed PyTorch. This is a bounded implementation failure, not scientific evidence.

Exactly one retry is permitted: create a new scratch environment, install the already pinned checked-in requirements before compiling QuaRot, restage the committed fix, and repeat the same frozen R/C contract. No second retry or E1 work is permitted.

Predeclared primary workload: single-sequence decode, one token at context length 2048. Secondary workloads are batch-16 decode at the same context when memory permits and one 2048-token prefill for the complete MLP path. The run must record any end-to-end path incompatibility as implementation evidence rather than substitute a different model silently.
