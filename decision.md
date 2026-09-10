# Current decision

State: **E0 PREPARATION**

The v2 plan supersedes the historical v1 Phase 0B stop. The retained QuaRot packed path exposes the exact online MLP operation E0 needs to price: generalized Hadamard before down-input activation quantization. The next permitted experiment is one paired R/C E0 timing run from a newly validated CSAIL stage. No E1 work is permitted.

Predeclared primary workload: single-sequence decode, one token at context length 2048. Secondary workloads are batch-16 decode at the same context when memory permits and one 2048-token prefill for the complete MLP path. The run must record any end-to-end path incompatibility as implementation evidence rather than substitute a different model silently.
