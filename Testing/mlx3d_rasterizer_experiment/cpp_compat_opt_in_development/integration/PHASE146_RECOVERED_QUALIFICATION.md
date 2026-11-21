# Phase 146 — Recovered scoped M4 qualification decision

**Status: PASS — scoped experimental guard, using existing GPU evidence and corrected CPU-only reaggregation.** This is *not* a production promotion.

## Provenance
- Source branch: `experiment/m4-phase146-reserve-transition-guard`.
- Original output: `phase146_guard_1.json` (reported FAIL from **aggregation bug**, retained unchanged).
- Original artifacts: `phase146_guard_1_artifacts/`, nine GPU-worker checkpoints.
- Corrected derivative output on user's M4: `phase146_guard_reaggregated_1.json` (user-reported PASS; full local JSON has not yet been uploaded to GitHub).
- CPU regression `test_phase146_aggregate_cpu.py`: **3/3 PASS**, user Mac output.
- CPU parity forensics: **0 differing RGB frames; 31/31 each of four blocks**.
- Root cause: old `summarize` selected first block 0 `mode='guard'` job, which was the **contracts** worker, rather than requiring `kind='lifecycle'`. Corrected selector committed in `f0c2c9519219de5630e2b3d113de731238060ada`. Original GPU workers were **not rerun**.

## Recovered qualification (user M4 terminal)

| Gate | Result |
|---|---|
| Original GPU workers | 9/9 verified with report/log hashes |
| Original contracts | 65/65 |
| Native dispatch boundaries | 5/5 |
| RGB parity | 124/124 bitwise-matched frame pairs |
| GPU admission ceiling | 30% |
| Minimum free memory at admission | 79% |
| GPU reruns for recovery | 0 |

### Performance and retained cache

Deltas below are experimental guarded candidate minus unchanged `keep` comparator; negative milliseconds represent improvement.

| Process block | First-small-frame Δ (ms) | Retained-cache Δ (bytes) | MLX peak Δ (bytes) |
|---|---:|---:|---:|
| 0 | −82.148 | +1,644,541 | 0 |
| 1 | −81.745 | +1,574,868 | 0 |
| 2 | −82.354 | +1,574,855 | 0 |
| 3 | −83.462 | +1,574,849 | 0 |

Median block-level first-small-frame difference (computed from rounded terminal values): **-82.251 ms**. Individual process-block reductions were all favorable, with no peak MLX increase. Cache overhead was approximately +1.5 MiB per block. Unlike Phase144's harness-only reset, Phase146 tested a separately opt-in `ExperimentalReserveResetRenderer` with a one-shot guard, scene-count boundary cases (1024,4096,4097) and camera-size edge (640 and 641 width), plus repeated large–small transitions. These tests do **not** establish broad workload, hardware, or training readiness.

## Gate decision

The experimental one-shot guard is qualified *for the specific tested Mac mini M4 source checkout and transition matrix*. Keep it as a separate reviewable candidate. Before production integration, independently qualify an actual production diff and additional scene/camera transitions, and verify absence of added latency/regressions in other workloads. Leave `feature/cpp-compat-renderer`, protected Fast, production sort defaults and PR #1 unchanged. PR #1 remains draft/unmerged.

No further nine-worker rerun is needed for this recovered result. Preserve original FAIL and original source-bound checkpoints for auditability; the CPU derivative report should be archived with provenance before any wider decision.
