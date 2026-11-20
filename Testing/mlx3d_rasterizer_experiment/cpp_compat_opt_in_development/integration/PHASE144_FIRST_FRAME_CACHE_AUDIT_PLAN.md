# Phase 144 — First-frame/cache attribution and fallback audit (planned; no new GPU qualification)

## Evidence-based rationale
Phase 140–143 PASS in commit `0e13dbe78d5e94219e5e43bf1c844f0a0165458a`. For the moderate Full HD fixture, the narrowly scoped experimental selector showed warmed median `combined-auto = -9.757 ms` (4/4 independent blocks) and peak MLX `-139.1 MiB`. Explicit scatter512 showed warmed `-9.248 ms` with the same dispatch. **No density passed the Phase141 active sizing screen; active sizing stays disabled.** Highres mixed (+0.733 ms median). Phase143 did not claim general first-render or memory-cache qualification. Original forward-only M4 release remains CONDITIONALLY READY.

Key anomaly: fullhd *active* first-frame `-11.691 ms` while warmed frame `+4.664 ms`, despite `-218.9 MiB` MLX peak and `+1319.6 MiB` retained cache. Fullhd *combined* first-frame `-4.838 ms` and warmed `-9.757 ms`, with `-265.3 MiB` retained cache vs auto. These are different process measurements; neither high-water memory nor system Metal cache state is an isolated kernel allocation. The next useful question is whether timing and retained-cache behavior changes with first-use order and scene lifecycle; blindly revisiting active sizing is not justified.

## Stage 1 — CPU-only audit using existing raw Phase143 evidence
- Read archived `qualification_evidence/phase140/phase143_completed/phase143_comparison.json`, `phase143.json`, and original per-mode worker JSON/checkpoints. Assert source digests/defaults and complete 61-worker matrix; do not substitute summary tables for missing raw worker records.
- Tabulate **per-process** first-frame, first warm frame, final warmed median, cache/active/peak/RSS at each measured stage, policy selection/dispatch, and after-scene-replacement first frame, by mode, block and workload.
- Compute block-matched candidate-minus-auto deltas, 4/4 signs, IQR, and identify any order/thermal correlation from recorded fields. Clearly mark which timing units are only user-space synchronized wall measurements.
- Verify traces for the important edge condition: small first replacement sometimes inherits previously reserved sort capacity and takes radix4 before resizing. Decide if this is only a performance artifact or also an unnecessary-memory/fallback risk. Do not change renderer defaults.
- Output `phase144_existing_evidence_audit_1.json` and `PHASE144_FIRST_FRAME_CACHE_FINDINGS.md` with an explicit answer on whether Phase143 evidence alone justifies a new GPU experiment.

## Stage 2 — New GPU tests **only if Stage 1 cannot answer**
Implement a narrow, separate-worktree, checkpointed M4 test for the highest-risk unresolved hypothesis. Use shared machine-wide `~/Library/Caches/SplatApple/gpu-experiment.lock`; check legacy GPU workers, then >=20% free system memory and **two consecutive <=30% GPU readings >=2 seconds apart** before each worker. Never run concurrent workers or upgrade the environment. Keep source and protected Fast SHA unchanged.
- Counterbalance process-first mode order, and distinguish *fresh process* from actual system-cold Metal compilation (the latter cannot be inferred without isolating driver cache).
- Stress a few scene swaps at fixed resolution, particularly highres->small->fullhd with inherited capacity and cross-mode fallback checks. Compare emitted frame hashes, trace, sustained cache, first-frame latency and long-run trend.
- Stop if the cache increase exceeds the existing bounded budget or correctness differs; retain all evidence and mark INCOMPLETE/FAIL appropriately.
- Do not add an active-sizing band without independent qualification and a clear memory/cache bound.

## Deliverables/decision
1. Evidence-grounded first-render/cache findings and actionable next experiment if warranted.
2. If no new GPU run is needed, document **NO GPU RETEST REQUIRED**, saving user time.
3. Otherwise commit script and non-overwriting run commands, execute only where Mac and resource gates are available; preserve pending status when not executed.
4. Keep `experiment/m4-phase144-cold-cache-audit` independent; no auto/default promotion, PR #1 merge, or claimed other-machine support.
