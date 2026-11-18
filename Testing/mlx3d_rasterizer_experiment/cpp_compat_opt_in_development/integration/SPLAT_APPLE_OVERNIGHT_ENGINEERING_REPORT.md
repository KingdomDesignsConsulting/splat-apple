# Splat-Apple overnight engineering report

November 18, 2026 · Mac mini M4, 24 GB unified memory

The original CPP-compatible renderer remains **CONDITIONALLY READY** for forward rendering from the documented M4 source checkout. Draft PR #1 remains open and unmerged. The strongest new end-to-end result was **7.2% lower warmed Full HD latency** with the experimental selector versus existing auto. All four processes used the already validated scatter512 implementation for that fixture. Separately, exact active sizing improved Full HD by about **6.7%**, with a material small-scene regression. These are independent comparisons against auto and the gains cannot be added.

The initial profiling batch produced three valid stage reports and exposed a Phase 128 harness setting error. The corrected batch passed all 42 workers. Its highest admitted utilization was 29%; busy readings triggered bounded cooldown. The extended qualification then passed fresh loader, legacy, API and full Phase 125 performance checks alongside the retained Phase 121–126 aggregate. A same-run resume reused all five verified checkpoints. No protected renderer source, adapter digest or sorting defaults changed.

## Minor goals

| Phase | Status | Outcome |
|---|---|---|
| 127 | PASS | Release scope, software provenance, API, legacy behavior, packaging and prior evidence audited at `f645298`. |
| 128 | INCOMPLETE | Measurements completed. Host matrix validation took 0.003–0.011 ms versus 0.222–0.318 ms for the GPU check. Full contract and render qualification of the candidate remains open. |
| 129 | INCOMPLETE | Twelve process-first workers passed parity. Active sizing improved first render in all tested large-workload blocks, with a warmed small-scene penalty. No general startup optimization accepted. |
| 130 | PASS | Live allocations remained flat at the matched scene across six update cycles per workload, and dropped on release. Cache stayed reusable. No reduction is attributed to any mode. |
| 131 | PASS | Fail-closed fixture, standard and extended runs, and checkpoint resumes passed. Fresh Phase 125 Full HD: explicit scatter512 beat selective final3 by 2.508 ms mean across four process comparisons, with all four favoring scatter512. |

## Renderer-wide performance

At Full HD and high resolution, mask/sort measured 72.63 and 116.48 ms, roughly 60% and 62% of the corresponding uninstrumented frames. Raster was the next largest at 36.59 and 49.10 ms. Profiling used host walls with synchronized MLX evaluation. Driver compilation, allocation and host/GPU wait effects are included where they occur. Small-workload instrumentation introduced proportionally larger overhead.

The active-size candidate shortened the Full HD sort from 33.15M to 31.57M slots and eliminated the unused tail mask at the cost of an earlier scalar synchronization. The warmed median paired change was −8.09 ms at Full HD, −3.33 ms at high resolution and +0.326 ms on small scenes. Each workload used four fresh processes and several camera positions. Exact outputs and traced path checks passed. Retain as a large-workload research candidate and reject universal use. Tests recorded allocator and process peaks for the whole comparison process, so memory impact by mode remains open. See [performance decision](PHASE134_PERFORMANCE_DECISION.md).

## Adaptive GPU scheduling

The experimental API considers the actual sort buffer at the composite boundary. It selects scatter512 only for an explicitly attested M4 running 1920×1080 with 8,160 tiles and a 32–34M-slot buffer. Otherwise it uses existing auto. The baseline path was observed as MLX for small and radix4 for high resolution. The full comparison included auto, radix4, selective final3, explicit scatter512 and adaptive, with exact RGB parity, host threshold checks, traced dispatch, camera transitions and replacement scenes.

Adaptive was about 8.39 ms faster than auto at Full HD, with a four-process range of −8.87 to −8.13 ms. It was effectively even with explicit scatter512 there. The high-resolution fallback had a +0.80 ms paired median difference with mixed direction across blocks. Mode-isolated memory, standalone dispatch overhead, long-run retention, thermal telemetry and representative scenes throughout the eligibility band remain open. The observed band midpoint is qualified for this investigation. The experimental selector is **incomplete for integration**. See [adaptive decision](PHASE137_ADAPTIVE_SCHEDULING_DECISION.md).

## Evidence and release decisions

- [Phase 127 release hardening commit](https://github.com/KingdomDesignsConsulting/splat-apple/commit/f645298) and [Phase 131 runner commit](https://github.com/KingdomDesignsConsulting/splat-apple/commit/0063302).
- [Experimental scripts and profiling evidence](https://github.com/KingdomDesignsConsulting/splat-apple/commit/4f91801) and [experimental branch](https://github.com/KingdomDesignsConsulting/splat-apple/tree/experiment/m4-renderer-perf-adaptive-2026-11).
- Benchmark outputs are in `overnight_results/`, with original reports, exact worker commands, raw logs and provenance in `qualification_evidence/overnight/`. The qualification uses Python 3.13.15, macOS 26.7.1 arm64, MLX/MLX-Metal 0.32.4, NumPy 2.4.2 and verified MLX3D 0.3.0. Disposable environments left gs-mps intact.
- Installed Fast SHA-256 `4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`. Repository Fast SHA-256 `56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710`. Adapter SHA-256 `e8f9079908e0d08b28f4af125ed544c76d25b23ed572a273b8b20316db7192ec`.
- Reports and compact supporting evidence are mirrored in the [Splat-Apple Drive folder](https://drive.google.com/drive/folders/1YjulpBQLJhrW09ge00jLKK3TW2-woqyN). GitHub holds the authoritative source, commits and raw archives.
- [PR #1](https://github.com/KingdomDesignsConsulting/splat-apple/pull/1) tracks the original renderer. The protected M4 baseline is CONDITIONALLY READY. Active sizing has a qualified fixture-specific result and a known small-scene regression. The host validation prototype and adaptive policy need further qualification. No candidate is ready for a release integration PR. Broader release and default promotion are unsupported. CI pass and independent approval are unestablished.

## Ranked continuation

1. Qualify full host-side matrix validation against all 65 contracts, lazy matrices and supported dtypes. Confirm paired renderer latency with the safety checks intact. Low implementation risk and a measurable small-scene target.
2. Replicate active sizing and adaptive routing on varied scene density/cameras with mode-isolated memory. Establish actual break-even points and test the eligibility-band edges with real dispatch. Medium risk, with the largest measured gains.
3. Split tail-mask and stable sort timing with separately evaluated inputs and verify exact sentinel ordering. Target the dominant large-workload cost with a single isolated change.
4. Profile effect of a cache budget and dense compositing separately. Track thermal state, settled memory and tail latency before choosing another kernel or workspace change.

The original M4 matrix is complete. To reproduce the two candidate comparisons from the experimental worktree, use a fresh output name. The runner gates every GPU worker at 30%, checks memory pressure and preserves logs.

```sh
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_overnight_renderer.py --phase 133 --phase 137 --out phase133_137_replication_NEW.json
```
