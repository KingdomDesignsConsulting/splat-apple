# Phases 140–143 combined engineering report

**Latest status: qualification complete.** See the [continuation results](PHASE140_143_CONTINUATION_RESULT.md) for the passed Phase 140–143 matrices and scoped decision. The initial interrupted session is preserved below.

November 20, 2026 · Mac mini M4 · `experiment/m4-phase140-density-memory` · base `2a00d57`

The shared GPU lock, stable admission and verified resume passed real Mac smoke testing. The Phase 140 GPU matrix remained incomplete after two bounded attempts because the GPU repeatedly crossed 30% between samples. Five of 36 workers completed and are available to resume. Neither run launched a worker above the threshold. No active-sizing model was accepted, and the combined candidate remains experimental.

| Phase | Status | Evidence |
|---|---|---|
| 140 | **INCOMPLETE** | Five workers, including both sparse-small pairs on seeds 12345 and 12346. The moderate-small auto worker has no completed pair. Full HD and high-resolution workers are pending. All completed sparse-small pairs matched RGB bitwise through changes in camera and scene. |
| 141 | **INCOMPLETE** | CPU analysis ran on the verified pairs. Both seeds reject active sizing for sparse small. Larger break-even evidence is missing. No active bands were enabled. |
| 142 | **INCOMPLETE, CPU function tests PASS** | Separate opt-in renderer and five selector tests cover capacity endpoints, prior-frame availability, shape, tile domain, scene size and unknown-device fallback. Mac execution is pending. |
| 143 | **NOT RUN** | The executable matrix includes the 65 original contracts, five native dispatch boundary cases, and five mode-isolated policies across three resolutions and four exactly counterbalanced blocks. The checker verifies trace selection, bitwise RGB, memory and timing. |

## Available performance and memory evidence

The completed sparse-small fixture uses 2,949 Gaussians with smaller projected splats and a wider placement range. Control used about 18.5K–18.6K sorted slots with about 13.8K active slots, leaving 25.5% unused. Active sizing was **+0.782 ms and +0.729 ms** slower across the two seeds, a median **+0.755 ms**. It reduced the mode-isolated MLX peak by **0.219 MiB** and after-release cache by **0.094 MiB**. All expected outputs, including the replacement scene and restored original, matched the control. Saved traces show the MLX route at this size. Full per-frame measurements, counts, first-render timings, selection data and process/MLX memory are in the worker JSON and logs. Process counters include the runtime, allocator and scene. They cannot measure individual kernels.

The Phase 139 evidence is relevant but separate. Active sizing reduced peak MLX memory by 218.7 MiB at Full HD and 333.2 MiB at high resolution. The Full HD result also had about 3.75 GiB more retained cache and 4.90 ms higher median warmed time in that process-isolated experiment. Phase 134 separately showed a warmed Full HD gain. These differences rule out assuming either a general speedup or that active sizing and adaptive scatter gains add together.

The Phase 141 screen requires both seeds of an observed density to save more than 1% and 0.5 ms, avoid peak growth, and add at most 256 MiB of allocator cache. At least two densities in the same resolution must meet the screen before that resolution can contribute trial bands. These are screening criteria for a later independent qualification, not a proven break-even rule. Any experimental sizing band uses a previously observed active count for the current scene and resolution. Replacement resets history. There is no extra GPU readback to make the decision. Host free memory is an admission check, not a policy input.

The intended Phase 142 fallback uses existing automatic capacity behavior. Experimental sorting is eligible only for an explicitly attested M4, 1920×1080, 8,160 tiles and 32–34 million actual sort slots, inclusive. The protected native eligibility check still applies. If exact active sizing moves the sort below the band, the experimental selector returns to `auto`. Small scenes, unknown devices and other shapes fall back. No trial band is currently enabled. There is no basis to recommend integrating the new selector into PR #1. The prior adaptive policy remains the best evidenced scoped alternative for the previously tested Full HD fixture.

## Locking, admission and evidence

Shared advisory lock: `~/Library/Caches/SplatApple/gpu-experiment.lock`. The runner scans for known legacy Python experiment processes before and after acquisition. No such interference was observed. Unrelated GPU activity remains subject to the utilization gate. The lock is held across setup, admission, worker execution and parity analysis; each worker requires the inherited open descriptor before importing GPU libraries. Closing the owning descriptor releases the lock after any inherited worker exits. The file persists. A destination lock and an inactive source check protect resume. Checkpoints are published atomically, with original timestamps/paths, file hashes, manifest and admission history retained. Reports and logs remain exclusive writes.

The CPU suite passed 8 Phase 138 tests, 15 lease/checkpoint tests and 5 policy tests. The termination and inherited-descriptor contention tests were CPU simulations. Physical M4 validation deliberately paused after one real worker, then reused that checkpoint and ran only the missing worker. The two GPU smoke outputs matched bitwise.

| Run | New verified workers | Reused | Busy readings | Peak accepted utilization |
|---|---:|---:|---:|---:|
| Smoke pause | 1 | 0 | 3 | 28% |
| Smoke resume | 1 | 1 | 6 | 28% |
| Density attempt 1 | 2 | 0 | 9 | 30% |
| Density attempt 2 | 3 | 2 | 21 | 30% |

Every launch required two consecutive readings at or below 30%, spaced two seconds apart, and at least 20% free system memory. Four-second settling and ten-second busy cooldowns were retained. Both density attempts reached the bounded 12-sample limit and exited without launching the next worker. GPU memory results were captured in a pinned, disposable Mac arm64 Python 3.13 / MLX and Metal 0.32.4 / verified MLX3D 0.3.0 environment. No protected Fast source, qualified public API or existing default was changed.

Commits: [machine-wide lease and physical smoke](https://github.com/KingdomDesignsConsulting/splat-apple/commit/0e1cdf9), [policy and initial density evidence](https://github.com/KingdomDesignsConsulting/splat-apple/commit/5b4d72d), [resumed artifacts and continuation](https://github.com/KingdomDesignsConsulting/splat-apple/commit/46d8d0b). [Draft PR #1](https://github.com/KingdomDesignsConsulting/splat-apple/pull/1) remains unmerged. The source reports and raw artifacts are under `qualification_evidence/phase140/`. Google Drive upload of the source/evidence archive was rejected by automatic approval review; the evidence is available in GitHub.

## Resume and continue

This command resumes from the latest archived checkpoint directory. The batch creates a new output directory. It starts Phase 143 only after Phase 140 passes, Phase 141 analysis is complete and Phase 142 CPU checks pass. All GPU launches repeat the lock and stable admission checks.

```sh
cd '/Users/Shared/Main Hard Drive/My Projects/Projects In Progress/Applications/Gaussian Splatting/splat-apple-phase140-density-memory' && B=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration && bash "$B/continue_phase140_143.sh" "$B/qualification_evidence/phase140/phase140_density_2_artifacts"
```

If Phase 140 stops again, supply the newest batch's `phase140_artifacts` directory to the same command. If Phase 143 stops after Phase 140 completed, use the balanced entry point in the same worktree with a new report filename:

```sh
B=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration
conda run --no-capture-output -n gs-mps python "$B/run_phase143_balanced.py" --phase 143 --attest-m4 --resume-from '<batch directory>/phase143_artifacts' --out phase143_resumed_NEW.json
```
