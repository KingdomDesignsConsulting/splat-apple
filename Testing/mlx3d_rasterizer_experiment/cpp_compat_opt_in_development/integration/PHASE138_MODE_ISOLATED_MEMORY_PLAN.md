# Phase138 — M4 mode-isolated memory qualification (Mac PENDING)

## Purpose
Phase 128 host-side W2C finite checking passed **all 65 contracts** and exact bitwise
output parity across 12 workers. Its only replicated frame-time advantage was on
small workloads (eager −0.173 ms, lazy GPU-produced W2C −0.050 ms, 4/4 processes
favorable for each). Large-workload effects were mixed. Keep this camera
candidate experimental and do not claim a broadly faster renderer.

The Phase 130 and 137 memory counters were measured in processes containing
multiple renderers. They cannot attribute MLX cache or process RSS to a single
mode. The next justified test is **one sorting/selection mode per fresh process**.

## Design
Script: `run_phase138_mode_isolated_memory.py` (also mirrored unchanged in
the Splat-Apple Drive root). Based on experimental branch head `b41c45f`;
the original M4-conditional PR #1 branch is not changed.

For each of 3 fixed synthetic workloads and 4 blocks (60 workers total),
launch fresh interpreter processes within the pinned disposable runtime,
one per mode:
- `auto` (existing adapter default)
- `selective_final3`
- `scatter512` (explicit `selective_final3_scatter512`)
- `active` (duplicate_sort_mode=`active`)
- `adaptive` (separate `ExperimentalAdaptiveRenderer`, **explicit attested M4 only**)

The same input seed (12345), camera sequence, frame sizes, and memory budgets
are used within a block. Sample MLX active, peak and cache and Darwin process
RSS high-water at scene creation, construction, first frame, four warm frames,
small-scene replacement, restoration, and engine release. Compare full RGB
SHA256 of every corresponding frame to the isolated `auto` control after
the workers finish. A mismatch is FAIL.

Safeguards: GPU utilization must be <=30% **before every worker**; bounded
cooldown only. `memory_pressure -Q` must report >=20% free; MLX cache limit
6 GiB, observed MLX active+cache must stay <=13 GiB. Every worker has an
independent log; output/report paths are exclusive and non-overwriting.
The user must supply `--attest-m4` explicitly. Do not run workers concurrently.

## Acceptance and interpretation
PASS requires all 60 workers (or all explicitly requested workload/block
combinations), matching source/default invariants, bitwise cross-process RGB
for each camera, and valid memory counters. The report is **incomplete** if a
resource gate remains busy; retain its partial evidence and rerun with a fresh
filename. The 4-block full matrix is needed for the broad Phase138 claim.

Per-mode fresh processes improve attribution but still contain Python, MLX,
the scene, compiled kernels and allocator caches. These are observed **total
per-process memory footprints**, not exact kernel-owned allocations. They
must not be added across samples or equated to Metal residency. Cold-start
measurements inherit existing system/driver caches. The scene uses one fixed
seed and is not a generalization to other densities.

No renderer source, Fast hashes, sorting defaults, or PR #1 state may change
as a consequence of this experiment alone. Do not accept an optimization
merely because one process has a lower peak.

## Mac execution
Run from an isolated worktree on Mac mini M4 in the `gs-mps` environment:
```sh
conda run --no-capture-output -n gs-mps python -m py_compile \
  Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase138_mode_isolated_memory.py

conda run --no-capture-output -n gs-mps python \
  Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase138_mode_isolated_memory.py \
  --attest-m4 --blocks 4 --out phase138_mode_isolated_memory_1.json
```
Report and artifact directory remain local until the user shares/uploads
them. Do not mark the run PASS until Mac executes and the JSON is inspected.

## After the result
If specific modes show robust mode-isolated savings, qualify another set of
scenes/seeds and validate dispatch/first-render parity before proposing
integration. Otherwise prioritize the separate first-render bottleneck;
avoid additional speculative kernel edits.

## Safe follow-on harness revision (v2)

A dedicated branch `experiment/m4-phase138-resumable-gate` adds a 4-second
post-worker settle period, requires two consecutive <=30% device-utilization
readings, and creates a checksum-verified checkpoint for each successful worker
so a later new-output run can resume without repeating verified blocks.

This is **not a retrofit of the currently running Phase138 v1 process**.
The existing v1 `_2_artifacts` folder lacks the provenance needed to
automatically qualify reused workers. Preserve the v1 evidence and do not pull
branch updates into its live worktree. Full usage, provenance conditions,
failure modes and the eight passing CPU-only tests are in
`PHASE138_STABLE_GATE_AND_RESUME.md`. Actual M4 GPU validation is pending.
