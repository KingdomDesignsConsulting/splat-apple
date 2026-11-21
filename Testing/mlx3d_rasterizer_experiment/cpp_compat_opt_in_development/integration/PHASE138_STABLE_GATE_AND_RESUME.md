# Phase 138 harness v2 — stable GPU entry and fail-closed resume

**Status:** implemented and CPU-only tested; **NOT YET M4 GPU-validated**. This work lives only on `experiment/m4-phase138-resumable-gate`. The original M4-qualified CPP renderer, source digests, sorting defaults, and draft PR #1 remain unchanged.

## Motivation

The current Phase 138 Mac trace repeatedly showed `ioreg` at 92–97% immediately after a worker, followed by 0–29% ~10 seconds later. This is compatible with residual GPU execution, driver accounting, or another system workload; **`ioreg` does not identify the owning process**, so no cause is assumed. All completed observed workers entered at or below 30%.

## Revised pre-worker GPU policy

- Require >=20% free system memory as before.
- After a *newly executed* worker, allow **4 seconds of settling** before the next GPU check. This is a polling delay, not a utilization override.
- Require **two consecutive readings <=30%** separated by 2 seconds. Any reading >30%, malformed reading, or unavailable reading prevents launch.
- A busy sample resets the streak; after a busy reading, retry after 10 seconds, bounded to 12 total readings. Failure returns INCOMPLETE, never launches the worker.
- Store all samples, the final two accepted readings, and the actual launch reading in each completed worker's checkpoint.

These are snapshots of device-wide utilization, not proof of thermal isolation. The original hard threshold is not relaxed.

## Durable checkpoints and resume

Each fresh run creates a dedicated artifact folder with:

- `phase138_manifest.json`: exact runner/helper/adaptive SHAs, input matrix, source/default invariants, caller package inventory digest, Python/platform and M4 attestation.
- `phase138_active.lock`: held while a parent is executing. A resume must reject a source still actively locked.
- `<workload>_<block>_<mode>.json` and `.log`: original worker evidence.
- `<workload>_<block>_<mode>.checkpoint.json`: written **only after exit code zero, PASS JSON and matching source/default and worker identity**, with a pair of admitted GPU samples, >=20% free-memory reading, worker JSON/log SHA-256s, and completion time.

A resumed run **always uses a new** `--out` and new artifact directory. It reads the source folder without modifying it. A worker is reused only if its checkpoint and both files verify, the source run is inactive, the exact matrix/code/runtime inventory matches, and its original pre-worker 30%/20% checks are preserved. Copied evidence contains provenance to the original checkpoint SHA. Missing checkpoints are **rerun**, invalid/tampered checkpoints cause a fail-closed error, and incomplete matrices cannot be marked PASS.

### Original running Phase138 v1

The existing `phase138_mode_isolated_memory_2_artifacts` was created by the **earlier harness, without these checkpoints**. This change cannot modify its already-running parent or retroactively prove pre-worker GPU admission from its worker JSON. Do **not** pull into that active worktree. If the v1 run completes, keep and analyze its output as usual. If it stops incomplete, preserve the old artifacts as evidence; they cannot be automatically reused for a qualified PASS without separately verified gate evidence. Future v2 runs will be resumable.

## Commands, only after current Mac run completes

Create an independent worktree:

```bash
cd "/Users/Shared/Main Hard Drive/My Projects/Projects In Progress/Applications/Gaussian Splatting/splat-apple-cpp-compat" || exit 1
git fetch https://github.com/KingdomDesignsConsulting/splat-apple.git experiment/m4-phase138-resumable-gate || exit 1
git worktree add --detach "../splat-apple-phase138-resumable" FETCH_HEAD || exit 1
cd "../splat-apple-phase138-resumable" || exit 1
conda activate splat_apple
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/test_phase138_harness_cpu.py || exit 1
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase138_mode_isolated_memory.py --attest-m4 --blocks 4 --out phase138_mode_isolated_memory_3.json
```

If **that v2 run** is interrupted, do not overwrite it. Use:

```bash
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase138_mode_isolated_memory.py --attest-m4 --blocks 4 --resume-from phase138_mode_isolated_memory_3_artifacts --out phase138_mode_isolated_memory_4.json
```

Use later free output numbers if those names are occupied. A resumed result retains both the original and newly executed worker records.

## Tests and limits

8 CPU-only tests passed:
1. Busy/low/busy/low/low sequence requires two good samples.
2. Repeated busy samples fail closed.
3. A malformed above-threshold reader result fails closed.
4. JSON checksum tampering rejects checkpoint.
5. Legacy artifacts lacking manifest/checkpoints cannot resume.
6. An active lock prevents concurrent resume.
7. Fully completed run resumes all five verified workers without GPU reruns.
8. Partial simulated worker failure resumes completed workers, reruns incomplete workers, and completes the matrix.

No real M4 GPU jobs were executed by this change; the Mac run must establish actual behavior and results. Never claim Phase138 PASS from these CPU tests.
