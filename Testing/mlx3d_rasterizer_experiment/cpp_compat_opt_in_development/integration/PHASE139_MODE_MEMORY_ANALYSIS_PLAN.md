# Phase 139 — CPU-only audit of completed Phase138 memory qualification

**Mac report status:** user terminal reported `RESULT PHASE138 MODE_ISOLATED_MEMORY PASS` for `phase138_mode_isolated_memory_2.json`. This completes the original Phase138 v1 run; no repeat 60-worker GPU qualification is necessary. **Numeric memory comparisons are pending receipt/analysis of the raw report.**

## Inputs and provenance

The original report is located locally at:

`/Users/Shared/Main Hard Drive/My Projects/Projects In Progress/Applications/Gaussian Splatting/splat-apple-phase138-memory/phase138_mode_isolated_memory_2.json`

The file is not automatically available in Google Drive. The report was produced by the original Phase138 v1 harness; **do not use the later `--resume-from` v2 harness on these v1 artifacts.**

New script:
`analyze_phase138_mode_isolated_memory.py` (stdlib-only; no MLX import, GPU launch, or source modifications).

Drive mirror: https://drive.google.com/file/d/11cnLlZ0cvTjDEVDuoLUVQr9m9TjAoCzb/view

Run from the existing Phase138 Mac worktree, after fetching the separate analysis branch:

```bash
cd "/Users/Shared/Main Hard Drive/My Projects/Projects In Progress/Applications/Gaussian Splatting/splat-apple-phase138-memory" || exit 1
git fetch https://github.com/KingdomDesignsConsulting/splat-apple.git experiment/m4-phase139-memory-review || exit 1
test ! -e ./analyze_phase138_mode_isolated_memory.py || { echo "Analyzer already exists; refusing overwrite"; exit 1; }
git show FETCH_HEAD:Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/analyze_phase138_mode_isolated_memory.py > ./analyze_phase138_mode_isolated_memory.py || exit 1
python -m py_compile ./analyze_phase138_mode_isolated_memory.py || exit 1
python ./analyze_phase138_mode_isolated_memory.py \
  --input phase138_mode_isolated_memory_2.json \
  --out phase139_mode_memory_analysis_1.json
```

Output files (exclusive, non-overwriting):
- `phase139_mode_memory_analysis_1.json`
- `phase139_mode_memory_analysis_1.md`

The user may also upload the original Phase138 raw JSON into the existing Splat-Apple Drive folder for direct remote analysis. Once the report is accessible, inspect the numeric memory and elapsed-time differences and decide whether a further GPU experiment is justified.

## Analysis gate

The analyzer requires:

1. Phase138 status PASS; exactly **60** workers, with all 3 workloads, 4 blocks, and 5 modes represented exactly once.
2. Every worker exit code 0, nested report PASS, matching workload/block/mode and original source/default invariants.
3. Fast source SHA `56a2b035…e31710` and host-camera adapter SHA `2554ff81…da6425e` (candidate branch).
4. Every recorded pre-worker GPU admission `<=30%`, with a valid timestamp and at least `20%` free system memory.
5. Preserved nonempty GPU gate-attempt history for all 60 workers ending in an admitted sample.
6. The expected first/warmed/restored image hashes, timeline stages and numeric memory counters; **bitwise equality across modes within each scene/camera/block**.
7. Nonnegative, finite memory counters, monotonic cumulative MLX peaks and Darwin RSS high-water, no `active+cache` sample above the original 13 GiB cap.
8. Immutable user source: the input file is read and hashed but not modified; output paths are exclusive and fail if already occupied.

Per workload and mode, calculate medians of four independent worker values, **block-matched candidate-minus-auto** differences and counts of blocks with lower memory. Compare MLX peak, post-release cache and live allocations, Darwin RSS high-water and warmed-frame times. Record adaptive selection decisions where present.

**Caveats:** fresh-process MLX peak/cache and RSS high-water include scene, runtime and allocator overhead. They are *not pure kernel memory* and cannot be added together; execution was sequential and single synthetic scene seed was used, so timing differences are descriptive and not standalone performance qualification. No automatic source/selector promotion or PR merge.

## Current status

- Original Phase138 v1 user execution: **PASS** (terminal evidence).
- Phase139 analyzer: **committed / uploaded / Python syntax checked / synthetic fail-closed tests passed**.
- Phase139 analysis of actual raw Phase138 JSON: **PENDING**; cannot conclude which sorting mode uses less memory without examining that file.
- Protected original CPP-compatible renderer: **CONDITIONALLY READY for qualified M4 source checkout**.
- Draft PR #1: **unmerged**.
