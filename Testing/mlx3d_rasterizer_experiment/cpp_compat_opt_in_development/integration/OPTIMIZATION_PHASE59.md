# Metal renderer optimization: Phases 56–60

## Final integrated result

The existing public opt-in renderer now selects automatic duplicate sizing, tile ranges and stable tile sorting. The persistent adapter uses the same defaults. Geometry and exponential-alpha math are preserved. Normal cpp/python renderer dispatch is unaffected.

The accepted algorithmic change is a stable four-bit Metal radix sort. SIMD-local ranks, block histogram prefixes and stable scatter preserve the preceding global depth order within every tile. Forty exact stable-order cases passed against NumPy, including partial threadgroups, repeated keys, sentinels and lengths over one million. Full rendering then matched the frozen pre-change renderer bitwise across the final 100K and 700K paired sequences. The public 700K Full HD dispatch also passed after promotion.

The automatic path uses radix sorting at 4M slots or more and retains MLX sorting below that threshold. It uses binary tile ranges at 1M slots or more, and the adaptive reserve for duplicate demand of at least 1M. These are algorithm-selection thresholds, not scene-capacity limits. Radix is written for Apple Silicon 32-lane SIMD groups and validated on this Mac.

| Final workload | Frozen reference median ms | Candidate median ms | Median paired ratio |
|---|---:|---:|---:|
| 4,096 / 1280×720 | 1.321 | 1.509 | 1.0031 |
| 100,000 / 1920×1080 | 27.427 | 21.254 | 0.7754 |
| 700,000 / 1920×1080 | 188.739 | 115.320 | 0.6138 |

The small-scene standalone median was higher while its paired ratio was nearly neutral; do not claim a small-scene speedup. At 700K, the median paired ratio corresponds to 38.6% lower latency. The varied-camera median was 117.298 versus 169.167 ms and its paired ratio was 0.6479. Total time across 50 varied/revisited poses fell from 11,021.97 to 8,615.40 ms. All images matched bitwise.

The 700K diagnostic after the camera survey measured tile sorting at 74.654 versus 221.586 ms and tile ranges at 0.688 versus 13.232 ms. Those are synchronized wall-time measurements after different reserve histories, not hardware GPU timestamps or a decomposition of the steady medians. Tile sorting and compositing remain the largest costs; the optimization reduces them rather than making their work disappear.

The radix comparison reported 3.71 GB peak active allocation for the two-renderer process, versus 2.32 GB in the earlier adaptive/MLX comparison. This is a material workspace tradeoff. These runs do not isolate per-renderer peak memory; allocator cache remained about 17.88 GB. No fixed memory capacity is presented as universally safe.

Environment recorded for these runs: macOS 26.7.1 (25G241), Python 3.13.15, MLX 0.32.3, NumPy 2.4.2 and mlx3d 0.3.0 in `gs-mps`. Hardware is the user-specified Mac mini M4 with 24 GB unified memory.

To select the reference policies explicitly:

```python
PersistentCppCompatRenderer(gaussians, duplicate_sort_mode="capacity",
    tile_range_mode="scan", tile_sort_impl="mlx")
```

To reproduce the final automatic candidate, pass `--candidate-sort auto --candidate-ranges auto --candidate-algorithm auto` to `bench_phase59_adaptive.py`. The `phase60_radix_*.json` filenames identify the radix experiments; their inherited benchmark phase field is 59, and `candidate_algorithm` records `radix4` explicitly.

The sections below retain the preceding adaptive-only investigation and its raw measurements; they are not the final radix results.

Development remains on `feature/cpp-compat-renderer` in the separate `splat-apple-cpp-compat` worktree. No changes were merged into main. Existing experiment results were preserved.

## Investigation and measured bottlenecks

The feature worktree initially stopped at Phase 55. The remote branch contained Phase 56 (`4dfde5d96f7dca3e1da1535915facc020ecc7f5c`), which was fetched and incorporated by fast-forward. No saved local Phase 56 result was found. Phases 57–59 independently surveyed 45 poses with translation, rotation and focal-length changes.

The primary cost is stable tile-key sorting, amplified by sorting unused reserve and retaining a high-water capacity after zooming out. Tile-range construction is a second avoidable bandwidth cost. Geometry, global depth sorting, prefix sums, duplicate expansion and compositing were also measured.

The source retains a capacity integer and requests MLX output arrays on each frame; it does not explicitly retain Metal output buffers. Stable shapes permit allocator reuse. Exact active-sized arrays reduced the diagnostic sort time but regressed end-to-end time. Allocation/shape effects are a plausible explanation, not a directly measured hardware attribution.

Representative synchronized diagnostic after the large-scene camera survey (milliseconds; dispatch and synchronization included):

| Operation | Frozen reference | Adaptive reserve + automatic ranges |
|---|---:|---:|
| Fused geometry/projection | 1.481 | 1.623 |
| Global depth sort | 2.152 | 2.063 |
| Prefix sum and gather | 1.230 | 0.837 |
| Active count reduction | 0.391 | 0.246 |
| Duplicate expansion | 12.058 | 9.159 |
| Tile-key sort, including pending mask work | 240.356 | 123.113 |
| Tile-range construction | 14.672 | 3.934 |
| Tile compositing/output | 34.662 | 35.264 |

These forced-evaluation diagnostics perturb scheduling and are not pure GPU timestamps. They must not be summed to predict normal frame time. They were collected after the varied-camera survey, when the reference retained a larger high-water reserve. Final acceptance uses uninstrumented paired timings below.

Scene activation/covariance setup is measured outside steady frame timing. The setup metric includes Python construction and cache preparation; it does not isolate only covariance GPU execution. CPU/GPU synchronization is included in end-to-end timings. The exact/bucketed candidates add a mid-frame scalar readback; the adaptive candidate does not add one after initial sizing.

## Paired end-to-end results

Each final run uses two renderers and randomized execution order, six warm-up frames and 20 steady-camera pairs. Every image is evaluated before stopping its timer. NumPy readback and bitwise comparison occur outside the timers. The reference is the exact SHA-256-verified pre-change source, not another configuration of the new implementation.

| Gaussians / resolution | Reference median ms | Candidate median ms | Median paired ratio |
|---|---:|---:|---:|
| 4,096 / 1280×720 | 1.321 | 1.509 | 1.0031 |
| 100,000 / 1920×1080 | 30.650 | 26.396 | 0.8820 |
| 700,000 / 1920×1080 | 273.868 | 222.031 | 0.8120 |

Varied-camera results include 45 poses and five shrink/regrowth revisits:

| Gaussians | Reference / candidate medians ms | Median paired ratio | Reference / candidate total ms | Reference / candidate maximum ms |
|---|---:|---:|---:|---:|
| 4,096 | 1.364 / 1.414 | 1.0205 | 82.43 / 86.28 | 4.24 / 5.13 |
| 100,000 | 27.419 / 26.735 | 0.9574 | 1567.75 / 1519.93 | 89.99 / 79.82 |
| 700,000 | 198.999 / 202.570 | 0.8868 | 12605.69 / 12059.61 | 938.05 / 806.99 |

The 700K varied-camera standalone candidate median is slightly slower even though its median paired ratio and sequence total improve. Growth and shape changes remain latency sources. Do not describe every camera transition as faster.

Absolute times differ from historical Phases 49–55. Background GPU activity and allocation state were not fully isolated during these runs. Each GPU test started only after a utilization reading of 30% or less; randomized pairing reduces but does not eliminate environment effects. These are measured paired improvements, not a guaranteed speedup for arbitrary scenes.

## Implemented changes and decisions

- Adaptive reserve uses 5% headroom plus a small minimum margin, grows and rerenders on overflow, and reduces stale high-water capacity for the next frame after a substantial demand drop. It never assumes a fixed scene capacity.
- Automatic sizing keeps the original reserve policy below 1M duplicates and selects the adaptive policy for larger counts. Automatic tile ranges keep the scan for small lists and use per-tile lower/upper-bound searches for lists of at least 1M slots.
- Exact active sorting and 1M bucket sorting remain experimental controls. Exact sizing regressed end-to-end timing. Bucket sizing created several sort shapes and produced only marginal paired gains; neither was promoted.
- Fixed 32M sizing was rejected as a production policy. The new pose survey observed 50,799,710 active duplicates; this is an observation, not a universal bound.
- Component loads replace address-space-specific vector pointer casts in the raster kernel. This fixes a pre-existing one-Gaussian compilation error when MLX supplies constant-address-space inputs.

For the 700K steady sequence, sorted slots fell from 39,428,667 to 33,120,029. For the two 32-bit duplicate arrays and 32-bit permutation alone, that is 75.7 MB less logical storage/work per frame, excluding masks and sort scratch. This is not a measured process-RSS reduction. The two-renderer process reported a 2.32 GB peak active allocation and 17.87 GB allocator cache at the end; these shared process counters cannot establish per-renderer memory savings.

## Validation scope

- Bitwise images against the exact pre-change source across the 100K and 700K final paired sequences, including all varied poses and revisits.
- Image and alpha parity for empty and 4,096-Gaussian scenes; finite, consistent one-Gaussian output with the repaired kernel. The old source cannot compile that tiny case and is not used as its oracle.
- Deliberate initial capacity of one slot, fully culled views, nonzero background, nonmultiple image dimensions, actual changed scene data, invalid policy rejection and source-hash rejection.
- Installed MLX3D source verification remains enabled. The repository source SHA is updated explicitly after edits; no integrity check is bypassed.

These gates establish regression parity with the previous validated renderer. They do not establish new universal pixel equivalence with the original C++ renderer. No differentiable-training changes were made.

## Reproduction

Use conda environment `gs-mps` in the feature worktree. Check GPU utilization before each GPU command and wait if it exceeds 30%. Preserve the reference:

```sh
git show 4dfde5d96f7dca3e1da1535915facc020ecc7f5c:Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py > /tmp/splat_fast_reference.py
```

Run `test_phase58_regression.py --reference /tmp/splat_fast_reference.py` and `bench_phase59_adaptive.py --reference /tmp/splat_fast_reference.py --out NEW_RESULT.json` from the integration directory scripts. The benchmark accepts `--n`, `--width`, `--height`, candidate sizing/range/sort options and refuses to overwrite results. Raw JSON files at the worktree root retain every paired measurement and per-frame duplicate count.

Current source SHA-256: `71a179154d8f216b3ef247271bea47cc277afd74dc27a0cd1d7f0696e332d89b`.
