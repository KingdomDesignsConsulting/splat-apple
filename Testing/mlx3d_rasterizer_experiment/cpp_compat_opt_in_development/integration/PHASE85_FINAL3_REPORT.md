> **Historical Phase 85 report (pre-integration).** The final3 algorithm was subsequently integrated as an **opt-in per-instance** `tile_sort_impl="selective_final3"` setting in Phase 89; defaults still have not changed. See [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) for Phases 86–89, later user-Mac results, and current validation status. This report's statements about never importing the candidate in production describe the Phase 85 snapshot only.

# Phase 85: experimental stable three-category final radix pass

Implemented and benchmarked outside production on the Mac gs-mps environment. The permutation-only candidate reduces standalone Full HD tile-sort latency by about 9% in the paired run and reduces 700K Full HD renderer latency by about 4–5%, repeated in a fresh process. Production source, adapter defaults, integrity checks, installed packages and PR draft status are unchanged.

## Prior findings reviewed

- Phase 82 measured a four-pass 8160-tile sort at 63.935 ms, with the final rank/histogram at 6.894 ms and final scatter at 7.559 ms in its separately synchronized diagnostic. Those stage numbers are not pure GPU timestamps and must not be summed to predict normal sort latency.
- Phase 83 confirmed the extra pass at the 4096-tile boundary and roughly 63–65 ms at 8160 tiles, with or without sentinels.
- Phase 84 is a design-only rejection: its executable checks the production sorter against NumPy but implements no binary candidate. Its conclusion is correct: real tiles 4096..8159 and UINT_MAX cannot share a final category. Its status is not a measured performance result.

Base feature-branch revision: `7f42e98e5c9e3bd5d5adb2569e9eba6257704cec`.

## Implementation and ordering argument

`radix_final3_experiment.py` is never imported by production. It reuses the hash-verified production kernels for the earlier four-bit digits, then replaces only the final rank/histogram and scatter. Full HD still uses **four passes**, not three: shifts 0, 4 and 8, then the specialized final pass for shift 12.

The final categories at 8160 tiles are:

| Category | Keys |
|---|---|
| 0 | Real tile IDs 0..4095 |
| 1 | Real tile IDs 4096..8159 |
| 2 | UINT_MAX sentinel only |

Earlier stable LSD passes order the lower bits. SIMD-exclusive ranks preserve lane order, subgroup prefixes preserve subgroup order, and block prefixes preserve block order. The final stable partition therefore orders the high digits while retaining lower-bit ordering and original order within equal real keys and within all sentinels. Sentinels remain UINT_MAX in the original input; the input array is never rewritten.

Specialization is used only when the top digit of every valid real key is 0 or 1. Other domains, including 4095, 8193 and 3600 tiles (1280×720), delegate to the unmodified production sorter. The same input contract as production applies: one-dimensional uint32 real tile keys below num_tiles, or UINT_MAX. There is no sentinel-fraction assumption and no added data-dependent CPU readback.

Two variants isolate the changes: `with_keys` retains final key and permutation outputs; `indices_only` writes only the permutation, which is all the sorter API returns. At 33,154,238 slots the latter avoids 132,616,952 bytes (126.47 MiB) of final key output/writes. The histogram and its prefix shrink from 16 rows to 3; threadgroup count storage shrinks from 128 uints to 24. These are logical buffer/work reductions, not isolated RSS measurements.

Renderer experiments use a scoped in-process override that is restored even on exceptions. They do not patch files, installed modules, public defaults or integrity declarations. This test hook is single-threaded experimental infrastructure, not a production API.

## Correctness

348 cases × two candidates passed exact NumPy stable permutation and production-permutation comparisons. Cases cover lengths 0, 1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 4096, 65537 and 1,000,003; masked/unmasked inputs; all-sentinel and duplicate-only arrays; sorted and reversed data; strided input; the explicit Phase 84 alias counterexample; and fallback domains.
Both candidates were also compared against every production permutation entry during each large timed benchmark round. Renderer checks compared the actual uint32 bit patterns of float32 RGB and alpha, not just numeric allclose or floating-point equality. All six scene/resolution configurations passed at exact and padded capacities, across 12 moving-camera poses, after actual scene-data changes, and for fully culled nonzero-background views. The fresh-process 700K confirmation also passed every bit check.

This establishes parity against the current validated renderer. It is not a new universal equivalence claim against the original C++ renderer.

## Standalone sort results

8160 tiles; three warmups per mode and 12 randomized paired rounds. GPU completion is inside timers; correctness checks are outside. Medians in milliseconds:

| Slots | Sentinels | Production radix4 | Final3 with keys | Final3 indices only | Indices-only paired ratio |
|---|---|---:|---:|---:|---:|
| 1,000,003 | none | 4.154 | 3.928 | 3.827 | 0.9275 |
| 1,000,003 | 1/7 | 2.318 | 2.187 | 2.134 | 0.9134 |
| 8,000,003 | none | 15.253 | 14.331 | 14.030 | 0.9196 |
| 8,000,003 | 1/7 | 15.600 | 14.711 | 14.230 | 0.9094 |
| 33,154,238 | none | 64.246 | 59.876 | 58.605 | 0.9100 |
| 33,154,238 | 1/7 | 64.112 | 59.686 | 58.211 | 0.9109 |

## Renderer results

Matched camera, scene, capacity policy and slot count. Both paths explicitly select radix sorting; this isolates the sort change from automatic backend thresholds. Image and alpha GPU completion are timed; the context hook and bit comparisons are outside timers. Initial matrix: three warmups and 12 randomized paired rounds per layout.

| Gaussians / resolution | Layout | Production ms | Final3 ms | Median paired ratio |
|---|---|---:|---:|---:|---:|
| 4,096 / 1280×720 | exact_unmasked | 5.172 | 5.653 | 0.9666 |
| 4,096 / 1280×720 | headroom_masked | 2.223 | 2.270 | 1.0416 |
| 4,096 / 1920×1080 | exact_unmasked | 3.173 | 3.100 | 0.9660 |
| 4,096 / 1920×1080 | headroom_masked | 3.035 | 3.029 | 0.9754 |
| 100,000 / 1280×720 | exact_unmasked | 10.346 | 10.428 | 1.0121 |
| 100,000 / 1280×720 | headroom_masked | 10.378 | 10.884 | 1.0331 |
| 100,000 / 1920×1080 | exact_unmasked | 21.037 | 19.923 | 0.9500 |
| 100,000 / 1920×1080 | headroom_masked | 21.554 | 20.201 | 0.9428 |
| 700,000 / 1280×720 | exact_unmasked | 58.306 | 58.532 | 1.0020 |
| 700,000 / 1280×720 | headroom_masked | 59.402 | 59.153 | 0.9971 |
| 700,000 / 1920×1080 | exact_unmasked | 113.717 | 108.617 | 0.9581 |
| 700,000 / 1920×1080 | headroom_masked | 116.264 | 111.898 | 0.9589 |

The small-scene results are noisy and do not establish a gain. The unsupported 720p fallback has no consistent benefit; the 100K headroom case regressed by a median paired 3.3% in this run. Do not present this as an improvement for all domains or scene sizes.

Fresh-process confirmation, 700K / 1920×1080, 24 paired rounds:

| Layout | Production ms | Final3 ms | Median paired ratio | Within-run bootstrap 95% interval |
|---|---:|---:|---:|---|
| exact_unmasked | 111.163 | 106.394 | 0.9549 | [0.9471, 0.9616] |
| headroom_masked | 114.922 | 109.808 | 0.9516 | [0.9488, 0.9598] |

## Scope, limitations and decision

- GPU tests started only after utilization readings at or below 30%. Recorded pre-test readings were 14%, 0%, 5% and 0%. Conditions during the runs were not fully isolated from other system activity.
- These are warm wall-time measurements, including Python/MLX dispatch and synchronization. There are no new hardware timestamp or cold-start claims.
- Bootstrap intervals describe within-run pairs and do not cover between-run or hardware uncertainty. The separate confirmation adds evidence for the target 700K Full HD case; it is not exhaustive hardware validation.
- The kernels retain the existing Apple Silicon assumption of 32 lanes per SIMD group and 256-thread blocks. Other GPU architectures have not been validated.
- Process memory counters in the JSON include all compared modes and allocator cache; they do not isolate per-candidate peak memory.

Decision: the sentinel-safe candidate is implemented, correctness-tested and has a repeatable target-workload improvement. Keep it experimental for architect review. Production source, source hashes and default selection remain unchanged; PR #1 remains draft and is not merged.

Environment: python 3.13.15, macos 26.7.1, mlx 0.32.3, numpy 2.4.2, mlx3d 0.3.0.

## Mac gs-mps execution

From the separate feature worktree:

```bash
conda run --no-capture-output -n gs-mps bash Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase85_final3.sh
```

The runner echoes each shell command, creates a new timestamped result folder, checks the 30% GPU limit before each stage and refuses to overwrite existing output. A busy/missing GPU reading exits with code 75 before the next GPU stage. For an additional target confirmation:

```bash
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/bench_phase85_final3.py --mode render --scenes 700000 --resolutions 1920x1080 --rounds 24 --out NEW_CONFIRMATION.json
```

Check GPU utilization immediately before that direct Python command. The committed evidence is under [phase85_results](phase85_results/): [correctness](phase85_results/correctness.json), [sort](phase85_results/sort.json), [renderer matrix](phase85_results/render.json), [confirmation](phase85_results/render_confirm.json). Each benchmark records production and experiment hashes, raw paired samples and software versions.

Production hashes:

- `Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py`: `bfc0f34a3e1a3157178518509467a3f4720ecf803333d989b047d301cacd25f8`
- `mlx_gs/renderer/mlx3d_cpp_compat.py`: `f88398fd3e6f4a869626cb3ec640d9d67c449293056838287362b878ff91b9ba`

Experiment SHA-256: `020febeaeb800d25009bd87f32702cac367c636151ee368107985c26d3f0b416`.
