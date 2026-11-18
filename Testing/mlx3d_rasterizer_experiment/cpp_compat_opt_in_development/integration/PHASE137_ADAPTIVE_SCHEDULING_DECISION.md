# Phases 135–137 adaptive scheduling decision

Retain the separate experimental opt-in. Full HD earned a reproducible gain on this fixture. Broader policy and memory qualification are pending.

Phase 135 chooses at the native composite boundary, where actual sort capacity and tile count already exist as Python integers. It requires explicit supported-device attestation and imposes no scalar readback or online timing. Phase 136 uses a standalone persistent renderer subclass. It reuses the verified class, preserves the existing validation and atomically replaces scenes. The existing adapter and its auto semantics are independent of the experiment.

| Condition | Selected mode | Observed dispatch |
|---|---|---|
| Attested M4, 1920×1080, 8,160 tiles, 32–34M slots inclusive | Explicit scatter512 | 512 intermediate scatter at 33,154,187 slots |
| Small, 920 tiles and 45,422 slots | Existing auto | MLX sort path |
| 2560×1440, 14,400 tiles and 54,526,961 slots | Existing auto | Radix4 |
| Unattested device or all other inputs | Existing auto | Existing capacity-dependent route |

The narrow band is an experimental hypothesis anchored to Phase 125 and the current 33.15M-slot fixture. Its edges were checked by the host selector matrix. The fixture validated actual dispatch in the band. The native eligibility checks still guard every specialized kernel. The worker also checked camera transitions, replacement with smaller scenes and the unattested path. The simulated unsupported condition was run on M4.

| Median of process medians (ms) | Small | Full HD | High resolution |
|---|---:|---:|---:|
| Auto | 1.554 | 116.337 | 184.654 |
| Radix4 | 1.246 | 117.493 | 185.712 |
| Selective final3 | 1.291 | 111.804 | 186.900 |
| Explicit scatter512 | 1.170 | 108.744 | 184.219 |
| Adaptive | 1.232 | 108.638 | 185.466 |

Each workload used four fresh processes, six paired samples per process with rotated and reversed ordering, and bitwise RGB comparisons. Full HD adaptive minus auto was −8.394 ms (~7.2%) using the median of process-level paired medians, with a range of −8.871 to −8.131 ms. All four Full HD blocks used the intended specialized path. That gain comes from selecting the existing kernel. Adaptive and explicit scatter512 were effectively even. Small and high-resolution fallback timing differences are within the limitations of comparing the same dispatch. High-resolution paired median was +0.800 ms, with a range of −1.341 to +1.570 ms. Sample maxima for auto/adaptive were 129.55/113.00 ms at Full HD and 199.18/201.48 ms at high resolution. Raw reports include first-use orders, timing tails and process memory. Cold observations share a per-process driver cache across modes.

**Phase 137 benchmark execution PASS. Broader integration qualification INCOMPLETE.** Dispatch overhead was included in full frame timing but not timed separately. Five policies share each worker's allocator. Long-running policy retention, mode-isolated memory, thermal telemetry and the remainder of the band across different scenes are unqualified. Keep explicitly experimental and defer an integration PR and default promotion. [Benchmark matrix](overnight_results/phase137_adaptive_qualification_1.json).

```sh
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_overnight_renderer.py --phase 137 --out phase137_replication_NEW.json
```
