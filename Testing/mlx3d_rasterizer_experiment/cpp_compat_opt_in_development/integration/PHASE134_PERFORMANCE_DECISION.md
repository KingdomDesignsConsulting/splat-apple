# Phases 132–134 renderer performance decision

Phase 132 profiled four synchronized instrumented/uninstrumented pairs per workload. RGB and alpha remained bitwise equal. Host stage timing includes explicit evaluation and any unevaluated dependencies. Stage fractions are diagnostic estimates. The small workload is particularly sensitive to the added synchronization.

| Stage median (ms) | Small | Full HD | High resolution |
|---|---:|---:|---:|
| Uninstrumented native frame | 2.39 | 120.60 | 189.09 |
| Tail mask and tile sort | 0.74 | 72.63 | 116.48 |
| Raster compositing | 0.92 | 36.59 | 49.10 |
| Depth ordering, prefix, duplication | 0.60 | 10.14 | 15.02 |
| Fused projection and geometry | 0.82 | 1.86 | 1.81 |
| Tile ranges | 0.30 | 0.78 | 1.10 |

At Full HD and high resolution, a hypothetical 20% reduction in the mask/sort stage corresponds to roughly 12% of total frame time before overhead. Raster is the second large-workload target. Covariance is precomputed before warmed frames. The earlier Phase 128 and 129 reports cover activation and first-use overhead separately.

Phase 133 tested exactly one hypothesis using the existing opt-in active duplicate sizing. It reads total interactions earlier, allocates exact sort shapes, and avoids masking an unused tail on these nonempty workloads. No protected kernel edits. Phase 134 replicated it in four fresh processes per workload, fixed scene seed 12345, paired order randomized with seeds 13300–13303 and several camera positions. All timed RGB pairs matched bitwise and the traced sort used the expected capacity.

| Workload | Median paired delta, active − auto | Range of process medians | Processes favoring active | Verdict |
|---|---:|---:|---:|---|
| Small | +0.326 ms | +0.226 to +0.515 ms | 0/4 | Reject universal use |
| Full HD | −8.092 ms (~6.7%) | −9.114 to −3.562 ms | 4/4 | Retain as a scoped candidate |
| High resolution | −3.332 ms (~1.8%) | −5.802 to −1.261 ms | 4/4 | Secondary candidate |

Full HD actual radix length fell from 33,154,187 to 31,574,441. High-resolution length fell from 54,526,961 to 51,929,464. First-render order was counterbalanced and both large workloads had lower process-first observations in all four blocks. Each process measured both modes, so the recorded MLX and RSS peaks cannot establish a memory reduction. Per-worker memory pressure was checked and GPU entry was gated. Thermal telemetry was not isolated. The four-block range is descriptive, not a confidence interval.

Retain the existing production defaults. Use isolated per-mode memory workers and distinct scene seeds/densities before proposing integration. [Comparative matrix](overnight_results/phase134_comparative_benchmark_matrix_1.json) and [stage results](overnight_results/phase132_renderer_stages_1.json).

```sh
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_overnight_renderer.py --phase 132 --phase 133 --out phase132_134_replication_NEW.json
```
