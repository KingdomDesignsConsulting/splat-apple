# Phase 128 host camera validation qualification

The candidate replaces the MLX finite reduction for the validated 4×4 world-to-camera matrix with a host finite check after MLX provides its values. Shape, dtype, camera intrinsics, source guards and atomic scene replacement use their existing paths.

The original 65 contracts passed in a disposable Python 3.13 / MLX-Metal 0.32.4 / verified MLX3D 0.3.0 runtime before benchmarking. These covered 13 supported bitwise reference cases, 44 negative input cases, six negative mode cases and two recovery cases. Each comparison worker also ran 24 dtype/finite-value cases against both adapters. The float64 check used the CPU stream. Both one-shot and persistent images matched bitwise across all three workloads with eager and independently constructed lazy matrices.

The reference used for the contracts is the original Phase 124 reference. For timing, the control is the then-current adapter frozen directly from `c32ce63` and checked against its known digest. Each workload used four fresh processes. At each of four camera positions, both A/B orders ran for each matrix kind. Timings are warmed and include synchronization as performed by the public persistent renderer. Camera-prep timing stops before evaluation of the resulting camera slices. Every GPU worker had an independent pre-launch utilization and memory-pressure check.

Negative deltas favor the host candidate. Each number is the median of four within-process paired medians. The render columns include the entire public persistent frame.

| Workload | Eager camera Δ ms | Lazy camera Δ ms | Eager frame Δ ms | Lazy frame Δ ms |
|---|---:|---:|---:|---:|
| small | -0.204 | -0.025 | -0.173 | -0.050 |
| fullhd | -0.417 | -0.046 | -0.776 | -0.111 |
| highres | -0.507 | -0.115 | +0.031 | +0.001 |

| Workload and matrix | Frame process medians favoring host | Frame median range (ms) | Control frame median (ms) |
|---|---:|---:|---:|
| small eager | 4/4 | -0.219 to -0.117 | 1.026 |
| small lazy | 4/4 | -0.125 to -0.008 | 1.062 |
| fullhd eager | 3/4 | -0.996 to +0.211 | 121.445 |
| fullhd lazy | 2/4 | -3.561 to +1.140 | 121.736 |
| highres eager | 2/4 | -0.738 to +1.269 | 185.886 |
| highres lazy | 2/4 | -2.455 to +2.239 | 185.489 |

The highest admitted utilization was 30%. The raw reports contain exact commands, matrix checks, process memory, paired timings and the accepted and blocked GPU checks. Memory counters describe both renderers within each process.

Candidate adapter SHA-256: `2554ff81b3ae12a40a1b7cc07b6cb79769dc0c4036216cf3bf6e8e91ada6425e`. This is an experimental candidate benchmark. The release branch still carries its original adapter.

From this experimental worktree, run:

```sh
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase128_host_camera.py --out phase128_host_camera_NEW.json
```
