# Renderer qualification

Run from the separate CPP compatibility worktree. The caller Python starts isolated child processes. Fresh GPU phases create disposable pinned runtimes.

- Quick runs the CPU loader lifecycle and verifies source/default invariants. A quick PASS covers only those checks.
- Standard adds fail-closed aggregation of the retained Mac Phase 121–126 evidence.
- Extended adds fresh absent-dependency legacy execution, API contracts and the Phase 125 benchmark. Add repeated `--workload small|fullhd|highres` flags for diagnostics. Only the complete matrix can report full fresh qualification.

```sh
B=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration
conda run --no-capture-output -n splat_apple python "$B/run_renderer_qualification.py" --profile extended --run-dir phase131_m4_extended_1 --out phase131_extended_1.json
```

Resume using the same profile, workload flags and run directory, plus `--resume` and a fresh `--out` name. Manifest matching pins Git HEAD, script hashes, renderer digests, workloads and seeds. A checkpoint references a complete report and its hash. Interrupted experiments keep their attempt directories and restart as complete experiments, preserving the fixed seed and balanced process order. Reports from partial experiments are never pooled. A hard process kill can leave a .lock file. Confirm its PID has exited before moving that lock aside.

Extended checks system memory pressure before each fresh GPU phase, and the Phase 125 runner checks it before every workload worker. Existing GPU runners use fresh ioreg readings with a hard 30% ceiling. Phase 125 uses bounded cooldown, records readings, and exits incomplete if unavailable. Workloads run sequentially. The memory check is a conservative host reading rather than a guarantee against allocation failure.

A Phase 126 missing-evidence fixture exercises fail-closed reporting without GPU work. CHECKPOINT and report commands always use fresh output filenames.
