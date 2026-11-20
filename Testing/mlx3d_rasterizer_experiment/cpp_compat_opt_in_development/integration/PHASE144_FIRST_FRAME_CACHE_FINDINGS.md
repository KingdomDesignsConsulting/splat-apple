# Phase 144 first-frame and cache findings

**PASS.** The investigation confirmed a scene-transition latency problem and measured a narrow reserve-reset candidate. The reset remains a harness-only experiment.

The CPU audit independently verified all 61 Phase 143 checkpoints. The experimental selector beat auto on the process-first Full HD frame by a median 4.838 ms (3/4 blocks) and high resolution by 3.669 ms (3/4). The small result was 0.159 ms faster (2/4). Those observations do not isolate compilation. Phase 129 measured source load around 31–35 ms, separate from construction and the first synchronized frame. Phase 130 observed flat live allocation across six same-scene return points. Phase 139 showed that active sizing can reduce peak MLX allocation while increasing retained cache sharply. No thermal or system-cold cache variation was measured.

The Phase 143 small replacement first frame inherited 33.0–33.2M Full HD or 54.4–54.5M high-resolution sort slots for about 8.7K active interactions. It dispatched radix4 at 640×360 before the existing reserve-shrink path took effect. Replacement medians were 55.7 and 86.4 ms. Later frames already benefitted from the native shrink.

## Isolated comparison

Harness commit [cd0fd1a](https://github.com/KingdomDesignsConsulting/splat-apple/commit/cd0fd1a8277711d1c8654eb9b4fa6391279735f7). Eight fresh processes in four counterbalanced blocks used identical inputs within each block. Each rendered one process-first high-resolution frame followed by six high-resolution → 1,024-Gaussian small replacement → 700,000-Gaussian Full HD restore → high-resolution cycles. The trial resets the inherited reserve immediately before the first small replacement frame when it is at least 2²² slots. Native initialization determines the new size. The trial does not touch the initial large frame.

| Measurement | Keep median (ms) | Reset median (ms) | Median paired reset − keep (ms) |
|---|---:|---:|---:|
| First high-resolution frame | 224.647 | 222.476 | −0.872 |
| First small replacement | 90.608 | 0.937 | **−89.104** |
| Warm small | 1.725 | 1.207 | −0.531 |
| First Full HD restore | 115.308 | 115.827 | +0.444 |
| First high-resolution return | 294.808 | 293.559 | −1.117 |
| Warm high resolution | 187.746 | 187.858 | +0.103 |

Wall-time medians are taken across processes. Paired medians use within-cycle deltas before aggregation. Separate median library imports were about 134 ms, verified Fast load 32 ms, and construction 45 ms per process in both modes. Fixture construction happens between library imports and verification and is outside those recorded intervals. Startup frame wall time includes geometry, sizing synchronization, allocations and GPU work. Existing Metal driver caches were shared across processes, so startup timings cannot isolate shader compilation. Process-first paired differences were mixed.

The reset sent the first small frame to MLX sorting at about 12.1–12.3K slots instead of radix4 at about 54.5M. First-small-frame paired medians improved in all four blocks (88.1–90.4 ms saved). All 124 matched frame checksums passed, including sort routing changes and scene restores. There was no consistent first-render, Full HD restore, or warmed high-resolution benefit. A small first frame still pays the native sizing readback, which costs far less than the oversized sort in this fixture.

Each pair had identical MLX peak bytes. Live MLX allocation at the matched warmed high-resolution points was flat at about 106.4 MiB. The cache grew as new buffer shapes were encountered, reached about 5.67 GiB at the first subsequent warmed high-resolution frame, then increased by only about 43 KiB across five further cycles. Active memory dropped to about 37.5 MiB after renderer release with scene fixtures still live. Retained cache was about 5.73 GiB in both modes, and RSS high-water observations are in the raw report. This is evidence of bounded allocator reuse in the tested six-cycle sequence. Longer lifetimes and other workloads are outside this measurement.

Keep the reserve reset as a narrow transition prototype. Production integration needs independent qualification of a real guard, the complete API and dispatch matrix against that code, and more transition sequences. Continue to reject automatic active sizing promotion and untargeted prewarming. All eight workers observed two consecutive readings at or below 30% under the machine-wide lock, with minimum admitted free memory 68%. The pinned disposable environment used Mac arm64 Python 3.13, MLX/MLX-Metal 0.32.4 and verified MLX3D 0.3.0.

[Raw startup audit and checkpointed comparison](qualification_evidence/phase144_145/).

```sh
B=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration
conda run --no-capture-output -n gs-mps python "$B/run_phase144_lifecycle.py" --attest-m4 --out phase144_lifecycle_NEW.json
```

If interrupted, append `--resume-from <previous_artifacts_directory>` and use a fresh output path.
