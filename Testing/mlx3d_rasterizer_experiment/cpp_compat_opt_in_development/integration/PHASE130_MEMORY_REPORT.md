# Phase 130 memory lifecycle

**PASS for the bounded lifecycle investigation.** Three workloads completed an initial render and six small-to-original scene replacement cycles with bitwise equality on return. Live MLX allocations at the original scene return were flat in each sampled cycle.

| Workload | Active on original-scene return | Active after release | Cache after release |
|---|---:|---:|---:|
| Small | 5.89 MiB | 0.45 MiB | 11.50 MiB |
| Full HD | 111.83 MiB | 37.64 MiB | 2.32 GiB |
| High resolution | 148.72 MiB | 37.63 MiB | 3.72 GiB |

Inputs remain held after engine release, so active allocation is expected. Cache records reusable buffers, and process RSS/MLX peak are high-water readings. Full HD cache grew about 23 MiB mainly on the second cycle and showed only small later increments. There was no unbounded live growth observed within six cycles. The benchmark used a 6 GiB allocator cache cap and a 13 GiB monitored live-plus-cache ceiling. No forced clearing was used. The data do not quantify a per-mode memory saving or establish results for unbounded lifetimes. For a reduction claim, compare capped and uncapped workers one policy per process. [Timeline](overnight_results/phase130_memory_lifecycle_1.json).
