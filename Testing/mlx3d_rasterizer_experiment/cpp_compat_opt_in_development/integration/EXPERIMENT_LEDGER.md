# CPP-compatible renderer — experiment ledger and current status

**Updated:** 2026-10-26. **Branch:** `feature/cpp-compat-renderer`. **Review:** [draft PR #1](https://github.com/KingdomDesignsConsulting/splat-apple/pull/1). **Status:** experimental feature branch, unmerged. The default `tile_sort_impl="auto"` is unchanged.

This is the current navigation and decision ledger. The original [integration review](INTEGRATION_REVIEW.md), [Phases 56–60 report](OPTIMIZATION_PHASE59.md), and [Phase 85 report](PHASE85_FINAL3_REPORT.md) remain historical records; do **not** interpret their older “current source” statements as describing the newest branch. Raw Mac results from Phases 61–89 are principally in the draft PR discussion and user-local JSON outputs; not all raw files have been committed.

## Validated foundations (Phases 41–61)

- 41–43: staged CPP-compatibility geometry/exponential-alpha tests; see integration review. Subsequent source modifications require their own validation.
- 46–58: resolution, capacity, cache, camera and stable-sort/range diagnostics. See the scripts `bench_*phase46*` through `bench_phase58_*` and the Phase 59 report for interpreted findings. Not every phase has a consolidated standalone result file.
- 59–60: automatic adaptive duplicate sizing, binary tile-range lookup, stable 4-bit radix tile sorting. 700K Full HD median reference **188.739 ms** → optimized **115.320 ms** (38.6% lower); 100K Full HD **27.427 → 21.254 ms**; 4096 scene not established faster. Forty exact stable-order tests, renderer parity and varied-camera tests passed. See `OPTIMIZATION_PHASE59.md`.
- 61: follow-up end-to-end 700K Full HD reference **185.81 ms**, optimized **113.61 ms**; 100K **27.02 → 20.33 ms**; image parity passed.

## Capacity growth, policy, and prewarm investigation (Phases 62–80)

| Phase(s) | Finding / disposition |
|---|---|
| 62–66 | Zoom/camera transitions can overflow adaptive capacity and trigger an expensive second composite. Phase 66 found 8/22 retry frames for auto. Hysteresis and naive preflight policies were not accepted. |
| 67–68 | Introduced opt-in `duplicate_sort_mode="guarded"`: check exact duplicate count before expansion, preventing full-work retries. Phase 67 jump sequence 8 → 0 retries; Phase 68 matched-capacity steady auto **123.111 ms**, guarded **122.894 ms**. Default sizing unchanged. |
| 69 | Reject 20% extra reserve vs 5%; persistent radix sorting cost and larger transitions outweighed fewer growths. |
| 70–71 | Larger sort capacities raise steady sort time; first exposure to large shape sometimes incurs very large one-time latency. Small/large return in same process usually avoids repeating the spike. |
| 72–74 | Explicit large-shape prewarm can shift visible first-use latency off the critical path but costs memory and work; Phase 74 total incl prewarm **2457.56 ms** vs cold **2378.89 ms** (+3.3%), peak +~751 MB. Do not enable automatically. |
| 75–77 | Radix-only isolated-process experiments initially favored exact-shape preparation; Phase 77 exact prewarm total **1079.45 ms** vs cold **1773.14 ms** in one sequence. These are sensitive to prior GPU/cache conditions. |
| 78–79 | Controlled generic-GPU, near-size and exact prewarm replication overturned single-run interpretation. Phase 79 four-block medians: cold **546.72 ms**, generic **554.80 ms**, near **661.14 ms**, exact **644.08 ms**, including prep. Cold first-run outliers suggest machine history effects, not proven cache mechanism. |
| 80 | Immediate cold-sort first median **125.98 ms**; 30-second idle first median **142.71 ms** across four trials each. No reliable cache expiration pattern. **Reject automatic prewarm/size-class policy** pending new evidence. |

## Sort bottleneck and sentinel-safe last pass (Phases 81–85)

- **81:** matched-capacity 700K Full HD steady auto **115.124 ms**, guarded **115.486 ms**; synchronized tile sorting ~**70.8 ms**, compositing ~**34–36 ms**, expansion ~**8 ms**. Stage measurements include synchronization, not pure GPU kernel times.
- **82:** 33.15M-slot radix median **63.935 ms**; per-digit rank/hist ~6.9 ms, scatter ~7.6–11.3 ms. Total scatter contribution exceeds rank.
- **83:** 4095-tile three-pass sort ~**49 ms** vs 4096-tile four-pass ~**64 ms**; 8160-tile four-pass ~**63–64 ms**. The extra fourth pass is necessary for key correctness.
- **84:** design-only exercise, **not a measured faster kernel**. A binary high-digit partition collides with UINT_MAX sentinels; requires categories 0, 1, sentinel.
- **85:** stable specialized last radix pass (still four passes total), with 3 categories and indices-only final scatter. Exact NumPy/production permutation checks: **348 cases × two variants**. RGB and alpha were bitwise equal over tested renderer matrix including camera/scene updates. The historical detailed report is `PHASE85_FINAL3_REPORT.md`; original implementation remains available in `radix_final3_experiment.py`.
- Phase 85 **user Mac revalidation**, 33.15M masked slots: radix4 **63.324 ms** vs indices-only final3 **58.653 ms** (~7.4% faster). Full renderer 700K Full HD exact **110.732 → 106.096 ms** (~4.19%), masked **113.302 → 107.872 ms** (~4.79%); 700K 720p near-neutral. The full Mac pipeline initially stopped at a GPU-busy gate after successful sort; render-only resume subsequently passed. User-local output directory: `phase85_validation_20261026_124242/` (not asserted to be uploaded).

## Selective dispatch work (Phases 86–89)

| Phase | Observed status |
|---|---|
| 86 | Conservative serial *experimental hook*: select final3 only for sort capacity >= **8,000,000 slots**, tile count >= **8,000**, and sentinel-safe final-pass eligibility. Six renderer workloads passed Mac RGB/alpha parity; 700K Full HD exact **111.060 → 104.561 ms** (~5.85%), masked **113.061 → 107.945 ms** (~4.53%). Fallback display labels do not imply a different sorter actually ran. |
| 87 | Per-pose branch audit (after fixing the parity helper's dictionary-key mismatch), 15/15 full-HD camera transitions selected final3, bitwise output parity; medians **112.128 → 106.826 ms**; capacities ~33.15M → 53.28M. Did not exercise radix4 fallback. |
| 88 | Mixed-scene validation explicitly exercised **24 radix4 fallback**, **8 final3** calls, with bitwise parity. Moving-pose 700K Full HD medians **105.402 → 101.157 ms** (~4.03%); fallback workloads show measurement variation. |
| 89 | **Implementation committed, user Mac test pending as of this update.** The validated Fast renderer snapshot now accepts per-instance opt-in `tile_sort_impl="selective_final3"`, eliminating the process-global experimental hook. The adapter source SHA check was updated. `bench_phase89_instance_selective.py` compares instance-local policy against radix4 using randomized pairs and bitwise image/alpha checks. Do not claim Phase 89 passed until Mac output is provided. |

### Current selection semantics

`tile_sort_impl="selective_final3"` is **opt-in**: use indices-only three-category final pass when capacity >= 8M, tiles >= 8000 and final-pass plan eligible; otherwise use radix4. The source still defaults to `tile_sort_impl="mlx"`, while the existing CPP adapter defaults to `"auto"`; neither default was promoted. The renderer setting is per instance, unlike the Phase 86–88 scoped module override. **Do not equate this with a production rollout, integration into site-packages, or a merged PR.**

### Current next validation

From the separate feature worktree with `gs-mps` active:

```sh
git pull https://github.com/KingdomDesignsConsulting/splat-apple.git feature/cpp-compat-renderer
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/bench_phase89_instance_selective.py --rounds 8 --out phase89_instance_selective.json
```

The benchmark refuses to overwrite existing output. Review individual branch selections, correctness and performance before proposing any policy promotion. Also verify threaded/concurrent instantiation behavior separately if that is an operational requirement; removing global monkeypatching does not itself constitute concurrency validation.

## Open issues and documentation hygiene

1. **Phase 89 on-device result pending.** Neither renderer parity nor performance for the newly integrated per-instance path is yet established by user Mac execution.
2. **The PR #1 description predates later work.** The PR discussion comments and this document carry the newer evidence; a description rewrite should preserve its original integration provenance.
3. **Source documentation is versioned by revision.** `INTEGRATION_REVIEW.md` describes an old digest; `PHASE85_FINAL3_REPORT.md` describes the pre-integration experimental implementation. Use the present commit and adapter digest for current source integrity, not historical digests.
4. **Not every raw experiment result is in GitHub.** For audit-quality reproducibility, upload original user-local phase JSON, script versions, Mac hardware/software info, and command lines; do not manufacture unavailable evidence.
5. **Avoid performance overclaim:** medians from different experiments are not directly controlled A/B comparisons; small fallback timings are noisy and stage-level forced-sync measurements are not GPU timestamps.

## Source index

- Historical foundation: `INTEGRATION_REVIEW.md`, `OPTIMIZATION_PHASE59.md`.
- Phase 85 full design, correctness and first measured results: `PHASE85_FINAL3_REPORT.md`.
- Individual phase programs: `bench_phase*.py`, `run_phase*.sh`, `test_phase*.py`.
- Experimental Phase 86–88 hook: `selective_final3_experiment.py`.
- Per-instance path: `mlx3d/splatting/fast.py`, source guard and public adapter in `mlx_gs/renderer/mlx3d_cpp_compat.py`.
- Evidence narrative: [draft PR #1](https://github.com/KingdomDesignsConsulting/splat-apple/pull/1) discussion.
