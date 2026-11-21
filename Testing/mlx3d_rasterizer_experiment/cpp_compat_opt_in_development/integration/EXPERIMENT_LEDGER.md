# CPP-compatible renderer — experiment ledger and current status

**Updated:** 2026-10-26 (Phase 93 Mac validation included; Phase 94 pending).

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
| 89 | **Mac test PASSED:** Instance-local opt-in `tile_sort_impl="selective_final3"` validated without process-global patching; 24 radix4 fallbacks, 8 final3 selections, full RGB+alpha bitwise parity. Moving-pose 700K Full HD medians **105.368 → 102.043 ms** (3.16% faster, startup frames excluded). Initial selective frame **381.848 ms**, compared to baseline **118.732 ms**, is an unresolved first-use outlier; do not promote automatically. The adapter source SHA guard was updated. |

### Current selection semantics

`tile_sort_impl="selective_final3"` is **opt-in**: use indices-only three-category final pass when capacity >= 8M, tiles >= 8000 and final-pass plan eligible; otherwise use radix4. The source still defaults to `tile_sort_impl="mlx"`, while the existing CPP adapter defaults to `"auto"`; neither default was promoted. The renderer setting is per instance, unlike the Phase 86–88 scoped module override. **Do not equate this with a production rollout, integration into site-packages, or a merged PR.**

### Current next validation

From the separate feature worktree with `gs-mps` active:

```sh
git pull https://github.com/KingdomDesignsConsulting/splat-apple.git feature/cpp-compat-renderer
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/bench_phase89_instance_selective.py --rounds 8 --out phase89_instance_selective.json
```

The benchmark refuses to overwrite existing output. Review individual branch selections, correctness and performance before proposing any policy promotion. Also verify threaded/concurrent instantiation behavior separately if that is an operational requirement; removing global monkeypatching does not itself constitute concurrency validation.

## Phase 90 — isolated-process cold-start follow-up

Mac M4, 700K Full HD, four counterbalanced blocks / eight fresh Python processes, five frames each. Median results: radix4 setup **61.250 ms**, first **134.820 ms**, repeat **113.158 ms**, total including setup **647.670 ms**, MLX peak **2,380,810,512 bytes**. Native per-instance selective final3 setup **59.619 ms**, first **129.351 ms**, repeat **107.080 ms**, total **615.296 ms**, peak **2,234,714,280 bytes**. The earlier isolated **381.848 ms** first-frame outlier did not recur. These are observed process-level measures, not proof of a fixed GPU cache mechanism. No automatic prewarming is justified; production default remains unchanged.

## Phase 91 — guarded capacity camera-transition acceptance

User Mac M4 test **PASS**, two rounds of the 15-pose sequence with verified matching workloads and bitwise RGB/alpha equality. Excluding each round's initial two frames: radix4 median **146.006 ms**, p95 **172.278 ms**, max **482.778 ms**, total **3988.614 ms**; native selective final3 median **136.845 ms**, p95 **165.011 ms**, max **396.039 ms**, total **3733.046 ms**. Improvements: median ~6.3% and total ~6.4%, but individual capacity-growth spikes occurred in different implementations at pose 31 (42,477,817 slots) and pose 34 (53,283,577 slots). Single-run maximum comparisons are not sufficient for a reliable worst-case claim. Phase 92 diagnostic was committed to isolate first-use and repeat timing for these capacity shapes in independent processes; **Mac validation pending**.

## Phase 92 — forced-capacity first-use and revisit

User Mac M4 **PASS**, four independent-process blocks per sorter, fixed camera, forced 33.15M → 42.48M → 53.28M capacities with repeated frames and revisits. Median first/repeat ms by size: radix4 33.15M **130.676/113.717**, 42.48M **158.154/134.238**, 53.28M **186.242/156.125**; selective final3 **125.311/109.655**, **145.474/126.927**, **169.109/149.270**. Final3 lower on all nine sequence-step medians. Median MLX peak bytes radix4 **3,718,284,028** vs final3 **3,483,484,820**; allocator cache **8,808,550,724** vs **8,240,468,188** bytes. Block 0 radix4 first-use spikes ~226 ms (42.48M) and ~387 ms (53.28M), not repeated in later blocks. Revisited capacities were generally faster. Cause of first-use cost remains unproven. Phase 93 committed to isolate radix sorting with pre-evaluated input keys in separate processes, Mac run pending. Default still unchanged.

## Phase 93 — sort-only capacity transitions

Mac M4 **PASS**, four independent-process blocks per mode, key inputs explicitly pre-evaluated before timed sorting. Stable-order NumPy oracle passed. Median first/repeated milliseconds: at 33.15M slots radix4 **71.787/63.064** versus final3 **66.096/57.360**; at 42.48M **89.753/80.641** versus **81.237/72.604**; at 53.28M **114.546/100.936** versus **105.590/91.380**. Every one of nine sequence-step medians favored final3. Median peak bytes radix4 **3,606,331,952** versus final3 **3,371,532,744**; cache bytes **6,878,028,360** versus **6,309,927,776**. Radix4 block0 53.28M first exposure **306.27 ms** was an outlier, later blocks ~113–115 ms. This identifies radix sort as a contributor but does not isolate compiler or allocation behavior.

## Phase 94 — forced-synchronization radix stage diagnostic

Mac M4 **PASS**, radix4 and final3 independently measured at 33.15M, 42.48M, 53.28M slots across first-use, repetition and revisit. Repeated full-sort instrumented timings, radix4 → final3: **68.754 → 61.658 ms** (33.15M), **89.049 → 80.586 ms** (42.48M), **109.756 → 106.537 ms** (53.28M). Fourth rank+scatter at 33.15M repeated: radix4 **6.92+7.47 ms** vs final3 **2.95+5.74 ms**. On 53.28M first exposure radix4's first scatter **28.44 ms** vs second frame **17.29 ms**, and final3 **30.25** vs **18.26 ms**; earlier-pass first-use penalty survives specialization. These forced evaluations change scheduling; stage times are **not pure GPU kernel timestamps** and may not sum to normal sort performance. No source-path or default changes.

## Phase 95 — scatter rehearsal rejected

Mac M4 **PASS**, 36 fresh-process trials (3 blocks × 3 capacities × 2 sorters × 2 preparation policies). Median total time including preparation, cold → scatter rehearsal: 33.15M radix4 **202.244→223.882 ms**, final3 **182.539→206.575 ms**; 42.48M radix4 **252.893→282.030 ms**, final3 **232.976→261.167 ms**; 53.28M radix4 **319.075→359.069 ms**, final3 **291.997→328.380 ms**. Rehearsal cost ~22–33 ms and increased peak MLX memory roughly 27%; first timed sort was usually unchanged or slower. **Reject scatter-preparation / automatic prewarming**. Retain the native final3 opt-in and default behavior.

## Phase 96 — isolated rank/prefix/scatter by digit

Mac M4 **PASS** for each of 33,154,187, 42,477,817 and 53,283,577 slots, six software-synchronized samples per radix digit (shifts 0, 4, 8, 12). Shift-0 first scatter vs repeated scatter median: **14.516 vs 9.170 ms**, **19.099 vs 11.775 ms**, **24.068 vs 14.551 ms** (in capacity order). Repeated shift-0 rank medians: **6.962**, **8.712**, **10.883 ms**; prefix **0.801**, **0.881**, **1.087 ms**. Repeated shift-12 scatter: **7.287**, **9.394**, **11.863 ms**. The first scatter at shift 0 shows size-scaled one-time overhead; later digits lack the same consistent first-call penalty. No hardware GPU timestamps or root-cause attribution. Keep prewarm rejected; next isolate write traffic with a paired scatter variant.

## Phase 97 — scatter store-order A/B rejected

Mac M4 **PASS** exact key and permutation equality at all three capacities and four digit shifts. Switching stores to indices-before-keys did not produce a consistent performance gain. Median relative shifts (% candidate vs native): 33.15M **-1.52, -13.91, +18.81, -27.84**; 42.48M **+1.69, -0.17, -6.18, -13.96**; 53.28M **-2.07, +0.72, +8.66, -3.41** for shifts 0, 4, 8, 12. Substantial variance and isolated large candidate outliers. **Reject promotion of reversed store order.** No production changes.

## Phase 98 — last-pass indices-only traffic lower bound

Mac M4 **PASS** full exact permutation equality for native two-output scatter vs experimental indices-only scatter. At 33.15M slots medians **7.445→5.976 ms** (-19.73%); 42.48M **9.799→7.619 ms** (-22.24%); 53.28M **11.713→9.316 ms** (-20.46%). The omitted final key write is 4 bytes per slot. This is a *last-pass-only* control: earlier radix passes need keys for subsequent digits, and native selective final3 already writes indices only on the last pass. Do not promote this as an additional optimization or change production defaults.

## Phase 99 — early-pass index scatter plus key gather rejected

Mac M4 **PASS** correctness for deterministic first-pass identity-index diagnostics at 33.15M/42.48M/53.28M slots, shifts 0, 4, 8; exact intermediate keys and permutations matched native scatter. The alternative was consistently slower (candidate median penalty **+124% to +216%** across nine cases). Breaking the two-output intermediate scatter into index-only output then an MLX gather adds excessive memory traffic and launch overhead. **Reject**; no production modification. Importantly, identity-input-index parity is a *first-pass diagnostic only*, not proof that this formulation works for nonidentity later-pass indices.

## Phase 100 — packed intermediate scatter rejected

Mac M4 **PASS** exact reconstructed keys and permutation for packed 64-bit scatter plus mandatory unpack, at 33.15M/42.48M/53.28M slots and shifts 0,4,8. Packed+unpack median latency was **+18.7% to +69.2%** slower than native fused two-output scatter in all nine cases. By capacity, penalties shift 0/4/8: 33.15M **+50.8/+18.7/+55.4%**; 42.48M **+69.2/+52.5/+68.7%**; 53.28M **+41.2/+41.4/+39.3%**. **Reject** when conversion is necessary for subsequent radix passes. No production change.

## Phase 101 — radix scatter threadgroup launch size

Mac M4 **PASS**, full exact key/index parity for threadgroup sizes 128/256/512 at 33.15M, 42.48M, 53.28M slots for radix shifts 0,4,8. Excluding first sample, **512 threads** improved relative to validated 256 in all nine pairs: 33.15M **8.82/9.14/8.59%**; 42.48M **9.82/10.08/12.54%**; 53.28M **10.07/10.51/5.98%**, by shift 0/4/8. **128 threads** was slower in all nine pairs (approximately 10–15%). This is an isolated scatter microbenchmark, not a claim that complete sorting or full rendering improves by the same percentages. No production change yet.

## Phase 102 — complete final3 radix sorter, scatter launch 512

User Mac M4 **PASS**, randomized paired full-sort exact permutation parity at all three capacity sizes. Excluding initial sample, 256 → 512 thread scatter median latency: 33,154,187 slots **57.751 → 55.307 ms** (-4.23%); 42,477,817 slots **74.020 → 70.722 ms** (-4.46%); 53,283,577 slots **94.204 → 89.867 ms** (-4.60%). Three ordinary LSD passes used the alternative launch size; final specialized indices-only pass unchanged. First 33.15M trial showed outlier 512-thread **209.696 ms** vs 256-thread **71.842 ms**; investigate startup/tail separately. Renderer parity and end-to-end benefit remain **unverified**; do not promote validated source/defaults yet.

## Phase 103 — renderer-level 512-thread scatter validation

User Mac M4 **PASS** 30 frames, exact RGB/alpha and matching workload counts, two rounds of camera transitions including capacity growth 33.15M → 42.48M → 53.28M. Across the benchmark's selected frames (excluding first two per round), frame median **139.581 → 137.492 ms** (-1.50%), P95 **172.612 → 167.078 ms** (-3.21%), sum **4079.379 → 3578.085 ms** (-12.29%). **Caution:** sum and max changes heavily influenced by 256-group **684.026 ms** growth frame and 512-group **212.426 ms** growth frame; these do not establish a repeatable 12% speedup. Renderer implementation was scoped-patched for serial experiments only; no production source/default changes. Next: balanced fresh-process transition qualification.

## Phase 104 — fresh-process renderer scatter512 validation

Mac M4 **PASS**, four independent processes × two rounds × 15 frames (120 rendered paired frames, 104 selected pairs). All bitwise RGB/alpha and workload-count checks passed. Per-process selected-frame median delta (512 vs 256): **-0.987%, -1.218%, -0.399%, +0.651%** (three wins, one regression), median of four deltas **-0.693%**. Pooled median paired difference **-1.685 ms**, 512 faster on **62.5%** of selected pairs. Excluding capacity growth steps 9/10: median difference **-1.563 ms**, same **62.5%** win rate. Growth asymmetry: all four 42.48M first-growth steps slower with 512 (at block 0: 402.557 vs 151.213 ms); at 53.28M second growth three blocks show 256 severe spikes (795.647, 445.228, 439.977 ms). Overall sum advantages **must not** be interpreted as stable renderer speed gains. Phase105 should counterbalance order/seeds to test confounding of first growth and cold compilation. No production/default change.

## Phase 105 — counterbalanced renderer growth execution order

Mac M4 **PASS**, eight fresh processes, four each forced **256-first** and **512-first** at initial capacity growth from 33.15M → 42.48M → 53.28M, with bitwise RGB/alpha parity. At step 9 (42.48M), median paired delta (512 minus 256) **-25.536 ms** with 256 first but **+18.424 ms** with 512 first; at step 10 (53.28M), **-25.872 ms** versus **+22.496 ms**. First-executed configuration incurs extra transition cost; direction reverses with order, and severe outliers occur (256 first 416.883 / 638.991 ms; 512 first 379.911 ms). Do not attribute transition advantage to scatter threads. Non-growth steady pairs still favor 512: **-1.057 ms, 63.64% wins** (256-first processes) and **-2.373 ms, 78.41% wins** (512-first processes). Next evaluate repeat warmed, fixed-capacity frames as a distinct qualification. **No production changes.**

## Phase 106 — warmed fixed-capacity renderer 512-scatter qualification

Mac M4 **PASS**, four fresh processes, fixed 53,283,577-slot workspace, 104 selected frame pairs in rounds 1/2 (growth round 0 excluded), bitwise RGB/alpha and workload parity. The 512-thread intermediate scatter improved medians in **all four process blocks**: **-2.391%, -1.679%, -1.120%, -0.917%** (median-of-block percentage **-1.400%**). Pooled paired difference **-2.206 ms**; 512 won **85/104 = 81.73%** of paired frames. This is the clearest repeatable steady-state renderer evidence; it does not imply a 4.5% renderer improvement from the isolated full sorter. Next qualify a guarded opt-in candidate including fallback and source integrity before any default change. Production renderer unchanged at this checkpoint.

## Phase 107 — guarded 512 scatter gate

Mac M4 **PASS**, exact stable permutation versus native radix4 and validated selective_final3 at four capacities. **1,048,576 slots / 256 tiles** correctly falls back to 256-thread native radix4; **8,388,608**, **33,154,187** and **53,283,577** slots at 8,160 tiles select **512-thread** early scatter, all exact parity PASS. Experimental single-run candidate times 3.082, 18.956, 57.663, 93.237 ms respectively (not controlled A/B benchmark); source unchanged during test.

## Phase 108 — instance-local opt-in candidate (Mac acceptance PENDING)

New instance-local mode **`tile_sort_impl="selective_final3_scatter512"`** changes only intermediate two-output scatter launch size to 512; the three-category indices-only final scatter stays 256. Eligibility stays identical to selective_final3 (capacity >=8M, tiles >=8000, sentinel-safe final pass), otherwise fall back to unchanged radix4; prior `selective_final3`, `auto` and default modes unchanged. Updated source digest guard in `mlx_gs/renderer/mlx3d_cpp_compat.py` after source change. **Mac acceptance is not yet established**: `bench_phase108_integrated_scatter512.py` compares two independent real renderers across growth/warmed frames and exact RGB/alpha without global hooks; inspect its results before declaring integration qualified. PR remains draft.

## Phase 108 — integrated instance-local scatter512 PASS

Mac M4 **PASS**, actual separate renderer instances using `selective_final3` and `selective_final3_scatter512` without process-global patch. Full bitwise RGB+alpha parity and identical workload counts across three rounds of 15 camera poses (including 33.15M → 42.48M → 53.28M growth). Local source SHA guard loaded successfully. For 26 warmed frames (rounds 1/2, steps ≥2), median **141.967 → 139.001 ms** (**-2.089%**), candidate won **23/26 = 88.46%**; peak shared MLX memory 3,511,501,460 bytes. A large 256 growth outlier **551.259 ms** and a 512 growth outlier **155.733 ms** at 53.28M are not grounds for assigning growth benefit to threadgroup size, given Phase105 order confounding. Integrated mode is accepted as **explicit opt-in candidate**, not new default; further fresh-process replication and explicit fallback coverage pending.

## Phase 109 — repeated integrated opt-in acceptance PASS

Mac M4 **PASS** 3 independent Phase108 renderer processes, exact RGB/alpha parity and source integrity in every process, 26 warmed comparisons per block (78 total). Per-block warmed median 256→512 (ms): **139.875→137.392 (-1.775%)**, **138.829→137.002 (-1.316%)**, **138.486→137.192 (-0.934%)**. All three blocks favored 512, frame win rates **92.31%, 84.62%, 88.46%**, totaling **69/78 (88.46%)**. Three direct-sorter fallback permutation checks PASS at 1,048,576 slots/256 tiles, 1,048,576 slots/8160 tiles and 8,388,608 slots/256 tiles. Limitation: these tests did *not* instrument actual renderer-level fallback routing; they alone do not demonstrate fallback dispatch for real scenes. Keep new option opt-in; default unchanged.

## Phase 110 — renderer fallback dispatch acceptance PENDING

Added `bench_phase110_renderer_fallback.py`, testing two real rendering cases (4096 splats / FullHD and 700K splats / 640×360). Serial, scoped instrumentation counts actual radix4/final3 calls and verifies 256-thread intermediate scatter with exact RGB/alpha parity versus existing `selective_final3` mode, restoring all hooks before rendering the reference. Mac results are pending; no production change.

## Phase 110 — live renderer fallback PASS

Mac M4 **PASS**: scoped serial instrumentation verified actual `selective_final3_scatter512` renderer dispatched to `_radix_tile_argsort` exactly once, never called `_selective_final3_argsort`, and used **only 256-thread intermediate scatter** for two ineligible workloads. Small-capacity 1920×1080 (4,096 Gaussian scene) sorted **196,331 slots** in four 256-thread scatter passes; 640×360 scene (700K Gaussians, fewer than 8,000 tiles) sorted **6,298,685 slots** in three 256-thread scatter passes. Both PASS bitwise RGB/alpha parity and identical workloads against existing `selective_final3`. No production changes, defaults remain unchanged. Final consolidated acceptance gate pending.

## Phase 111 — consolidated opt-in acceptance PASS

Mac mini M4 **PASS**. Combined Phase110 live renderer fallback dispatch and Phase109 three fresh-process real-renderer parity/speed tests, all source SHA checks passed. All three warmed process medians improved with `tile_sort_impl="selective_final3_scatter512"`: **-2.2226%, -2.1221%, -1.6017%**; median process gain **2.1221%**. Full RGB/alpha equality and workload parity in each integrated process, fallback 256-thread radix4 in both real-scene ineligible cases. This closes the explicit opt-in qualification gate on the tested M4 workload. `auto`, `selective_final3`, other legacy modes and defaults are **unchanged**. Do **not** represent as cross-hardware qualification or automatic default promotion. Draft PR #1 remains unmerged.

## Phase 112 — default-preservation and source-guard regression PASS (Mac M4)

Mac mini M4 **PASS** running source commit `9d65ba0c` with minimal single-Gaussian constructor fixture. Original uncorrected test failed from missing Gaussian inputs, a test-only fixture defect. Observed report: native Fast default `mlx`, persistent adapter `auto`, one-shot adapter `auto`; accepted modes `mlx`, `radix4`, `auto`, `selective_final3`, `selective_final3_scatter512`; invalid modes rejected; local source SHA `56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710` matched the adapter; simulated source tampering rejected; source-loader restoration PASS. In-process hash mock only; no repository source changes. User-provided console report records `RESULT phase112 PASS` and was produced after pulling `9d65ba0c`. Later script-only commit `34e7b5ae` strengthens the negative test to require a sorting-mode-specific `ValueError`; **that refinement has not separately been Mac-run**, and does not negate the passing Phase112 run. No renderer implementation/default or digest edits. Phase111 continues to qualify 512-thread scatter strictly as explicit opt-in for tested M4 workloads; no cross-device or global-default qualification.

## Phase 113 — cross-resolution workload survey PASS, performance mixed (Mac mini M4)

Mac M4 **PASS exact RGB/alpha parity and equal workload counts** for every compared frame at three resolutions, using 700,000 Gaussians and 8 warmed paired frames each. 1280×720 (3,600 tiles; ineligible for 512 scatter): warmed median 256→512 **69.261→68.511 ms** (-1.084%; 5/8 candidate wins), paired median delta -1.203 ms; threshold eligibility 0/8. This is a **fallback comparison**, not evidence of a 512-thread speedup. Full HD 1920×1080 (8,160 tiles): **139.479→138.035 ms** (-1.035%; 6/8 wins), paired median delta -2.637 ms; eligible 8/8. High-resolution 2560×1440 (14,400 tiles): **236.748→239.183 ms** (+1.029% regression; 3/8 wins), paired median delta +2.436 ms; eligible 8/8. Recorded cumulative process MLX peak-memory highs after cases: 1,616,414,876; 3,514,991,252; 6,190,007,036 bytes respectively; **not independent per-case or per-mode memory use**. Warming rounds grew sort capacity: subthreshold 26,287,149, Full HD 53,340,719, highres 89,374,675 slots. Cold-growth outliers excluded, consistent with Phase105 confounding. **Decision:** retain 512-thread mode as explicit opt-in, no global default or all-resolution performance claim. Probe highres in independent fresh-process/counterbalanced validation before investigating launch geometry; no production changes.

## Phase 114 — fresh-process high-resolution replication (Mac PENDING)

Committed `run_phase114_highres_replication.py` in `8d431d13`. Runs four fresh-process Phase113 highres-only blocks by default, each with four rounds of four fixed poses; cold round excluded (12 warmed comparisons per block, 48 total). Requires exact RGB/alpha and equal workload counts, plus 512 eligibility for every warmed pair. Preserves subprocess logs and JSON evidence and reports per-block relative timing, win rates, and separate-process MLX peak high-water memory. Phase113 now supports `--order-offset`; Phase114 counterbalances first-executed mode across the four subprocess blocks (0/1/0/1) and reports order-group medians. Performance direction is observational, not an automated speed gate. Does not isolate thermal effects, GPU scheduling, or per-kernel memory. **Mac result pending**. No production, default, or source digest change.

Phase114 harness correction: commit `95273f81` repairs a Python `SyntaxError` caused by literal `\
` text in the order-group summary block. Previous invocation did not execute benchmark cases and supplies no Phase114 performance evidence. Mac rerun pending.

Phase114 further harness correction: `b0a34672` fixes a second literal `\
` in the final summary print dictionary (line 104), found by the Mac `py_compile` check. Inspected the full script for remaining `\
` sequences; only legitimate newline string literals for JSON writes remain. No Phase114 benchmark results yet; Mac rerun required.

## Phase 114 — Mac result

PASS correctness for four fresh-process high-resolution blocks, 48 warmed comparisons. Relative 512 changes by block: +1.884%, -0.270%, +0.651%, -0.499%. Median +0.1907% (512 slower). Offset 0 group +1.2677%; offset 1 group -0.3841%. Two processes favored each mode. Keep 512 opt-in; no default promotion. Raw results remain on user's Mac in phase114_highres_replication_4. No production changes.

## Phase 115 — within-pose high-resolution order isolation (Mac PASS; order-sensitive)

Implemented `bench_phase115_order_isolation.py` (`2cb331fc`). At 2560×1440 / 700,000 Gaussians, warm the maximum-capacity pose for both independent renderer instances, then compare identical poses with two opposite execution orders (AB, BA), alternating which order block comes first. Exact RGB/alpha and workload parity mandatory for every pair, all observed capacities must exceed the 8M eligibility threshold, and warmed (round >0) first-mode groups must be balanced. Report conditional median 512-minus-256 differences for both first-execution groups, win counts, and their contrast; do not mistake this exploratory estimate for a causal result. Output file refuses overwrite. Mac M4 **PASS correctness**: 4 rounds, 32 total pose-order comparisons, 24 warmed pairs; exact RGB/alpha and workload parity. With 256 first, median 512−256 = **−1.7153 ms** (512 won 8/12); with 512 first, median **+2.4000 ms** (512 won 4/12); contrast **−4.1154 ms**. Peak MLX high-water 6,190,007,036 bytes. Conditional median contrast indicates execution-order sensitivity, but drift and repeated samples prevent a causal hardware claim. No consistent high-resolution speedup established. Raw JSON remains on user's Mac. No production/default/SHA changes.

## Phase 116 — symmetric highres ABBA/BAAB order control (Mac PENDING)

Implemented `bench_phase116_symmetric_order.py` (`464c1244`, refined `a4807f97`) to compare identical highres poses with four-render symmetric order ABBA or BAAB, alternating the pattern. Both independent engines are warmed at maximum capacity; exact cross-mode RGB/alpha parity, matching counts, and 8M-slot eligibility required. Per-quadruplet means cancel a linear timing drift under equal-duration measurement assumptions; non-linear drift, thermal effects, and competing GPU load remain uncontrolled. Three warmed rounds (default) yield 12 balanced quadruplets and 48 timed warmed renders. Performance is descriptive; no production changes. **Mac M4 PASS correctness.** 4 rounds, 16 quadruplets (12 warmed, 48 warmed render calls). ABBA median 512−256 **+5.1374 ms** (512 won 1/6); BAAB **−1.2402 ms** (512 won 5/6); combined warmed median **+0.5503 ms** (512 slightly slower). MLX peak high-water **6,307,971,900 bytes**. Results are observational and strongly order-pattern dependent; no validated high-resolution speedup or production/default changes. Raw result remains on the Mac.

Existing follow-up `bench_phase116_randomized_order.py` (already present on the branch; do not duplicate) randomizes pose and AB/BA block order by seed, averages both orders within each pose/round, and reports descriptive round-cluster bootstrap intervals. Mac acceptance of that follow-up is **pending**.

### Phase 116 — independent symmetric rerun (Mac M4 PASS)

Second four-round Phase116 symmetric ABBA/BAAB execution passed exact RGB/alpha and workload parity. Twelve warmed quadruplets (48 render calls): overall median 512-minus-256 **+0.7528 ms**, ABBA **+1.2200 ms** (512 won 1/6), BAAB **+0.4105 ms** (512 won 2/6), 512 won only 3/12. ABBA-vs-BAAB median contrast decreased from 6.378 ms in the first run to 0.809 ms; this indicates less order-group separation in this sample, not established stability or causal elimination of order effects. MLX peak high-water **6,307,971,900 bytes**. Both groups now mildly favor 256; retain opt-in/defaults. The existing randomized-order follow-up has not been run. Raw JSON not uploaded.

## Phase 117 — randomized-order follow-up (Mac PASS correctness)

Executed existing `bench_phase116_randomized_order.py` as Phase117 follow-up: 8 rounds, seed 1162026, 700,000 Gaussians, 2560×1440, 56 warmed comparisons (28 matched pose-round pairs). Exact RGB/alpha and workload parity PASS. 512−256 order-averaged mean **−1.3820 ms**, median **−2.0892 ms**; by first-executed mode, 256-first median **+0.2953 ms** (512 wins 14/28), 512-first median **−1.3971 ms** (wins 17/28). Mean first-order contrast **+0.4043 ms**, median **+0.6561 ms**. Descriptive round-cluster bootstrap interval for mean **[−2.8681, +0.0810] ms**, includes zero. MLX peak 6,190,007,048 bytes. **Decision:** correctness accepted; no conclusive highres performance advantage. The script internally reports phase 116 despite this sequence being labeled Phase117; preserve provenance. Retain opt-in/defaults. Raw JSON remains local.

## Phase 118 — independent randomized-order replication (Mac PENDING)

Committed `run_phase118_randomized_replication.py` (`fbfaa179`). Four separate subprocesses with distinct deterministic seeds each execute the existing randomized AB/BA benchmark for eight rounds; outputs include all child JSON/logs and aggregate mean-of-process-means, process median, sign counts, and range. Requires each child correctness PASS and expected matched-pair counts. Results pending; do not claim improved high-resolution performance or change defaults.

## Phase 118 — independent randomized-order replication PASS (Mac M4)

Four independent fresh-process runs passed correctness, preserving child JSON/log evidence locally. Order-averaged mean 512-minus-256 deltas by process: **-1.4578, +1.2801, +0.896, -0.929 ms** (console-rounded last two). Two of four processes favored each mode. Median of process mean deltas **-0.01665 ms**; mean of process means **-0.05273 ms**. These near-zero aggregate effects and sign reversals do not establish a high-resolution 512 advantage. **Decision:** close the current high-resolution speedup claim as unsubstantiated under randomized fresh-process testing; retain explicit opt-in and unchanged defaults. Earlier Phase111 Full HD opt-in qualification remains valid for its tested workloads. No production code modification.

Qualification decision through Phase118: see `PHASE118_QUALIFICATION_DECISION.md`. Full HD opt-in evidence retained; high-resolution benefit unsubstantiated. Next work is API documentation/source-guard regression coverage, not repetitive timing.

## API documentation and integrity automation (Mac validation PENDING)

Added `CPP_COMPAT_API.md` (`cf4d01f7`) documenting one-shot versus persistent calls, input activations, camera conventions, five sorting modes, explicit 512 eligibility, and integrity safeguards. Added `run_cpp_compat_api_integrity_gate.py` (`8d8a04e3`) to run Phase112 invariant/source-tamper checks and public dispatch smoke testing in isolated subprocesses, capture logs, write a refusal-to-overwrite JSON report, and exit nonzero on failure. No production renderer, installed MLX3D, or source SHA changes. **Mac execution pending.**

## CPP-compatible API/source-integrity gate — Mac PASS

Mac M4 executed `run_cpp_compat_api_integrity_gate.py` after pulling branch revision `f37b4718`. Both child checks completed successfully: **PASS phase112** (sorting defaults, mode validation, source hash/tamper rejection and recovery) and **PASS public_dispatch_smoke** (public rendering path matches direct validated renderer, as tested by the smoke script). Final console: `RESULT CPP_COMPAT_API_GATE PASS report cpp_compat_api_integrity_1.json`. This confirms the gate ran at the tested revision; the report is local to the Mac and was not uploaded to GitHub. No renderer code, defaults, installed MLX3D, or source digest changes. PR #1 remains draft.

## PR #1 integration readiness — targeted review (Mac pending)

Reviewed current PR metadata, opt-in public dispatch and adapter, and API documentation. Identified a gap: previously passing gate only exercised one-shot public rendering; `PersistentCppCompatRenderer.update()` parity across unchanged/updated scenes was not included. Added `test_persistent_cpp_compat_api.py` (`0f4cd7fb`) and integrated it into the gate (`4d8af1fa`), requiring three subprocess passes. Corrected stale API documentation about Mac validation. Existing source guard pins both installed MLX3D and checked-in local Fast hashes; this intentionally limits portability and requires a known installed baseline. **Expanded gate untested on Mac**; do not mark PR ready or merge. No renderer/default/digest changes.

## Expanded three-check API integrity gate — Mac M4 PASS

After pulling the integration-readiness changes, Mac execution reported `PASS phase112`, `PASS public_dispatch_smoke`, and `PASS persistent_scene_update`, followed by `RESULT CPP_COMPAT_API_GATE PASS report cpp_compat_api_integrity_2.json`. Thus the newly added persistent scene-update exact-parity regression passed alongside the source/default guard and one-shot dispatch smoke test. This closes the specific untested-persistent-update gate item from the prior integration review. Raw report remains on the Mac; this is a focused smoke/regression gate, not broad device/dependency validation. No renderer code, sorting defaults, or validated SHA modified. PR remains draft pending separate readiness decision.

## Phase 119 — dependency failure isolation (Mac PENDING)

Added `test_phase119_dependency_guard.py` (`3aea6572`), exercising missing `mlx3d.splatting.fast`, mismatched installed Fast digest, and post-mock loader recovery. Expanded API gate to four checks (`a274a8d4`). This is a non-destructive guard regression; actual compatibility with different MLX3D releases remains unqualified. Pending Mac execution. No production/default/digest changes.

## Phase 119 — dependency guard Mac PASS

Expanded four-check gate Mac M4 PASS: phase112, public_dispatch_smoke, persistent_scene_update, dependency_guard. Final output `RESULT CPP_COMPAT_API_GATE PASS report cpp_compat_api_integrity_phase119_1.json`. The tested missing/mismatched MLX3D failure simulation and loader restoration passed. Report remains local. No production/default/source digest changes.

## Phase 120 — camera and scene replacement regression (Mac PENDING)

Implemented `test_phase120_camera_scene_edges.py` (`8cb15cf2`): persistent versus one-shot exact RGB parity for unchanged and translated cameras, image size changes, and scene replacement with different Gaussian count and opacity. Extended combined gate to five checks (`89f052ab`). Test not yet Mac-executed. No production/renderer/hash changes.

## Phase 120 — five-check API integration gate Mac M4 PASS

User executed combined integrity gate after pulling Phase120 changes: `PASS phase112`, `PASS public_dispatch_smoke`, `PASS persistent_scene_update`, `PASS dependency_guard`, `PASS camera_scene_edges`; terminal result `RESULT CPP_COMPAT_API_GATE PASS report cpp_compat_api_integrity_phase120_1.json`. Camera translation, image-size transition and replacement-scene parity checks therefore passed on tested Mac M4 environment. The JSON report remains local. These smoke tests do not cover every malformed input, MLX3D release or GPU. Retain opt-in defaults, source SHA, and draft PR status.

## PR #1 final integration audit (NOT READY for general merge)

See `PR1_FINAL_INTEGRATION_AUDIT.md` (`6c5645ef`): identified reproducible clean-environment MLX3D installation gap (installed Fast source pinned to a SHA but no installation recipe), and clarified that cached `_renderer_class` checks hashes only at first class load, not continuously. M4 five-check gate remains PASS, but portability and real missing-dependency legacy behavior in a clean install are not qualified. Keep draft; preserve current source/defaults/digests.

## Installation reproducibility — provenance probe implemented (Mac PENDING)

Added read-only `probe_mlx3d_provenance.py` (`169a450c`, refined `692d52d3`) and `MLX3D_REPRODUCIBLE_INSTALL.md` (70dd9684). They inspect and document exact installed and checked-in SHA-256 values, available package direct URL/Git origin/commit, and the separate clean-environment qualification sequence. **No upstream immutable MLX3D revision has yet been verified, and no clean environment has been created or tested.** Mac provenance output is needed to identify that revision. Current `gs-mps` remains untouched. Draft PR blocker remains open.

## MLX3D provenance probe — Mac M4 BASELINE_MATCH

The Mac read-only probe reported `MLX3D_PROVENANCE BASELINE_MATCH`: installed Fast at `gs-mps/lib/python3.13/site-packages/mlx3d/splatting/fast.py` SHA-256 `4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`; repository-local Fast SHA-256 `56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710`. Original report `mlx3d_provenance_1.json` remains on Mac. Terminal excerpt does not expose distribution metadata or checkout URL/commit, so reproducible immutable upstream installation remains unverified. Existing environment unchanged; PR remains draft.

## Installed MLX3D distribution identified; PyPI wheel hash pending

Uploaded provenance JSON reports `mlx3d==0.3.0`, `direct_url=null`, installed Fast SHA matching required `4ef36ac2…edfcbc`. `installed_checkout` erroneously reports Homebrew git parent and must not be treated as MLX3D revision. PyPI lists the official 0.3.0 wheel (July 22 2026), but its embedded Fast SHA has **not** been independently verified. Added non-installing exact-version wheel hash check `verify_mlx3d_pypi_wheel.py` (`d809a3a2`) and updated `MLX3D_REPRODUCIBLE_INSTALL.md` (`f112da55`). Pending Mac result; do not mark clean-environment blocker resolved.

## MLX3D 0.3.0 wheel verified; Phase121 isolated installation (Mac PENDING)

Mac mini M4 downloaded PyPI wheel `mlx3d==0.3.0` and confirmed embedded `mlx3d/splatting/fast.py` SHA-256 `4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`: `RESULT MLX3D_WHEEL_PROVENANCE PASS`, local report `mlx3d_wheel_provenance_1.json`. This closes published-wheel Fast identity, not reproducible clean install. Implemented `run_phase121_isolated_install.py` (`32b99061`) using a disposable Python 3.13 venv, pinned MLX/numpy, pre-MLX3D absence/legacy renderer-module import, exact wheel revalidation and installation, five-check regression gate, saved logs and package inventory. Documented in `MLX3D_REPRODUCIBLE_INSTALL.md` (`66adb176`). **Phase121 Mac run pending; gs-mps untouched; PR draft and production hashes/defaults unchanged.**

## Phase121 disposable install — Mac M4 PASS (user terminal evidence)

Run from feature branch commit `10e8163` in `gs-mps` on Mac mini M4. All seven runner stages reported PASS: `create_venv`, `install_base`, `absent_dependency`, `download_exact_wheel`, `install_exact_wheel`, `five_check_api_gate`, `package_inventory`. Final: `RESULT PHASE121 PASS report phase121_isolated_install_1/phase121_summary.json`. This supports reproducible `mlx3d==0.3.0` wheel installation and five-check API integrity under an isolated Python 3.13 venv in the tested environment; raw JSON and logs remain local and should be inspected/archived for audit. **Important qualification:** absent-dependency stage tested legacy renderer *import*, not actual `python`/`cpp` backend render execution. Multi-device portability, installed-package variants, and full general merge readiness are not established. Preserve source hashes/defaults; PR remains draft pending Phase122–126.

## Open issues and documentation hygiene

1. **Native sorter cold-start outlier was not reproduced in Phase 90.** Follow-up across four fresh processes per mode favored final3 on first and repeated frames. Additional camera-transition tail and allocation validation remains appropriate before default promotion.
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

## Phase122 — actual legacy execution without MLX3D: Mac M4 PASS

`run_phase122_legacy_isolation.py` executed three actual rendering checks in a disposable Python 3.13 venv with **no MLX3D installed or discoverable**: deterministic `python` rendering (including unchanged default dispatch), newly built native `cpp` rendering, and predictable opt-in dependency rejection followed by successful legacy rendering. All three PASS in `phase122_legacy_isolation_3.json`; no CPP skip counted as success. The native extension and source copy were temporary and removed after testing. Fresh GPU readings before workers were 22%, 30%, and 18%.

Attempt 1 failed at PyPI transport before testing. Attempt 2 exposed a test-harness namespace leak from the staged `integration/mlx3d` directory and stopped before renders. That worker path leak was corrected; both failed reports/logs remain retained alongside the passing attempt. This is not evidence of a legacy renderer defect. Exact cached wheel archives were validated (CRC, RECORD, versions, compatible tags, recorded SHA) and installed offline into the fresh venv; `gs-mps` package inventory was unchanged. The venv shares its base Python/toolchain, not base site-packages; this is not an OS-hermetic build claim.

Independent inspection of Phase121's saved summary, five-check gate, and inventory confirmed PASS. Its final runtime was **MLX/MLX Metal 0.32.4**, despite starting installation with MLX 0.30.6; the wheel install upgraded MLX. Phase122 explicitly pins the actual resulting 0.32.4 stack. Source digests and sorter defaults remain unchanged. Evidence: `qualification_evidence/phase122/`.

Mac reproduction (new report name required):
```bash
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase122_legacy_isolation.py --out phase122_legacy_isolation_NEW.json
```

PR #1 remains draft and unmerged; this closes actual tested legacy isolation on this M4 stack, not general portability.

## Phase123 — source-integrity lifecycle and recovery: Mac/CPU PASS

`run_phase123_source_lifecycle.py` tests exact adapter code in disposable copies with fixture installed/local sources and fixture digests; it does not overwrite site-packages or the validated Fast source and executes no renderer kernels. The first report found four cleanup failures: missing export, interrupted execution, rejection after cached success with changed files, and missing source after cached success. The adapter now removes its owned stale module before an uncached attempt and cleans up/re-raises on execution, export, or interruption failure. No Fast source, expected digest, or sorter default changed.

All ten lifecycle checks PASS in `phase123_source_lifecycle_2.json`; the failing `_1` is retained. Recovery, uncached rejection, same-class cached reuse, and post-load disk changes were exercised. The cached class deliberately does not continuously rehash disk contents. This is an acceptable **documented immutable-deployment/load-time contract**, not hot reload or live tamper detection; restart after source/package changes. Existing external class references cannot be revoked. `CPP_COMPAT_API.md` now states this contract.

The actual installed-dependency five-check gate also PASSed in `phase123_api_integrity_1.json` at 9% pre-test GPU utilization. This supplements mocked lifecycle coverage with real source/default, dispatch, persistent update, dependency and camera/scene regression checks. Evidence: `qualification_evidence/phase123/`. PR remains draft and unmerged.

Mac reproduction:
```bash
python3 Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase123_source_lifecycle.py --out phase123_source_lifecycle_NEW.json
```
The lifecycle test is CPU-only. A separate real API gate requires the qualified MLX3D runtime and GPU <=30%.

## Phase124 — API contracts, negative boundaries and recovery: isolated M4 PASS

The new `run_phase124_api_contract.py` runs in a disposable pinned Phase121-final runtime, verifies the exact MLX3D wheel/installed Fast SHA and `pip check`, and compares supported inputs against the frozen pre-Phase124 adapter from commit `6487258`. The initial report exposed actual rejected-update scene mutation and missing safe input boundaries. Its first negative harness intercepted class lookup too early; the final harness guards actual native construction instead so ordinary argument-evaluation errors are not mislabeled. Original report/logs are retained as discovery evidence, not a blanket assertion that every initial negative row was a renderer defect.

The adapter-only correction validates scene/camera/background shapes, real numeric types and defined finite-value conditions before native work; preserves original activation dtype/rounding; normalizes supported three-component backgrounds; and constructs scene replacements before an atomic swap, retaining the capacity reserve. Supported source-defined empty/transparent/zero-scale/zero-quaternion behavior is retained. No native Fast source, validated digest, sorting default or legacy renderer was changed.

`phase124_api_contract_2.json` PASS: all **65 contract checks** plus the existing five-check API gate. All 13 supported formats/cases matched the frozen adapter bitwise for one-shot/persistent rendering. Negative cases use a safe guarded boundary where malformed native shapes could otherwise be unsafe; separate real tests establish rejected-update atomicity, recovery and valid replacement. Actual kernels executed for supported and recovery cases. Pre-test GPU readings were 10% and 9%. The caller gs-mps package inventory is unchanged. Evidence: `qualification_evidence/phase124/`.

Mac rerun (fresh filename):
```bash
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase124_api_contract.py --out phase124_api_contract_NEW.json
```

The qualified API contract is documented in `CPP_COMPAT_API.md`. Validation adds real work; Phase125 must measure actual integration performance rather than inherit old latency assumptions. PR #1 remains draft/unmerged.

## Phase125 — capacity/fallback regression: INCOMPLETE (2026-11-16)

Implemented `run_phase125_integration_performance.py`: six actual-dispatch capacity/tile boundary cases, small/Full HD/high-resolution workloads, four fresh processes per workload, counterbalanced paired warm timings, first-use construction/render observations, one-shot comparisons, pre-Phase124 adapter comparison, and explicitly process-wide RSS/MLX memory.

Both `_1` and `_2` reports are retained with raw child logs. Six boundary cases passed in each attempt, as did all four small-workload blocks. Attempt 1 completed three Full HD blocks; attempt 2 completed one. A subsequent fresh GPU gate exceeded 30% in both runs, so no further GPU worker started. High-resolution coverage and the complete Full HD aggregate remain untested. **This is INCOMPLETE, not PASS.** No general speedup or default promotion is justified.

The first attempt's completed small-workload aggregate measured median current-adapter overhead of about 0.248 ms against the frozen pre-Phase124 adapter. This is an observed validation cost, not a sorting improvement; retain it for follow-up. Partial Full HD worker results must not substitute for the planned balanced four-process aggregate. Source digests/defaults and caller environment remained unchanged. See `qualification_evidence/phase125/`.

Mac rerun when GPU is available (fresh report name):
```sh
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase125_integration_performance.py --out phase125_integration_performance_NEW.json
```

## Phase126 — formal release gate: NOT READY (2026-11-16)

Independent Phase121 log/inventory review and retained Phase122–124 evidence
close the tested M4 installation, actual legacy execution, loader cleanup and
API contract gaps. Phase125 required Full HD/high-resolution coverage remains
incomplete following GPU98%/99% gate readings. `run_phase126_release_gate.py`
fails closed on this required evidence; a missing-evidence self-test verifies
that absent reports cannot become PASS. No CI pass or independent approval is
claimed. See `PHASE126_FINAL_RELEASE_DECISION.md` for blockers, limitations,
source identities and exact reproduction commands. Installation/API/audit docs
now describe actual final MLX0.32.4 and five-check gate. PR stays draft/unmerged;
Fast digests and all sorting defaults remain unchanged.

## Phase125 attempt 3 — partial PASS, GPU gate INCOMPLETE (2026-11-16)

Mac mini M4 user terminal report: exact cached wheels verified, pinned disposable venv set up, initial GPU gate at 30% (allowed); six boundaries worker completed without error; all four small-workload blocks PASS, then Full HD block 0 PASS. The next `ioreg` pre-worker utilization check interrupted the run. Final `RESULT PHASE125 INCOMPLETE .../phase125_integration_performance_3.json`. Full HD blocks 1–3 and all high-resolution blocks remain missing. No kernel/correctness failure reported from completed workers. Do **not** blend partial blocks across independent attempts into a claimed complete run.

Implemented Phase125-only bounded GPU cooldown retry in `run_phase125_integration_performance.py` (`d998142e`): up to 12 independent pre-worker `gpu_gate()` checks separated by ten seconds, each still requiring utilization <=30%, with attempt outcomes saved in `gpu_gate_attempts`. If GPU remains busy/unreadable, stop INCOMPLETE, retaining logs and not launching the next worker. This avoids aborting on a transient post-worker GPU spike but does not bypass resource isolation. New harness change is **not yet Mac validated**. Use a fresh report name. PR draft/unmerged; source hashes/defaults unchanged.

## Phase125 attempt 4 — COMPLETE PASS by Mac terminal output (2026-11-16/17 UTC)

Following bounded 30% GPU cooldown retries, the fourth single-process-orchestrated qualification completed: boundary worker; all four small, all four Full HD, and all four high-resolution fresh-worker blocks PASS; package inventory written; final `RESULT PHASE125 PASS .../phase125_integration_performance_4.json`. Gate recorded transient busy utilization 60%, 44%, 37%, 35%, 38%, 34%, 37%, 33% before permitted <=30% readings. This **closes the execution-coverage blocker** from attempts 1–3; no GPU threshold was relaxed. The user-provided terminal trace is evidence of successful workflow completion, **not** independently inspected numerical timings or raw JSON. Phase126 must ingest the *complete* `phase125_integration_performance_4.json`, verify its invariants, 13 accepted GPU gates and 12 blocks, and reevaluate other archived checks. Before that, release decision remains NOT READY / requalification pending. Preserve previous reports and PR draft; do not infer a speedup or change defaults.

## Phase126 rerun report-path collision (2026-11-17)

User attempted CPU-only Phase126 aggregation with completed Phase125 attempt4, targeting `phase126_release_gate_2.json`; the report path already existed, so exclusive-create `open('x')` raised `FileExistsError` after aggregation. **No release decision can be inferred from the traceback**, and existing evidence must remain untouched. Added early output-existence validation to `run_phase126_release_gate.py` (`94f74534`) so future duplicate filenames fail immediately with a clear actionable error. Next run must use a fresh output name, e.g. `phase126_release_gate_3.json`, and preserve absolute `--performance` reference to `phase125_integration_performance_4.json`. Phase126 result remains pending; PR draft/unmerged and source digests/defaults unchanged.

## Phase126 reevaluation — CONDITIONALLY READY (Mac M4 user execution)

User ran CPU-only `run_phase126_release_gate.py --performance "$PWD/phase125_integration_performance_4.json" --out phase126_release_gate_3.json`, resulting in `RESULT PHASE126 CONDITIONALLY READY phase126_release_gate_3.json`. All **six** aggregate checks PASS: Phase121 installation, Phase122 real no-MLX3D legacy isolation, Phase123 loader lifecycle, Phase124 65 API contracts + five-check gate, Phase125 six boundaries + 12 fresh-process benchmark blocks, and source/defaults/review. Reported review snapshot head `91753f3b4dbc5c1c05bc8fc242c0df0dce502549` is historical/ancestor evidence, not a new review of later documentation-only commits; CI pass remains **unestablished**, no independent external approval claimed. This supersedes the historical NOT READY judgment for *tested M4 conditional qualification*, not general multi-device release. The Phase125 raw performance JSON and Phase126 aggregate JSON remain local and have not been independently inspected for numerical metrics. PR remains draft/unmerged. Source digests and sort defaults unchanged; no performance promotion.

## Phase125/126 uploaded JSON independently inspected — 2026-11-18

User supplied complete `phase125_integration_performance_4.json` and `phase126_release_gate_3.json` directly. Validated report-level structure: Phase125 status PASS, six boundary cases, 13 accepted GPU gates (19–30% at launch), four fresh worker blocks each for small/Full HD/highres, stable fast digest `56a2b035…e31710` and adapter digest `e8f90799…16db7192ec`; Phase126 status PASS, decision CONDITIONALLY READY, six checks PASS, blockers empty, source defaults `mlx/auto/auto`. Phase126 review evidence is anchored at earlier head `91753f3b`; this does not substitute for a fresh independent review or CI.

**Measured order-balanced warmed mean 512−control delta across four process blocks** (ms; negative faster): small 640×360/4096 Gaussians = **−0.07380**, only **1/4** blocks favor 512, actual radix4 fallback; Full HD 1920×1080/700K = **−3.27475**, **4/4** favor 512, actual eligible 512 scatter dispatch (final pass remains 256); highres 2560×1440/700K = **+0.93201**, **2/4** favor 512, radix4 fallback (14,400 tiles, final3 ineligible). Full HD observed gain roughly **2.6%** relative to ~125-ms control median; descriptive, not broad statistical confidence or a default-promotion warrant. Per-block Full HD mean differences −3.9130, −2.7324, −3.4389, −3.0146 ms. Small and highres negative/positive numbers must not be called specialized-kernel wins because they dispatched fallback.

Median per-process *current vs pre-Phase124 adapter* overhead deltas: small **+0.1910 ms**, Full HD **+1.1347 ms**, highres **+0.5831 ms**; the measurement includes full rendering and noisy paired samples, not isolated validation CPU cost. Memory is **process-wide** and MLX allocator/cache counters, not per-mode allocation. Reported MLX peak ~**2.54 GiB** and cache ~**4.40 GiB** Full HD, ~**4.34 GiB** peak and ~**7.62 GiB** cache highres, across respective process blocks. These are not additive allocations or reliable GPU memory comparisons by sorting mode. All source/default integrity checks remain unchanged; no renderer/default edit. PR remains draft/unmerged pending explicit human release/merge decision and any broader portability/CI requirements.

## Phase 127 — release candidate audit PASS

The tested M4 source checkout remains CONDITIONALLY READY. See `PHASE127_RELEASE_CANDIDATE_AUDIT.md` and `phase127_release_audit_1.json` for evidence and packaging scope.

## Phases 128–137 — overnight M4 results

Phase 131 standard/extended qualification and checkpoint resume passed. Phase 128 measured a host validation lead and Phase 129 measured startup costs. Both optimization gates remain incomplete. Phase 130 passed the bounded lifecycle profile. Phase 132 measured mask/sort as the largest confirmed large-workload stage. Phases 133–134 retained active sizing as a scoped candidate (~6.7% Full HD), rejecting universal use due to small-scene regression. Phases 135–137 tested an explicit selector (~7.2% Full HD versus auto), with broader integration pending. See `SPLAT_APPLE_OVERNIGHT_ENGINEERING_REPORT.md`, `overnight_results/`, and `qualification_evidence/overnight/` for exact commands, results and logs.

## Phase 128 follow-up — host camera validation

All 65 original contracts and 12 paired benchmark processes passed with exact image parity. Small scenes improved by 0.173 ms eager and 0.050 ms lazy, with four of four process medians favoring the candidate for each matrix kind. Large-workload frame gains were inconclusive. Every worker started at or below 30% GPU utilization. Candidate retained on the dedicated experimental branch. See `PHASE128_HOST_CAMERA_QUALIFICATION.md` and `qualification_evidence/phase128_host_camera/`.

## Phase138 — independent per-mode memory experiment (Mac PENDING, 2026-11-18)

Phase128 host camera candidate on separate experimental branch `b41c45f`
PASSED all 65 existing contracts and exact image parity in four processes
per workload. Small-camera eager/lazy frame process medians favored the
candidate 4/4, by −0.173/−0.050 ms; Full HD/highres results were mixed.
Do not promote it solely from these timings.

Created new branch `experiment/m4-phase138-isolated-memory` from the
qualified Phase128 candidate, without changing protected PR #1. New test
`run_phase138_mode_isolated_memory.py` (commits `1dd9d56d`, `c399d54f`)
launches one mode per fresh Python process (auto, selective_final3,
explicit scatter512, active sizing, attested-M4 adaptive), measures a
memory timeline during render, scene change and release, and validates
complete RGB checksum parity across independent mode workers.
`PHASE138_MODE_ISOLATED_MEMORY_PLAN.md` documents 60-worker full matrix,
<=30% GPU entry, >=20% memory free, cache and allocator caps, source/default
invariants, and the non-overwriting evidence policy. A copy of the exact
script is saved to Google Drive Splat-Apple folder:
https://drive.google.com/file/d/1N5JAvaZqKYehv9CjmTailRIk6y0qrrtz/view .
Script syntax compiled on a non-Mac host; actual Apple/Metal runtime
**NOT YET TESTED**, so no Phase138 PASS/performance claim. M4 qualification
remains CONDITIONALLY READY for prior validated checkout; no merged PR,
default promotion, source digest or package change.

Phase138 harness verification supplement: standalone Python syntax compilation PASS; a CPU-only 30-worker simulated orchestration test PASS for report aggregation, grouping, cross-worker checksum comparison and non-overwriting artifacts. The simulation used stubbed MLX/subprocess calls and does **not** validate actual GPU execution, Metal memory, sorting dispatch, hardware utilization or performance. Mac M4 full matrix remains PENDING.

## Phase138 harness v2 — stable GPU gate and reusable checkpoints (Mac PENDING)

User observed device-wide `ioreg` utilization spikes 92–97% between successfully admitted workers, generally settling to 0–29% after a cooldown. Causation (previous worker vs system/driver) remains undetermined. The **active Phase138 v1 Mac run is not modified or interrupted**. Prepared separate branch `experiment/m4-phase138-resumable-gate`, based on previous Phase138 code, with no changes to Fast source, adapter, modes, defaults, or original draft PR.

New runner uses 4s post-worker settling and **two consecutive <=30% samples**, 2s apart, bounded at 12 readings with 10s cooldown after busy observations; retains >=20% free-memory requirement. It checkpoints **only verified successful workers** (GPU admission pair, free-memory percentage, exact source/inventory SHA manifest, worker JSON/log SHA-256, identity, provenance). Resuming requires a new output directory and an inactive prior run, rejects tampering/mismatched code, and copies validated evidence without running those workers. No uncheckpointed result is silently promoted to PASS.

`test_phase138_harness_cpu.py`: **8/8 CPU-only tests PASS**, including spurious high utilization, threshold fail-closed, tampering, active lock, complete and partial resumed simulated runs. **Mac GPU qualification NOT RUN for this revision**. The older `phase138_mode_isolated_memory_2_artifacts` has no per-worker GPU-admission checkpoints; cannot automatically resume/qualify its partial files. Preserve it, and do not pull into the ongoing source checkout.

See `PHASE138_STABLE_GATE_AND_RESUME.md`. Google Drive scripts:
- https://drive.google.com/file/d/13EfNQqJlPpWamHniDbXO2z-VcMhSPssx/view
- https://drive.google.com/file/d/1Czhdcc2z4JGpZSe8FxzANT5IzhT9mAEF/view
- https://drive.google.com/file/d/1WATDIlHQpG7jsy256slZWzWC1cCrhxwI/view

Original M4 qualification remains CONDITIONALLY READY; the live Phase138 report has not been inspected and PR #1 remains draft/unmerged.

## Phases140–143 completed — scoped M4 policy PASS (2026-11-20)

The full user-provided completion report was fetched from GitHub commit `0e13dbe78d5e94219e5e43bf1c844f0a0165458a` and reviewed. Phase140 PASS: **36** worker slots (31 new, five reused), GPU entry maximum 30%, 18 busy samples. Phase141 PASS screening: **no eligible density bands**, hence do not enable automatic active sizing. Phase142 retained separate opt-in experimental policy. Phase143 PASS **61 workers**: 65 original API contracts, five native dispatch boundaries, all 60 RGB rendering workers bitwise equal, 23 busy samples, GPU entry max 30%, minimum admitted free memory 63%. All **97** checkpoints across Phase140/143 have two <=30% GPU samples separated by >=2 seconds; user reported local verification of GitHub commit/head and key hashes.

Full HD moderate-scene median mode-minus-auto warmed latency: **combined −9.757 ms** (four of four independent blocks, range −11.179 to −6.313 ms), explicit scatter512 **−9.248 ms**, adaptive **−1.872 ms**, active **+4.664 ms**. Combined first-frame delta −4.838 ms, process-wide MLX peak delta **−139.1 MiB**, retained cache delta **−265.3 MiB** vs auto. These results are for one qualified 1920x1080 band and cannot establish broad acceleration or kernel-owned memory savings. Highres combined warmed delta **+0.733 ms** and mixed results, small combined **−0.288 ms**. Native dispatch verified at 32M/34M inclusive with auto fallback outside band. Combined used NO active-sizing bands; *do not add* independently observed active or adaptive speedups.

Phase141 density screening rejected every active-sizing candidate, including Full HD dense −11.833 ms due to **+1393.5 MiB** retained cache; Full HD moderate +2.286 ms and +434.3 MiB retained cache; other density/resolution cases had inconsistent time/cache tradeoffs. This supersedes suggestions of automatic sizing promotion based only on Phase139 MLX peak reductions.

**Decision:** scoped experimental sort-only policy PASS on tested M4; preserve separate opt-in. General first-frame optimization, long-run memory/cache and performance beyond this specific band remain **UNQUALIFIED**. M4 source-checkout release remains CONDITIONALLY READY, original PR #1 draft/unmerged, native/adapter auto defaults untouched. No extra GPU tests are warranted solely to repeat Phase140–143. Evidence: `PHASE140_143_CONTINUATION_RESULT.md`, `PHASE140_143_COMBINED_ENGINEERING_REPORT.md`, and `qualification_evidence/phase140/` at commit `0e13dbe`. Next: Phase144 targeted initial-frame and cache-retention experiments in a separate branch, not editing the qualified candidate.

## Phases 144–145 — transition latency and final source-matched audit

Phase 144 PASS. Raw Phase 143 evidence isolated an oversized inherited sort reserve on the first small replacement frame. Eight fresh locked workers in four balanced blocks measured a median 89.104 ms saving with a harness-only reserve reset and 124 paired bitwise RGB matches. Peak MLX bytes were equal within each pair and live allocations stayed flat over six cycles. The prototype remains isolated. See `PHASE144_FIRST_FRAME_CACHE_FINDINGS.md`.

Phase 145 PASS. Verified the exact release at `0063302`, retained Phase 121–126/131 qualification, the exact Phase 143 experimental source, and ten fresh loader checks. Scope remains CONDITIONAL READY for the tested Mac M4 forward-rendering source checkout. See `PHASE145_FINAL_INTEGRATION_DECISION.md`.

## Phases 144–145 PASS; Phase146 isolated reserve-guard prototype (2026-11-20)

Commit `4735312fdecaef4781e409985b35029ee5eeb857`: Phase144 experiment demonstrated **−89.104 ms median** for first 640×360 frame after large→1024 Gaussian scene replacement by setting inherited reserve to zero before rendering; 4/4 blocks favor reset; 124/124 bitwise paired frames; MLX peak unchanged; six update cycles show flat live allocation and approximately equal 5.73 GiB retained cache. It is a harness-only modification, **not yet a qualified runtime guard**. Initial highres/warmed highres and FullHD restore do not have consistent speedups. Phase145 exact source audit PASS: 65 retained API contracts and ten fresh CPU loader checks; original release head `0063302` remains M4 forward-only CONDITIONALLY READY; no experimental code merged, no sort default promotion; PR #1 draft/unmerged.

Created `experiment/m4-phase146-reserve-transition-guard` to isolate the next directly warranted experiment. `phase146_reserve_guard.py` implements an explicitly M4-attested, opt-in wrapper that resets an inherited >=2^22-slot reserve only for the **first valid render after scene replacement** when scene size <=4096 Gaussians and view area <=640×360; ordinary renderer untouched. `test_phase146_reserve_guard_cpu.py` supplies host-only policy/atomicity tests; `PHASE146_RESERVE_GUARD_PLAN.md` defines mandatory real M4 GPU, dispatch, 65-contract, transition and memory qualification. **These new Phase146 tests have NOT been run on actual Mac/MLX GPU; the wrapper is NOT ready for release.** Do not extrapolate the 89 ms speedup to this guarded implementation until the new experiment passes.

Phase146 host-side CPU confirmation (user Mac): `python -m py_compile phase146_reserve_guard.py test_phase146_reserve_guard_cpu.py` and `python test_phase146_reserve_guard_cpu.py` completed: **six tests, OK**. This validates the pure policy and mock scene-update behavior only, not actual GPU dispatch or 65 API contracts on the new candidate. The user Mac worktree was created at experimental commit `b5930bf`; do not run the old source as if it contains new tests.

Implemented `run_phase146_reserve_guard.py`, using the previously physically validated Phase140/143 machine-wide GPU lease and checkpointed parent. It stages one 65-contract/five-boundary candidate job and four counterbalanced keep/guard lifecycle process pairs (nine workers), six large→small→large transitions per lifecycle worker, size and camera bounds, cross-mode image bitwise checks, and explicit small-path dispatch verification. See `PHASE146_RESERVE_GUARD_PLAN.md` for exact Mac commands and recovery. **The new Phase146 GPU runner is not yet executed and its physical qualification is PENDING.** No production source or default change; original M4 conditional readiness and draft PR #1 unaffected.

## Conda caller rename: `gs-mps` → `splat_apple` (2026-11-20)

User successfully renamed the **caller** Conda environment via `conda rename -n gs-mps splat_apple`. Verified Python 3.13.15, installed MLX/MLX-Metal **0.32.3** and NumPy 2.4.2 / MLX3D 0.3.0; `pip check` found no broken requirements. Updated **active** `conda activate`/`conda run -n` commands in root README, integration install/qualification guides, Phase138 run instructions, and `continue_phase140_143.sh` to `splat_apple`. Retained historical description of prior `gs-mps` and all archived evidence unchanged. This is a CLI invocation rename **only**: isolated qualification workers still pin MLX/MLX-Metal 0.32.4 in disposable venvs; source hashes, production defaults, validated source checkout, PR status, and prior GPU evidence are unchanged. The Phase146 Python runner uses the active caller interpreter and has no hardcoded Conda environment name. No reruns warranted for this documentation/launcher adjustment.

## Phase146 GPU run: workers complete, aggregate bitwise parity FAIL (2026-11-20)

User Mac terminal from `phase146_guard_1.json`: six CPU policy tests PASS; **nine of nine** GPU workers reached CHECKPOINT (one 65-contract/five-dispatch guard worker plus four keep/guard lifecycle pairs). The aggregate `summarize` failed with `AssertionError: Transition bitwise parity failed in process block 0`. The printed GPU gate pairs obeyed <=30% and >=2s; busy readings 39%, 33%, 31% and 35% were correctly excluded. The log has no individual frame checksum table and does **not** determine whether the failure is due to candidate rendering or the comparison fixture.

**Status: Phase146 FAIL, not qualified.** Do not relax the bitwise gate, merge the prototype, or change release defaults. No need to rerun all nine GPU workers: the `phase146_guard_1_artifacts/lifecycle_transition_moderate_{block}_{keep,guard}.json` worker reports are preserved and contain per-frame RGB digests, sort traces, slot counts and reserve state. Committed `analyze_phase146_parity_failure.py`, which performs **CPU-only** per-frame comparisons for all four blocks and emits the first mismatch per block plus a JSON report. Next engineering step: run this against **existing** artifacts, inspect first failing frame/route/reserve, then implement only a root-cause-specific targeted repair and requalify it separately.

## Phase146 false FAIL explained by aggregate worker-kind collision; GPU evidence retained (2026-11-20)

User ran `analyze_phase146_parity_failure.py` over the eight already checkpointed lifecycle worker reports and obtained **PARITY_PASS: 0 differing frames, 31/31 frames matched in every block (124/124 total)**. This contradicts the old parent report's `Transition bitwise parity failed in process block 0`, and the concrete cause is confirmed in the prior `run_phase146_reserve_guard.py::summarize`: it selected `next` on `job.block==0` and `job.mode=="guard"` without filtering `job.kind`, inadvertently comparing the *contracts* guard worker (first in the manifest) to the *lifecycle* keep worker. This is an **aggregate bookkeeping failure**, not a renderer pixel mismatch. Original `phase146_guard_1.json` retains status FAIL for provenance; do NOT mutate/delete it or rerun the nine completed GPU workers.

Fixed the summary to require `job.kind == "lifecycle"`; added `test_phase146_aggregate_cpu.py` to reproduce the collision and ensure true mismatches remain fatal. Added `reaggregate_phase146_verified.py`: reads original `phase146_guard_1_artifacts/manifest.json`, verifies **all nine original checkpoints** against immutable original manifest and report/log SHA256 and GPU admissions (two <=30% samples at least 2s apart, >=20% free memory), checks 65 contracts and five dispatch boundaries, calls the corrected `summarize`, then writes an independent `phase146_guard_reaggregated_1.json` with per-block first-small latency and memory deltas. This script creates no MLX runtime, never executes GPU work, and uses only existing evidence. **The recovery script and regression suite are committed but have NOT yet been executed on the user's Mac**, so Phase146 final qualification still awaits verified CPU reaggregation. If reaggregation fails, diagnose exact evidence; do not skip integrity checks or loosen bitwise criteria.

From the dedicated Phase146 worktree, once clean and updated:
```bash
B=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration
python -m py_compile "$B/run_phase146_reserve_guard.py" "$B/reaggregate_phase146_verified.py" "$B/test_phase146_aggregate_cpu.py" &&
python "$B/test_phase146_aggregate_cpu.py" &&
python "$B/reaggregate_phase146_verified.py" --artifacts phase146_guard_1_artifacts --out phase146_guard_reaggregated_1.json
```

## Phase146 — recovered scoped PASS, not production release (user M4, 2026-11-20)

Mac user executed corrected `test_phase146_aggregate_cpu.py`: **3/3 PASS**. The new `reaggregate_phase146_verified.py` verified all **nine original checkpointed GPU workers** (hashes and original manifest, <=30% admission pair >=2s apart, >=20% free RAM), confirmed **65 API contracts**, **five dispatch boundaries** and **124 bitwise RGB-matched frame pairs**, producing `phase146_guard_reaggregated_1.json`: **PHASE146 VERIFIED REAGGREGATION PASS**. Peak accepted GPU 30%, minimum accepted free RAM 79%. No GPU work was rerun.

One-shot guarded first-small-frame delta (candidate minus keep) across paired blocks: −82.148, −81.745, −82.354 and −83.462 ms; median of these reported block values **−82.251 ms**, all 4/4 favorable. Each block MLX peak Δ=0 bytes; retained cache Δ=+1,644,541, +1,574,868, +1,574,855, +1,574,849 bytes respectively. Qualifies the *explicitly opt-in experimental runtime guard for tested M4 transitions*; does not justify production default/integration, broad hardware or general-first-frame claim. Distinguish Phase144 harness-only reduction (−89.104 ms) and Phase146 actual guarded candidate (−82.251 ms).

The original `phase146_guard_1.json` FAIL was a **false aggregate selector bug** and remains intact; full original checkpoint evidence and user-generated corrected JSON remain local, not yet independently uploaded. `PHASE146_RECOVERED_QUALIFICATION.md` records decision, provenance and scope. Draft PR #1 and qualified release source/defaults untouched. Next justified work: independent scene/camera transition generalization and production integration analysis, **not** repeating original nine GPU workers.
