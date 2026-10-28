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
