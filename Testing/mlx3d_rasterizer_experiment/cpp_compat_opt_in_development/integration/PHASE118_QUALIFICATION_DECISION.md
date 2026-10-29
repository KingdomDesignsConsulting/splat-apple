# CPP-compatible renderer: Phase 101–118 qualification decision

## Decision

**Retain `selective_final3_scatter512` as explicit opt-in only.**
No global default promotion, no changes to the validated renderer source
snapshot, no adapter digest changes, and no merge of draft PR #1 on these
measurements alone.

## Functional acceptance

- Phase 111 consolidated gate: PASS on Mac mini M4. Exact RGB/alpha
  parity, real-renderer fallback routing, source guard, and three warmed
  process improvements at Full HD.
- Phase 112 defaults/integrity check: PASS at tested commit `9d65ba0c`.
- Phases 113–118: correctness PASS for the tested higher-resolution and
  execution-order scenarios. This does not establish coverage of other GPUs.

## Measured performance: distinguish workloads

| Evidence | Workload | Observed result |
| --- | --- | --- |
| Phase 111 | Full HD, warmed | All three runs favored 512; median process improvement 2.122% |
| Phase 113 | 1280 × 720 | 512 eligibility false; fallback measurements are not a 512 kernel gain |
| Phase 113 | 1920 × 1080 | 512 median 1.035% faster (8 warmed pairs) |
| Phase 113 | 2560 × 1440 | 512 median 1.029% slower (8 warmed pairs) |
| Phase 114 | 2560 × 1440, 4 processes | Median process change +0.191%; 2/4 favored 512 |
| Phase 115 | Highres, paired order | Conditional 512-minus-256 medians -1.715 / +2.400 ms |
| Phase 116 | Highres, symmetric order | Two independent runs: combined +0.550 / +0.753 ms |
| Phase 117 | Highres, randomized AB/BA | Mean -1.382 ms, round bootstrap interval [-2.868,+0.081] ms |
| Phase 118 | Highres, 4 randomized processes | Mean of process means -0.05273 ms; median -0.01665 ms; 2/4 favored 512 |

Positive changes mean slower 512. Wall-clock render plus `mx.eval` measurements
are not GPU kernel timestamps. First-use capacity growth and compilation
were previously shown to create order-confounded outliers. Peak MLX memory
figures are process high-water values and not per-mode memory costs.

## Practical selection guidance

Use the 512-thread mode only when explicitly selected and only with the
existing eligibility/fallback guards. Do not infer a reliable high-resolution
speedup; at 2560 × 1440 the present evidence is statistically inconclusive
and practically neutral. Avoid an auto-selection threshold based on these
small differences. Keep legacy defaults intact.

## Reopening criteria

Revisit renderer selection only with a meaningful measured benefit on a
new production-relevant bottleneck or target device, supported by exact
parity, actual dispatch verification, multiple independent processes,
execution-order control, and a predeclared practical performance threshold.
Keep original JSON/logs with hardware, revision, and environment details;
not all raw runs are checked into GitHub.

Current next work: documentation/API coverage review and source-guard
regression automation. No additional near-identical highres timing rerun
is justified by Phase 118.
