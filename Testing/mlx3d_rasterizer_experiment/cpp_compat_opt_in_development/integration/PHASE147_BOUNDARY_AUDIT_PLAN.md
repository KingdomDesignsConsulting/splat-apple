# Phase147 — Transition boundary and release-integration risk audit (PENDING CPU execution)

## Inputs and scope
Phase146 original GPU workers: **9/9** checkpoint hash verification, 65 API contracts, five native dispatch boundaries, and 124/124 bitwise RGB frame pairs passed corrected CPU-only reaggregation. The first small frame improved by a median **82.251 ms** across the four process-block medians; process-wide MLX peak delta = zero; retained-cache delta +1.5–1.6 MiB per block. The original aggregate FAIL was caused by a worker-kind selection bug and must be kept for provenance. See `PHASE146_RECOVERED_QUALIFICATION.md`.

Phase147 asks **whether these effects are confined to eligible transitions** or could regress fallback scenes. Avoid any automatic default promotion. The guarded renderer remains a separate opt-in experiment.

## Stage 1 — existing-evidence CPU audit (do this first)
`audit_phase147_transition_boundaries.py` is a read-only, nine-checkpoint verified script, based on the *original Phase146 manifest* and source-bound worker JSON. It checks every transition and frame digest from the four paired processes: scene counts 1024, 4096, 4097, and width 641 fallback; expected guard decisions; actual MLX small sorting route; paired timing by transition cycle. It reports the median and sign count of first-small-frame deltas by cycle and fails on unexpected reset, dispatch or bitwise parity. It does not open MLX or run GPU experiments.

From the already existing **Phase146 worktree**, after fetching the Phase147 script or creating a separate worktree from `experiment/m4-phase147-transition-boundary-audit`, run:

```bash
B=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration
python -m py_compile "$B/audit_phase147_transition_boundaries.py" &&
python "$B/audit_phase147_transition_boundaries.py" \
  --artifacts phase146_guard_1_artifacts \
  --out phase147_boundary_audit_1.json
```

**Important:** Original local Phase146 GPU artifacts are in the Phase146 worktree, not necessarily the Phase147 worktree. When using a separate worktree, pass an **absolute path** to those original artifacts, without copying/moving them. Never overwrite original failed/corrected reports.

## Stage 2 — only if the audit identifies missing qualification
Prioritize realistic boundary cases before broad performance searches: 4095/4096/4097 scene count, 640x360 versus adjacent sizes, gradual population decrease, 2²² reserve threshold, multiple independent seeds/cameras, invalid-update recovery, repeated large→small→large transitions and zero work under unattested devices. CPU unit tests for policy and exception safety must pass before new GPU work. If existing Phase146 frames already establish parity for a case, do not rerun it just for documentation.

If new Mac GPU execution is actually required, use shared OS-wide `~/Library/Caches/SplatApple/gpu-experiment.lock`, legacy process checks, two consecutive <=30% GPU readings >=2 seconds apart, >=20% free RAM, pinned verified MLX/MLX-Metal 0.32.4 disposable venv, sequential workers, integrity-hashed checkpoints and resume. No parallel GPU tasks.

## Decision
A Phase147 PASS is a **bounded experimental portability/transition claim only** unless new tests prove more. Do not change original M4 release source or default, merge PR #1, or infer support for other Apple Silicon hardware, training, gradients or concurrent renderer use. Publish `PHASE147_BOUNDARY_DECISION.md` and source-backed reports once executed. If CPU audit is sufficient, document `NO GPU RETEST REQUIRED`.
