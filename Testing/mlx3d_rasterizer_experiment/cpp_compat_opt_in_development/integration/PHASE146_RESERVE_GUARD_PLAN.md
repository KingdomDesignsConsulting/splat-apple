# Phase146 — Guarded reserve-reset prototype (PENDING M4 qualification)

## Motivation
Phase144 showed **−89.104 ms** median in the first 640×360 frame after replacing a large scene, across four counterbalanced process blocks, with **124/124 bitwise frame comparisons** and no detectable MLX peak difference. The original experimental harness reset `_capacity=0` immediately before that frame; it did not implement a runtime-managed guard. Phase145 preserved the M4 qualified release, PR #1 draft, and both sorting defaults.

## Candidate
`phase146_reserve_guard.py`: separate **explicit opt-in** `ExperimentalReserveResetRenderer` derived from the already experimental `CombinedRenderer`. Never modifies `mlx_gs/renderer/mlx3d_cpp_compat.py`, validated MLX3D Fast snapshot, package, production defaults or PR #1.
- Only M4-attested experimental use.
- Only the **first validated rendering call after successful scene replacement**.
- Incoming Gaussian count <=4096, view area <=640×360, preexisting inherited reserve >=2²².
- Host-only decisions, no new GPU readback.
- Let native capacity allocation handle the next frame; do not discard cache manually.
- Reset eligibility after the first rendered view, preserving native behavior on later frames.
- Expose `last_reserve_reset` decision provenance for checks.

This conservative **hypothesis** does not imply that the guard is production-ready. Unlike Phase144's external harness reset, it is now triggered in the experimental renderer. It may have untested effects on cameras, update atomicity, reserve transitions, and transient allocations.

## Verification before any GPU experiment
Run `test_phase146_reserve_guard_cpu.py`. It uses a fake base renderer and verifies the pure policy, threshold boundaries, first-replacement one-shot behavior, unattested fallback, large camera fallback and rejection atomicity. **CPU mocks do not validate actual GPU or MLX behavior.** Full source hashes, API and default invariants must be captured before physical tests.

## Required Mac qualification
Adapt the existing Phase144, 140 and 143 runners *on a distinct worktree* to test this actual opt-in wrapper against the reference CombinedRenderer:
1. 65 existing API contracts, lifecycle/atomic update and camera negative cases against unchanged source. Tests must validate the candidate as well, not merely replay historical PASS.
2. Fresh-process 4-block counterbalanced large→small→large sequences, with varying scene counts (1024/4096/4097), resolution boundaries, and camera changes. Include highres→small, FullHD→small and small→small. Compare **bitwise RGB** (and alpha if exposed), exact dispatch path, reserve before/after and allocations.
3. Ensure the first small frame now takes the expected small sorting path and does not fall back to a huge radix sort. Record first-render/warmed and subsequent large-frame performance, allocator cache, peak, RSS, and repeated scenes. Avoid universal memory savings claims from process metrics.
4. Shared machine-wide `~/Library/Caches/SplatApple/gpu-experiment.lock` held across GPU workers; scan for legacy workers. Two independent <=30% GPU utilization samples >=2 seconds apart, >=20% free RAM; bounded retries with checkpoint/resume and no concurrent workers.
5. Fail closed on source digest mismatch, invalid camera/scene, insufficient memory, busy GPU, checkpoint corruption or incomplete matrix.

Report `PHASE146_RESERVE_GUARD_QUALIFICATION.md` and raw hash-stable evidence. Only if it **passes** should it be proposed for a separately reviewable patch; do not merge or promote automatically.

## Reproduce CPU tests
From this experimental branch checkout:
```bash
B=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration
python -m py_compile "$B/phase146_reserve_guard.py" "$B/test_phase146_reserve_guard_cpu.py"
python "$B/test_phase146_reserve_guard_cpu.py"
```
No GPU test command is supplied yet because the physical qualification harness must first be implemented and reviewed. Do not run the Phase144 GPU experiment again merely to recreate the already archived evidence.
