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

## Phase146 physical harness now staged (M4 execution PENDING)

`run_phase146_reserve_guard.py` builds a **9-worker, 4-block** M4 qualification using the already validated Phase140/143 common runner. One GPU worker exercises the original **65 API contracts** on the opt-in `ExperimentalReserveResetRenderer` and the five native dispatch boundary checks; the other eight workers are four counterbalanced `keep`/guard lifecycle pairs. Each pair repeats highres → small (1024, 4096, 4097 Gaussian thresholds; 640×360 and 641×360 camera area boundary) → Full HD restore → highres in six cycles. The mode-and-block exact frame checksum matrix is mandatory, as are the expected host guard decisions, small MLX dispatch for eligible resets, and memory/first-frame measurements.

The harness inherits the machine-wide `~/Library/Caches/SplatApple/gpu-experiment.lock`, legacy-process check, two independent <=30% GPU admission samples spaced >=2 seconds, >=20% free memory, bounded retries and tamper-resistant individual checkpoint/resume. It uses the pinned disposable Python environment, preserves source/default invariants, and does not modify release files. Its source and documentation are committed; **no physical M4 GPU qualification has been executed for this runner**. Do not report it as PASS until the nine workers and aggregate report complete.

### From the existing Phase146 Mac worktree

The user ran the six CPU-only guard unit tests successfully on the previous experimental head. Fetch only after no experiment is active, and require a clean worktree:

```bash
cd '/Users/Shared/Main Hard Drive/My Projects/Projects In Progress/Applications/Gaussian Splatting/splat-apple-phase146-reserve' || exit 1
[ -z "$(git status --porcelain)" ] || { echo 'Worktree has local changes; preserve them before updating'; exit 1; }
git fetch https://github.com/KingdomDesignsConsulting/splat-apple.git experiment/m4-phase146-reserve-transition-guard || exit 1
git switch --detach FETCH_HEAD || exit 1
B=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration
python -m py_compile "$B/phase146_reserve_guard.py" "$B/test_phase146_reserve_guard_cpu.py" "$B/run_phase146_reserve_guard.py" || exit 1
python "$B/test_phase146_reserve_guard_cpu.py" || exit 1
python "$B/run_phase146_reserve_guard.py" --attest-m4 --out phase146_guard_1.json
```

If environment contention causes an INCOMPLETE report, resume with new evidence filenames (replace number if occupied):

```bash
python "$B/run_phase146_reserve_guard.py" --attest-m4 \\
  --resume-from phase146_guard_1_artifacts --out phase146_guard_2.json
```

The first command runs **the new 9-worker matrix only**, not Phase144's completed 8 GPU workers or Phase143's 61 workers. Report and raw checkpoint artifacts stay local until explicitly uploaded/committed. If the contract worker fails, inspect its log rather than changing the source or bypassing the failure; preserve all evidence. No merge or default changes.
