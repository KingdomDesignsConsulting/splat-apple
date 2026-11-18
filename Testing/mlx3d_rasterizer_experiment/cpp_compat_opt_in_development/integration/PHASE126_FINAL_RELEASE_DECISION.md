# Phase126 final release decision — NOT READY

Date: 2026-11-16. Repository: KingdomDesignsConsulting/splat-apple,
feature/cpp-compat-renderer, draft PR #1. **Keep draft and unmerged.**

The CPP-compatible renderer has passed the tested M4 installation, legacy
isolation, source lifecycle and API contract requirements. Required Phase125
performance/integration coverage is incomplete, so release readiness fails
closed. Committed test code and partial timings are not completed evidence.

| Phase | Executed result | Evidence under qualification_evidence/ |
|---|---|---|
| 121 | PASS: seven installation stages, five real API checks; independently inspected | phase121/ |
| 122 | PASS: actual Python and newly built CPP renders without MLX3D; isolated opt-in failure | phase122/*_3.json |
| 123 | PASS: ten load/reject/cache/recover cases; cleanup fixed in adapter | phase123/*_2.json; phase124/phase124_loader_regression_1.json |
| 124 | PASS: 65 contracts and five real API checks; 13 supported cases bitwise against frozen adapter | phase124/phase124_api_contract_2.json |
| 125 | INCOMPLETE: six boundaries and all small blocks pass, Full HD partial, high-resolution missing | phase125/*_1.json and *_2.json, raw child logs |
| 126 | NOT READY: required Phase125 completeness gate fails | phase126_release_gate_1.json |

## Findings and remaining blockers

Both Phase125 attempts stopped before a new worker when Device Utilization
rose to 98% and 99%, respectively (limit30%). Attempt1 completed three Full HD
blocks, attempt2 one. Neither has the planned four Full HD blocks or any
high-resolution blocks. Re-run to completion when GPU availability permits;
do not combine selectively chosen blocks or inherit historical timings.
No completed child reported a correctness or actual-dispatch failure.

The six boundary cases test capacities7,999,999/8,000,000/8,000,001 at8160tiles
and tile counts7999/8000/8001 with capacity8,000,001. RGB and alpha matched
bitwise and actual kernel dispatch matched the eligibility/fallback rule.
At2560x1440 (14,400tiles), final3 is ineligible: both selective modes use
radix4. High-resolution timings must be described as fallback integration
measurements, not direct scatter512 kernel measurements.

Phase125 attempt1 small-workload median adapter overhead was about0.248ms
against the pre-Phase124 adapter. This is real measured validation cost;
evaluate the full completed report before optimization or release claims.
First-use timings include constructor/first render after module loading,
not whole-process startup or system-cold compilation. One-shot measurements
include scene reconstruction/capacity costs; memory is process-wide high-water
RSS and MLX counters, never a mode-specific allocation comparison.

Earlier Phase111 (~2.122% warmed Full HD improvement) and Phase118 (neutral
high-resolution result) remain historical, narrowly scoped findings. They
do not qualify the current adapter's incomplete Phase125 or justify promotion.

## Integrity, contracts, and supported scope

Native Fast remains `mlx`; adapter defaults remain `auto`;
`selective_final3_scatter512` remains explicit opt-in. Both Fast digests are
unchanged. Exact identities and actual final pinned packages are documented
in `MLX3D_REPRODUCIBLE_INSTALL.md`; the initial Phase121 MLX0.30.6 request was
upgraded to0.32.4 during install. Wheel/cache manifests and failed attempts
are archived, not overwritten. The user's gs-mps inventory was unchanged.

Phase123 fixed adapter-owned stale/partial module cleanup, including missing
export and import interruption. Successful class reuse is intentionally
load-time-only validation, not live disk tamper detection. Use immutable
source/packages and restart after changes. Phase124 added boundary checks and
build-before-swap updates after a rejected update exposed scene mutation.
Supported input activation rounding and image parity are preserved. Rejected
updates leave the old scene usable. See `CPP_COMPAT_API.md` for exact inputs.

Qualified evidence is limited to Mac mini M4/24GB, macOS26.7.1 arm64,
Python3.13.15, MLX/Metal0.32.4, numpy2.4.2, verified MLX3D0.3.0. Other
devices/versions are untested. Forward rendering only; training/backprop,
concurrent shared render/update and broad portability are not qualified.
Disposable venvs share base Python/system toolchain, not site-packages.

## Diff, discussion, CI, and decision rules

Reviewed the handoff-to-current adapter diff and invariant checks: legacy
renderer and frozen Fast source unchanged; runtime edits are loader cleanup,
boundary validation and atomic scene replacement. This is not a fresh
line-by-line review of every historical experimental kernel. GitHub returned
no inline review threads; fetched discussion consists of author experiment
reports, not independent external approval. The combined status endpoint
returned no entries and checkout has no `.github` directory. Check-run status
was not independently established; **no CI pass is claimed**. Review snapshot
is retained in `qualification_evidence/phase126/github_review.json` with its
reviewed head. Later qualification-only commits do not change runtime code.

The CPU-only aggregate checks required report status/counts, current-adapter
evidence, digests/defaults, legacy build execution, retained install logs,
complete benchmark workload/block counts, GPU gates, and draft status. It
includes a SHA256 manifest of retained evidence. Missing evidence self-test
must fail closed. NOT READY persists until every required check passes.
If those pass with the current documented hardware/CI scope limitations,
the aggregate can report CONDITIONALLY READY; it never merges or marks ready.

## Exact Mac reproduction

From the separate feature worktree, use new report filenames:
```sh
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase125_integration_performance.py --out phase125_integration_performance_NEW.json
```
The runner creates its own pinned disposable runtime and checks fresh ioreg
Device Utilization before each GPU worker. Above30% it exits INCOMPLETE,
preserving artifacts. Do not start another GPU test while busy.

Re-evaluate retained evidence without GPU tests:
```sh
python3 Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase126_release_gate.py --out phase126_release_gate_NEW.json
```
Default aggregation deliberately reads the archived incomplete attempt2 and
exits1. After a completed Phase125 run, retain its JSON/artifacts, point
`--performance` at that report relative to `--evidence`, and re-evaluate all
checks against the same adapter revision. Refresh the GitHub review snapshot
and docs if implementation/review status changes. No explicit release or
merge authorization is implied by any gate result.

## Post-decision update — Phase125 attempt 4 completed (Phase126 rerun pending)

Mac terminal evidence reports `RESULT PHASE125 PASS` for `phase125_integration_performance_4.json`: boundary worker, four small, four Full HD and four high-resolution blocks all executed and passed. Bounded GPU cooldown retries allowed transiently busy readings to settle without relaxing the <=30% requirement before worker launch. This addresses the incomplete-coverage fact underlying the **2026-11-16 NOT READY decision**, but does **not** retroactively change that Phase126 decision: the new report and numeric results have not been independently examined and the CPU-only aggregate has not yet run against the newly completed artifact. Use `--performance` with its absolute path or archive it under `qualification_evidence/phase125/` before rerunning; retain old reports, check evidence provenance/invariants and review updated PR head. Until new Phase126 result, PR remains draft/unmerged; no default promotion.
