# PR #1 — Final integration-readiness audit (2026-10-29)

**Decision: NOT READY for general merge or default promotion.** Keep draft.
This is a code/metadata audit plus previous Mac M4 evidence, not a fresh
clean-environment installation or new device test.

## Reviewed scope

Compared branch against `main` (163 commits; 106 changed files reported by
GitHub), inspected public dispatch, adapter, setup.py, requirements.txt,
README, API documentation, and confirmed Phase120 five-check gate PASS.
The large experimental history is not a complete fresh line-by-line kernel
security review and must not be presented as such.

## Findings by severity

### BLOCKER — opt-in dependency installation is not reproducible

`requirements.txt` contains MLX but does not install MLX3D. `setup.py`
builds the separate PyTorch extension and does not supply MLX3D. The opt-in
adapter lazily imports `mlx3d.splatting.fast`, **then pins its installed
source file to SHA256
`4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`**.
The local validated Fast snapshot is separately pinned to
`56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710`.
There is no documented reproducible installer, immutable upstream revision,
or verified clean-environment setup for that installed Fast baseline.
The user's `gs-mps` installation passes; other installs may correctly fail
closed. Do not loosen hashes without independent source qualification.

**Required before general merge:** supply an independently reproducible
installation recipe and verify on a clean environment, including absence of
MLX3D for legacy paths. Alternatively explicitly scope integration to
developer-only/test-only with no claim of portable installation.

### IMPORTANT — source verification is load-time, not continuous

`_renderer_class` uses `lru_cache(maxsize=1)`: installed/local Fast hashes
are checked when the class is first loaded. Subsequent calls use the cached
class and do not rehash a modified disk file. The module remains loaded in
`sys.modules` after successful import. This is **load-time fail-closed**,
not tamper detection during an active process. Document accurately; do not
mistake Phase112 mock rejection for a live hot-reload security guarantee.

### IMPORTANT — forward-only and integration boundaries

The public `render(..., rasterizer_type="mlx3d_cpp_compat")` is a separate
explicit dispatch and bypasses ordinary `cpp`/`python` rasterizers.
The adapter returns only `(H,W,3)` RGB. Persistent renderer consumers must
call `update()` after Gaussian changes and must not share it across
concurrent calls. No gradient/backprop safety contract is established.
Legacy `cpp`/`python` behavior was inspected but not comprehensively
retested with absent MLX3D or installed dependency changes.

### IMPORTANT — evidence and promotion limits

Phase120 M4 five-check gate PASS. Phase111 Full HD opt-in benefit is scoped
to measured conditions; Phase118 high-resolution randomized replication
found no established advantage. Per-process high-water memory isn't
per-mode allocation. Some raw JSON/log results remain only on the user's Mac.
No other devices or MLX3D variants have been qualified. 512 remains opt-in.

## Acceptance checklist

- [x] Explicit forward-only dispatch, preserved legacy defaults (code review)
- [x] Five-check source/API/dependency-mock/camera/scene gate (Mac M4)
- [x] Experimental speed claims bounded by evidence
- [x] Documentation identifies exact installed/local hashes
- [ ] Reproducible clean-environment MLX3D install and first-run acceptance
- [ ] Real missing-MLX3D legacy backend smoke under clean environment
- [ ] Review/update source verification contract for load-time-only behavior
- [ ] Additional device/version qualification before portability claims
- [ ] Archive or attach raw evidence if audit-grade reproduction required

## Recommendation

Keep PR #1 **draft and unmerged**. Preserve existing hashes and defaults.
Next qualified engineering step: create a separate clean-env installation
and negative-dependency verification procedure that does not alter source,
or explicitly narrow intended release scope to developer opt-in only.
