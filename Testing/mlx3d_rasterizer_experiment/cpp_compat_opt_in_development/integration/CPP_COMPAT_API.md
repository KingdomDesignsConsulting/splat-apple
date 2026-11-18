# CPP-compatible Fast renderer — public opt-in API

This describes the **feature/cpp-compat-renderer** implementation. The existing
`cpp` and `python` backends are unaffected. This renderer is **forward-only**
and must not be selected for gradient-based training.

## Entry points

- `mlx_gs.renderer.renderer.render(gaussians, camera, rasterizer_type="mlx3d_cpp_compat")`:
  public one-shot dispatch. Creates a renderer anew for each call and returns
  an MLX `(height, width, 3)` RGB tensor.
- `mlx_gs.renderer.mlx3d_cpp_compat.render_cpp_compat(gaussians, camera, background=None, *, t_min=0.0001, duplicate_sort_mode="auto", tile_range_mode="auto", tile_sort_impl="auto")`:
  configurable one-shot entry point returning the RGB tensor.
- `PersistentCppCompatRenderer(gaussians, *, t_min=0.0001, duplicate_sort_mode="auto", tile_range_mode="auto", tile_sort_impl="auto")`:
  reusable renderer for camera sequences. Use `.render(camera, background=None)`
  for `(H,W,3)` RGB, and `.update(gaussians)` whenever scene data changes.
  **Not thread-safe**; do not share across concurrent render handlers.

`gaussians` is a Gaussians object or mapping with `means`, `scales`,
`quaternions`, `opacities`, `sh_coeffs`. The adapter expects log scales,
opacity logits, and SH coefficients; it activates scales/opacity and converts
SH-DC into clamped RGB. The camera exposes `W`, `H`, `fx`, `fy`,
`cx`, `cy` and `W2C`. The adapter applies a +0.5 principal-point offset.

## Sorting options

| `tile_sort_impl` | Meaning |
| --- | --- |
| `mlx` | Native MLX sort (underlying Fast renderer default) |
| `radix4` | Stable radix sorting |
| `auto` | Existing adapter default; **not** automatic scatter512 selection |
| `selective_final3` | Validated specialized final pass |
| `selective_final3_scatter512` | Explicit 512-thread intermediate scatter; specialized final pass stays at 256 |

The scatter512 mode requires sort capacity >= 8,000,000 slots, tile count
>= 8,000, and specialized-final-pass eligibility. Otherwise it falls back
to the original radix4 path. Modes preserve stable order and exact output
parity for the qualified M4 scenarios. Native Fast defaults to `mlx`; both
adapter entry points default to `auto`. **Do not change these defaults.**

Example (after preparing `gaussians` and `camera`):
```python
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer

renderer = PersistentCppCompatRenderer(
    gaussians, tile_sort_impl="selective_final3_scatter512"
)
image = renderer.render(camera)
renderer.update(changed_gaussians)  # required when the scene changes
```

## Integrity and deployment

The adapter hashes the checked-in Fast source snapshot and the installed
MLX3D Fast baseline before loading the local implementation. An unmatched
SHA fails closed. Do not patch installed MLX3D or modify
`integration/mlx3d/splatting/fast.py` without a separately qualified,
correctly updated adapter digest.

The current M4 qualification is in `PHASE127_RELEASE_CANDIDATE_AUDIT.md`.
Phase111 found a ~2.12% median Full HD warmed-process improvement for explicit
scatter512; Phase118 found effectively neutral high-resolution performance.
Neither implies a universal gain or justifies default promotion.

## Local regression gate

`run_cpp_compat_api_integrity_gate.py --out <new-report>.json` runs five
subprocess checks: phase112, public_dispatch_smoke, persistent_scene_update,
dependency_guard, and camera_scene_edges. All five passed on M4 in Phase124's
disposable MLX/Metal0.32.4 runtime. The isolated parent runner checks GPU
availability before starting the combined gate; direct gate callers must
check Device Utilization themselves and start only at or below 30%.
Reports are never overwritten. See `MLX3D_REPRODUCIBLE_INSTALL.md` for the
exact dependency identities and `PHASE126_FINAL_RELEASE_DECISION.md` for
the current **CONDITIONALLY READY** decision for the tested M4 source checkout.
Other OS/devices remain unqualified.

## Source-integrity lifecycle (Phase123)

Integrity is checked on the first uncached class load, not continuously.
Successful calls reuse the same cached class even if installed or repository
files subsequently change. Deploy source/package files immutably for the
lifetime of a process and restart after changes; this is not a hot-reload or
live tamper-monitoring API. The private cache-clear hook is qualification
infrastructure, not a supported renderer configuration operation.

Rejected uncached loads are never cached. The adapter removes its owned
module before a new uncached attempt and removes a partially initialized
module on execution/export failure or interruption, then re-raises the
original exception. After the underlying problem is corrected, a new
uncached call can recover. Previously returned class references cannot be
revoked by changing files or clearing this private cache.
## Qualified API boundary contract (Phase124)

The adapter validates metadata before native work. Scene values are real numeric
MLX arrays; qualified formats include float32, float16, and integer means.
Original activation dtype/rounding is retained rather than silently converting
float16 activation math to float32. Arbitrary device/dtype combinations are not
universally qualified.

| Field | Shape / value contract |
| --- | --- |
| means | `(N,3)`, finite |
| scales | `(N,3)` log scales; activated scales must be finite; `-inf` is allowed and activates to zero |
| quaternions | `(N,4)`, finite; zero quaternions retain the validated native behavior |
| opacities | `(N,)` or `(N,1)` logits; NaN rejected, saturated infinite logits allowed |
| sh_coeffs | `(N,K,3)`, `K>=1`; used DC coefficients finite; higher coefficients are ignored |

Mappings and attribute objects are accepted. `N=0`, transparent logits, zero
activated scales and zero quaternions have reference-matched defined behavior.
This does not qualify every possible malformed or degenerate numerical input.

Camera dimensions must be positive integers. `fx`/`fy` must be positive finite
numbers; `cx`/`cy` finite. `W2C` must be a finite real MLX `(4,4)` matrix.
No new rigid-transform or principal-point-inside-image restriction is imposed.
Background is `None` or a finite real MLX array of shape `(3,)`, `(1,3)` or
`(3,1)`, normalized to a three-vector. Finite values outside `[0,1]` are not
clamped by the adapter.

Missing fields and unsupported types/shapes fail with named Python exceptions
before unsafe native construction or rendering. Unsupported mode strings retain
the native renderer's clear `ValueError` behavior. Persistent `update()` builds
a validated replacement before swapping it into use; a rejected replacement
leaves the prior renderer/scene usable. Valid replacements retain the capacity
reserve, but private native object identity and diagnostic counters are not
public API guarantees. Concurrent render/update remains unsupported.

Scene checks run at construction/update; camera/background checks run per
render. These checks have cost, which Phase125 measures rather than assuming
the historical latency is unchanged. Both Fast digests and sorter defaults
remain fixed.
