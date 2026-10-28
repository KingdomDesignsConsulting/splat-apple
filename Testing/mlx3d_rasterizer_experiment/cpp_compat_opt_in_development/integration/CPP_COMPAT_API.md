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

The latest documented M4 qualification is in `PHASE118_QUALIFICATION_DECISION.md`.
Phase111 found a ~2.12% median Full HD warmed-process improvement for explicit
scatter512; Phase118 found effectively neutral high-resolution performance.
Neither implies a universal gain or justifies default promotion.

## Local regression gate

Run `run_cpp_compat_api_integrity_gate.py --out <new-report>.json` from this
feature worktree in `gs-mps`. This calls the Phase112 source/default/tamper
check and the public dispatch smoke test in separate subprocesses, captures
their logs, and writes a JSON summary without mutating source files.
It refuses to overwrite existing reports. The test needs the validated
MLX3D installation and Apple Silicon MLX runtime; no CI or Mac validation is
claimed until executed on the target machine.
