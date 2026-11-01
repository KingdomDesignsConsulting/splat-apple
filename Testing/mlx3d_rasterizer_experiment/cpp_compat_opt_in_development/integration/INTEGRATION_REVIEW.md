> Current feature development: the source now contains subsequent optimization
> experiments. The historical review below applies to SHA-256 `b6aa0444…`.
> See [OPTIMIZATION_PHASE59.md](OPTIMIZATION_PHASE59.md) for new measurements,
> regression coverage, and limitations. The adapter still checks the exact
> current source digest and the installed-package digest.

# CPP compatibility opt-in — integration review

## Status: reviewed development snapshot; no production merge

Google Drive source: `integration/mlx3d/splatting/fast.py` (35,866 bytes)
SHA-256: `b6aa0444116920d1ae5e526eaf5821e4970bf681c74ce04096fb7f57ae9c11d1`
Original installed MLX3D source SHA-256: `4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`.

The development snapshot is byte-for-byte the same file that was built, smoke-tested, and evaluated in Phases 41–43. The Google Drive copy was fetched and its SHA-256 recalculated.

## Review checks

All 12 static checks passed, implemented in `review_cpp_compat_integration.py`. They cover:

- Separate original and CPP-compatible geometry sources; original Jacobian frustum clamping preserved.
- CPP source unclamped EWA Jacobian, safe depth denominator, 1000-pixel radius cap, conservative offscreen ellipse pruning.
- Original linear-alpha raster source retained; opt-in raster substitutes exponential alpha.
- Constructor defaults `geometry_policy='default'` and `alpha_mode='linear'`, explicit dispatch, validated-mode pairing.

The test suite already executed on the user's Mac mini M4:

- Metal smoke test successful, installed/staged default alpha and RGB bitwise identical.
- Phase 41 400K/700K high-accuracy CPP parity on camera 72.
- Phase 42 14/14 correctness regressions passed at original `t_min=1/255`; Phase 42 camera 0 high-accuracy 400K/700K passed separately.
- Phase 43 14/14 default-mode bitwise comparisons passed, plus clean-GPU paired render ratios from 0.9782x to 1.0107x.

## Caveats

The available source is a *development snapshot*, not a change committed into the actual upstream `mlx3d` repository or the installed `gs-mps` environment. Runtime tests have been executed against the staged source and kernel dispatch, but an upstream project merge, package build, and post-merge validation still require the actual source-tree target. Do not overwrite site-packages. The scripts and source currently exist under the experiment's `integration/` directory.

## Repeat local static review

From `splat-apple` root:

```sh
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/review_cpp_compat_integration.py \
  Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py
```

## Suggested merge method

Integrate the reviewed source as an opt-in renderer change in the real MLX3D development checkout, using a dedicated branch. Re-run the smoke test and at least Phase 43 on that checkout before distributing a package. Preserve the original default behavior and keep CPP parity opt-in.
