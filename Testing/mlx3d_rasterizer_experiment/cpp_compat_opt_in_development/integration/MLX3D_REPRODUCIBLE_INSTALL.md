# Reproducible M4 opt-in installation qualification

Phase121 installation **PASS** was independently inspected during Phases122–126.
Seven stage logs and the five-check API gate are retained under
`qualification_evidence/phase121/`. This closes the M4 installation blocker;
the overall release decision remains **NOT READY** because Phase125 is incomplete.

## Qualified environment and immutable identities

Mac mini M4, macOS 26.7.1 arm64, Python 3.13.15. Final installed packages:
`mlx==0.32.4`, `mlx-metal==0.32.4`, `numpy==2.4.2`, `pillow==13.0.0`,
`tqdm==4.70.1`, `mlx3d==0.3.0`.
Phase121 initially requested MLX0.30.6, then MLX3D dependency installation
upgraded it to **0.32.4**. It does not qualify an MLX0.30.6 final stack.
The later disposable qualification runners pin the actual final versions.
The caller gs-mps runtime (MLX0.32.3) is separate and was not changed.

- Exact wheel: `mlx3d-0.3.0-py3-none-any.whl`
- Wheel SHA256: `bdfdffe97d2a933afb6bdac2c01ae609c119bf1e22fe129469f4ab03a59439bf`
- Embedded/installed Fast SHA256: `4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`
- Repository Fast SHA256: `56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710`

The earlier provenance walk reached Homebrew's parent checkout; that is
**not** an MLX3D source revision. Qualification uses the verified exact wheel
and file content, not that unrelated Git revision. Other package releases
and source digests remain unqualified and are rejected by the adapter.

## Reproduce without changing gs-mps

From the feature worktree, using a fresh report name:
```sh
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase124_api_contract.py --out phase124_api_contract_NEW.json
conda run --no-capture-output -n gs-mps python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase122_legacy_isolation.py --out phase122_legacy_isolation_NEW.json
```

`qualification_common.py` creates a temporary venv without system site packages,
pins the versions above, verifies the exact MLX3D wheel and its embedded Fast
before installing with `--no-deps`, runs `pip check`, and records commands/logs.
It checks Device Utilization before each GPU worker and blocks above 30%.
The legacy runner installs no MLX3D and builds its CPP extension only in a
disposable source archive using Xcode clang, setuptools80.10.2/nanobind2.11.0.
Base Python and system toolchain are shared; this is not OS-hermetic isolation.

Phase121 obtained the wheel through pip download. During later runs PyPI
transport timed out. The runners used exact locally cached pip wheels after
version/tag, ZIP CRC and hashed RECORD validation; the MLX3D wheel also had
the full expected SHA above. `wheel_cache_manifest.json` records exact wheel
hashes and cache source per run. Cache validation is content verification,
not a fresh online provenance check. If all required cached wheels are not
available, the runner attempts bounded PyPI downloads and fails on missing
dependencies. Retain the manifests/wheels for offline reproduction.

Repository requirements/setup.py alone do not install this optional path.
Do not overwrite site-packages or loosen source hashes. Source verification
is initial-load only; immutable deployments and process restart are required.
No other hardware, Python release or package-version combination is qualified.
