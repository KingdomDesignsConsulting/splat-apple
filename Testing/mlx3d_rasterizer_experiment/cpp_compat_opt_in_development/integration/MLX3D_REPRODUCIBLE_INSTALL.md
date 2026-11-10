# MLX3D dependency reproducibility — isolated installation plan

**Status: provenance collection implemented; clean install NOT YET QUALIFIED.**

The adapter requires the installed `mlx3d/splatting/fast.py` digest
`4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`,
and a checked-in local Fast snapshot digest
`56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710`.
The existing repo's `requirements.txt` does not install MLX3D, and its
`setup.py` is for PyTorch native extensions, not this opt-in Fast dependency.

## Step 1 — inspect the working environment (read-only)

In `gs-mps`, from the feature worktree:

```sh
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/probe_mlx3d_provenance.py --out mlx3d_provenance_1.json
```

The report includes the installed `fast.py` path, SHA, Python/OS version,
matching distribution metadata, any recorded pip direct URL and checkout
origin/commit when it resides in Git. It does **not** run pip, modify the
environment, edit source, or build kernels. A matching SHA is necessary but
does not establish that the same content can be installed from a pinned
public source. Inspect the report locally before sharing: paths and Git
remotes may contain sensitive information.

## Step 2 — find an immutable upstream source

Determine the exact MLX3D distribution/checkout and immutable revision whose
`mlx3d/splatting/fast.py` hashes to the expected installed digest. Do
not guess a package version or clone an arbitrary `main`. If the checked-out
source is modified/uncommitted, first document that discrepancy separately.
Only after a verified source revision is identified should a reproducible
installer or lockfile pin be committed.

## Step 3 — disposable clean environment (not yet authorized or performed)

Create a **separate** virtual environment or Conda environment; never mutate
`gs-mps` or site-packages. Install this repo's Python dependencies and the
verified immutable MLX3D revision with consistent versions. Verify installed
file hashes *before* any renderer calls. Run the combined API gate and
explicit missing-MLX3D checks in a separate no-MLX3D environment, proving
legacy `python` and `cpp` imports/selection remain independent (a usable
compiled `cpp` extension may require its own build).

Record exact commands, resolved versions, source revision, digest, output
logs, and machine details. Do not claim clean-install success until those
tests actually run on the target machine.

## Blocker disposition

Current evidence: Mac M4 Phase120 five-check runtime gate PASS. This is not
an independently reproducible installation. PR #1 remains draft, 512 mode
remains opt-in, and no source digests or production code were changed.

## New finding — installed package is `mlx3d==0.3.0`

The user-supplied `mlx3d_provenance_1.json` reports `distribution_metadata`
with `mlx3d` version **0.3.0**, `direct_url: null`, matching installed
and local hashes. The apparent `installed_checkout` pointing to
`/opt/homebrew` and `github.com/Homebrew/brew` is **not** MLX3D
source provenance: the probe's Git directory walk crossed into the Homebrew
parent checkout. Do not use that commit as an MLX3D revision.

PyPI has an official `mlx3d-0.3.0-py3-none-any.whl` uploaded
July 22, 2026, and the upstream project is
`https://github.com/amirhossein-razlighi/mlx3D`.
The existence of this wheel **does not yet verify** its embedded Fast source
against the user's installed SHA. Do not upgrade to 0.3.2 or assume equivalence.

### Next gate: download and inspect, without installing

In the original `gs-mps` environment run:

```bash
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/verify_mlx3d_pypi_wheel.py --out mlx3d_wheel_provenance_1.json
```

This uses `python -m pip download --no-deps --only-binary=:all:`
for exactly `mlx3d==0.3.0` from PyPI in a temporary directory, hashes
`mlx3d/splatting/fast.py` inside the wheel, reports the full wheel digest,
and deletes temporary downloaded data. It **does not install** the wheel,
alter `gs-mps`, or overwrite an existing result. A wheel Fast hash match
is a prerequisite to preparing an independently validated clean environment;
it does not itself prove that environment works.

## Phase 121 — published wheel verified; disposable install acceptance pending

On Mac mini M4 the PyPI `mlx3d==0.3.0` wheel's embedded
`mlx3d/splatting/fast.py` matched the installed source hash:
`4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`.
`RESULT MLX3D_WHEEL_PROVENANCE PASS`, local result:
`mlx3d_wheel_provenance_1.json`. This closes the wheel **source identity**
question, but not the clean installation or runtime gate.

The committed `run_phase121_isolated_install.py` creates a temporary Python
3.13 virtual environment (no system site packages), installs pinned
`mlx==0.30.6` and `numpy==2.4.2` from PyPI, confirms that MLX3D is absent
while the legacy renderer module imports, downloads the exact `mlx3d==0.3.0`
wheel, verifies its embedded Fast hash before installing it, then runs the
five-check API integrity gate from the feature worktree in that isolated
interpreter. It records logs, package inventory and JSON evidence to a
unique output directory. The temporary virtual environment is deleted after
the run (including on failure).

Run from the Mac M4 feature worktree after `git pull`:

```bash
python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/run_phase121_isolated_install.py --out-dir phase121_isolated_install_1
```

This will **download and install packages only into the disposable virtual
environment**; it will not modify `gs-mps`. Network access to PyPI is
required, and wheel/dependency downloads may be substantial. The report's
`phase121_summary.json` must say PASS and its `api_gate.json` must
contain five passes before we consider isolated installation qualified.
The absence check validates legacy renderer **import**, not complete
`cpp` and `python` rendering without MLX3D; a separate legacy execution
test is still needed.
