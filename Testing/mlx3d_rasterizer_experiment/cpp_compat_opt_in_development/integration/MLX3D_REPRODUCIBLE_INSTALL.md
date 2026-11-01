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
