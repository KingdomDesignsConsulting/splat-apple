#!/usr/bin/env python3
"""Read-only MLX3D provenance probe. No pip installs, source edits or imports of GPU kernels."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

EXPECTED_INSTALLED = "4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc"
EXPECTED_LOCAL = "56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710"
ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def git_info(path):
    worktree = path.parent
    process = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5)
    if process.returncode:
        return None
    root = Path(process.stdout.strip())
    result = {"checkout": str(root)}
    for key, cmd in (
        ("head", ["git", "-C", str(root), "rev-parse", "HEAD"]),
        ("origin", ["git", "-C", str(root), "remote", "get-url", "origin"]),
    ):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        result[key] = p.stdout.strip() if p.returncode == 0 else None
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("report exists; refusing overwrite")
    if not args.out.parent.is_dir():
        parser.error("report parent directory must exist")
    spec = importlib.util.find_spec("mlx3d")
    installed = None
    if spec and spec.submodule_search_locations:
        candidate = Path(list(spec.submodule_search_locations)[0]) / "splatting" / "fast.py"
        installed = str(candidate)
    installed_sha = digest(Path(installed)) if installed else None
    package_metadata = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        if "mlx3d" in name or "mlx-3d" in name:
            info = {"name": dist.metadata.get("Name"), "version": dist.version}
            direct = Path(str(dist.locate_file(""))) / f"{name.replace('-', '_')}-{dist.version}.dist-info" / "direct_url.json"
            try:
                info["direct_url"] = json.loads(direct.read_text()) if direct.is_file() else None
            except Exception as exc:
                info["direct_url_error"] = str(exc)
            package_metadata.append(info)
    local_sha = digest(SOURCE)
    report = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "installed_fast_file": installed, "installed_fast_sha256": installed_sha,
        "expected_installed_sha256": EXPECTED_INSTALLED,
        "installed_sha_match": installed_sha == EXPECTED_INSTALLED,
        "installed_checkout": git_info(Path(installed)) if installed else None,
        "distribution_metadata": package_metadata,
        "local_fast_file": str(SOURCE), "local_fast_sha256": local_sha,
        "local_sha_match": local_sha == EXPECTED_LOCAL,
        "status": "BASELINE_MATCH" if installed_sha == EXPECTED_INSTALLED and local_sha == EXPECTED_LOCAL else "BASELINE_NOT_MATCHED",
        "note": "Read-only provenance; matching hashes do not prove a reproducible install. Review origin/commit metadata before pinning.",
    }
    with args.out.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print("RESULT MLX3D_PROVENANCE", report["status"], "report", args.out, flush=True)
    print("INSTALLED", installed or "NOT FOUND", installed_sha or "NO HASH", flush=True)
    print("LOCAL", local_sha or "NO HASH", flush=True)
    if report["status"] != "BASELINE_MATCH":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
