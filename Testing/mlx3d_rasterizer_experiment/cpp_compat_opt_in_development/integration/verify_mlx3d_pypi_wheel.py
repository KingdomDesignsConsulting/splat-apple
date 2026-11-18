#!/usr/bin/env python3
"""Verify published mlx3d 0.3.0 wheel against the known Fast source.

Downloads to a disposable directory, does NOT install packages or edit gs-mps.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EXPECTED_FAST_SHA = "4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc"
VERSION = "0.3.0"
FAST_MEMBER = "mlx3d/splatting/fast.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("output already exists; refusing overwrite")
    if not args.out.parent.is_dir():
        p.error("report parent does not exist")
    report = {"test": "mlx3d_published_wheel_fast_hash", "version": VERSION,
              "expected_fast_sha256": EXPECTED_FAST_SHA, "status": "INCOMPLETE"}
    try:
        with tempfile.TemporaryDirectory(prefix="mlx3d_wheel_verify_") as td:
            command = [sys.executable, "-m", "pip", "download", "--no-deps",
                       "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                       "--dest", td, f"mlx3d=={VERSION}"]
            run = subprocess.run(command, capture_output=True, text=True, timeout=180)
            report["pip_exit_code"] = run.returncode
            if run.returncode:
                raise RuntimeError("pip download failed: " + run.stderr[-1000:])
            wheels = list(Path(td).glob("mlx3d-0.3.0-*.whl"))
            if len(wheels) != 1:
                raise AssertionError(("expected exactly one mlx3d 0.3.0 wheel", len(wheels)))
            wheel = wheels[0]
            report["wheel_filename"] = wheel.name
            report["wheel_sha256"] = sha(wheel.read_bytes())
            with zipfile.ZipFile(wheel) as archive:
                if FAST_MEMBER not in archive.namelist():
                    raise AssertionError("wheel lacks mlx3d/splatting/fast.py")
                report["wheel_fast_sha256"] = sha(archive.read(FAST_MEMBER))
            report["fast_sha_matches"] = report["wheel_fast_sha256"] == EXPECTED_FAST_SHA
            report["status"] = "PASS" if report["fast_sha_matches"] else "FAIL_FAST_HASH_MISMATCH"
    except Exception as exc:
        report["error"] = str(exc)
        if report["status"] == "INCOMPLETE":
            report["status"] = "ERROR"
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("RESULT MLX3D_WHEEL_PROVENANCE", report["status"],
          "wheel_fast_sha256", report.get("wheel_fast_sha256", "unavailable"),
          "report", args.out, flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
