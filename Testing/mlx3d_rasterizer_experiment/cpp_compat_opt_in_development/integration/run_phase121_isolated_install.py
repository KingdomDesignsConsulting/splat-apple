#!/usr/bin/env python3
"""Phase121: isolated wheel installation and API gate; never mutate caller env.

Creates a temporary venv, tests absent-MLX3D legacy import, installs pinned
mlx/numpy and verified mlx3d wheel, then runs the five-check integration gate.
Preserves logs/report outside disposable venv. macOS arm64 / Python 3.13.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
EXPECTED_FAST = "4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc"
EXPECTED_LOCAL = "56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710"
FAST_MEMBER = "mlx3d/splatting/fast.py"
ROOT = HERE.parents[3]


def run_step(name, command, output, env, timeout=900):
    with (output / (name + ".log")).open("w") as log:
        log.write("COMMAND: " + " ".join(map(str, command)) + "\n")
        log.flush()
        try:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=env, timeout=timeout, check=False)
            return proc.returncode
        except subprocess.TimeoutExpired:
            log.write("\nTIMEOUT\n")
            return 124


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.out_dir.exists():
        p.error("output directory exists: " + str(args.out_dir))
    if not args.out_dir.parent.is_dir():
        p.error("output parent directory must exist")
    args.out_dir.mkdir()
    report_file = args.out_dir / "phase121_summary.json"
    report = {"phase": 121, "status": "INCOMPLETE", "steps": [],
              "python": sys.version, "platform": platform.platform(),
              "note": "Temporary virtual environment; current gs-mps is untouched"}
    environment = os.environ.copy()
    # Avoid inheriting site-packages or Python import overrides from caller.
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        environment.pop(name, None)

    def step(name, command, timeout=900):
        rc = run_step(name, command, args.out_dir, environment, timeout)
        report["steps"].append({"name": name, "returncode": rc,
                                "log": name + ".log"})
        print(("PASS" if rc == 0 else "FAIL"), name, flush=True)
        if rc:
            raise RuntimeError(f"{name} failed (exit {rc}); see {name}.log")

    try:
        if sys.version_info[:2] != (3, 13) or platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("This recipe is qualified only for macOS arm64 Python 3.13")
        with tempfile.TemporaryDirectory(prefix="splat_apple_phase121_") as temp:
            tmp = Path(temp)
            venv_dir = tmp / "isolated"
            step("create_venv", [sys.executable, "-m", "venv", str(venv_dir)])
            py = str(venv_dir / "bin" / "python")
            step("install_base", [py, "-m", "pip", "install", "--index-url",
                 "https://pypi.org/simple", "mlx==0.30.6", "numpy==2.4.2"])
            # Real missing dependency: the opt-in module must not appear.
            absent_script = (
                "import importlib.util; assert importlib.util.find_spec('mlx3d') is None;"
                "import mlx.core as mx; from mlx_gs.renderer.renderer import render;"
                "print('PASS legacy render module import without mlx3d')"
            )
            step("absent_dependency", [py, "-c", absent_script])
            wheel_dir = tmp / "wheel"
            wheel_dir.mkdir()
            step("download_exact_wheel", [py, "-m", "pip", "download",
                 "--no-deps", "--only-binary=:all:", "--index-url",
                 "https://pypi.org/simple", "--dest", str(wheel_dir),
                 "mlx3d==0.3.0"])
            wheels = sorted(wheel_dir.glob("mlx3d-0.3.0-*.whl"))
            if len(wheels) != 1:
                raise AssertionError("expected one wheel; found " + str(len(wheels)))
            wheel = wheels[0]
            with zipfile.ZipFile(wheel) as archive:
                fast_sha = hashlib.sha256(archive.read(FAST_MEMBER)).hexdigest()
            report["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
            report["wheel_fast_sha256"] = fast_sha
            if fast_sha != EXPECTED_FAST:
                raise AssertionError("published wheel Fast SHA differs from required baseline")
            local = ROOT / "Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py"
            if hashlib.sha256(local.read_bytes()).hexdigest() != EXPECTED_LOCAL:
                raise AssertionError("local validated Fast snapshot changed")
            step("install_exact_wheel", [py, "-m", "pip", "install", "--index-url",
                 "https://pypi.org/simple", str(wheel)])
            # Keep import from target worktree: the scripts put ROOT on sys.path.
            gate_report = args.out_dir.resolve() / "api_gate.json"
            step("five_check_api_gate", [py, str(HERE / "run_cpp_compat_api_integrity_gate.py"),
                                        "--out", str(gate_report)], timeout=1800)
            result = json.loads(gate_report.read_text())
            if result.get("status") != "PASS" or len(result.get("checks", [])) != 5:
                raise AssertionError("five-check gate did not report PASS")
            step("package_inventory", [py, "-m", "pip", "freeze"])
            report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT PHASE121", report["status"], "report", report_file, flush=True)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
