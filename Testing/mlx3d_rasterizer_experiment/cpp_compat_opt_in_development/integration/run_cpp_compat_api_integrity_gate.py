#!/usr/bin/env python3
"""Non-destructive CPP-compatible API and source-integrity regression gate."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists: " + str(args.out))
    if not args.out.parent.is_dir():
        parser.error("output parent directory must exist")
    report = {"gate": "cpp_compat_api_integrity", "status": "FAIL", "checks": []}
    with tempfile.TemporaryDirectory(prefix="cpp_compat_integrity_") as tmp:
        for name, cmd in [
            ("phase112", [sys.executable, str(HERE / "bench_phase112_default_source_guard.py"),
                          "--out", str(Path(tmp) / "phase112.json")]),
            ("public_dispatch_smoke", [sys.executable, str(HERE / "test_runtime_adapter.py"),
                                       "--n", "4096"]),
            ("persistent_scene_update", [sys.executable, str(HERE / "test_persistent_cpp_compat_api.py")]),
            ("dependency_guard", [sys.executable, str(HERE / "test_phase119_dependency_guard.py")]),
            ("camera_scene_edges", [sys.executable, str(HERE / "test_phase120_camera_scene_edges.py")]),
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            item = {"name": name, "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr}
            if name == "phase112" and result.returncode == 0:
                phase = json.loads((Path(tmp) / "phase112.json").read_text())
                item["source_digest"] = phase["source_digest"]
                required = (phase.get("status") == "PASS"
                            and phase.get("tampered_source_rejected") is True
                            and phase.get("source_restoration") == "PASS"
                            and phase.get("invalid_modes_rejected") is True
                            and phase.get("legacy_default_values") == {
                                "native_fast": "mlx",
                                "persistent_adapter": "auto",
                                "one_shot_adapter": "auto",
                            })
                if not required:
                    item["returncode"] = 1
                    item["stderr"] += "Phase112 report failed required invariants.\n"
            report["checks"].append(item)
            print(("PASS" if item["returncode"] == 0 else "FAIL"), name, flush=True)
            if item["returncode"] != 0:
                break
    if len(report["checks"]) == 5 and all(x["returncode"] == 0 for x in report["checks"]):
        report["status"] = "PASS"
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("RESULT CPP_COMPAT_API_GATE", report["status"], "report", args.out, flush=True)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
