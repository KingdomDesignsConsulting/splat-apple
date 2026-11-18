#!/usr/bin/env python3
"""Resumable qualification. Only completed whole experiments become checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from qualification_common import HERE, ROOT, clean_env, command, digest, invariants, memory_pressure_gate, metadata, write_report
from run_phase126_release_gate import aggregate

WORKLOADS = ("small", "fullhd", "highres")

def machine_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

def successful_result(path, expected):
    report = json.loads(path.read_text())
    if report.get("status") != "PASS" or report.get("invariants") != expected:
        raise ValueError("Missing PASS or mismatched source/default invariants")
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "standard", "extended"), default="quick")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workload", choices=WORKLOADS, action="append",
                        help="Extended profile: select a diagnostic subset (four blocks per workload)")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a fresh summary filename")
    if args.workload and args.profile != "extended":
        parser.error("--workload requires extended")
    run = args.run_dir.resolve()
    run.mkdir(exist_ok=args.resume)
    lock = run / ".lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        return qualify(args, run)
    finally:
        lock.unlink()

def qualify(args, run):
    before = invariants()
    selected = tuple(w for w in WORKLOADS if w in (args.workload or WORKLOADS))
    manifest = {
        "head": machine_head(), "profile": args.profile, "workloads": list(selected),
        "seed_scheme": {"scene": 12345, "order": "12500 + process block"},
        "invariants": before,
        "scripts": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
    }
    manifest_path = run / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest changed; use a fresh run directory")
    else:
        write_report(manifest_path, manifest)
    stages = [("loader", "run_phase123_source_lifecycle.py", [], False)]
    if args.profile != "quick":
        stages.append(("retained", None, [], False))
    if args.profile == "extended":
        stages.extend([
            ("legacy", "run_phase122_legacy_isolation.py", [], True),
            ("api", "run_phase124_api_contract.py", [], True),
            ("performance", "run_phase125_integration_performance.py",
             [arg for workload in selected for arg in ("--workload-filter", workload)], True),
        ])
    rows = []
    for name, program, flags, gpu in stages:
        folder = run / name
        folder.mkdir(exist_ok=True)
        row = {"name": name, "status": "INCOMPLETE"}
        rows.append(row)
        try:
            checkpoint = folder / "checkpoint.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                result = folder / saved["attempt"] / "result.json"
                if digest(result) != saved["result_sha256"]:
                    raise ValueError("Checkpoint evidence digest mismatch")
                successful_result(result, before)
                row.update(status="PASS", attempt=saved["attempt"], resumed=True)
                continue
            attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))) + 1:03d}"
            attempt.mkdir()
            result = attempt / "result.json"
            row["attempt"] = attempt.name
            if gpu:
                row["memory_free_pct_before_stage"] = memory_pressure_gate()
            if program:
                rc = command([sys.executable, HERE/program, *flags, "--out", result],
                             attempt/"execution.log", timeout=3600, env=clean_env())
                if rc:
                    row["exit_code"] = rc
                    if result.exists() and json.loads(result.read_text()).get("status") == "FAIL":
                        row["status"] = "FAIL"
                    continue
            else:
                retained = aggregate(HERE/"qualification_evidence",
                                     "phase125/phase125_integration_performance_4.json")
                write_report(result, retained)
                if retained["status"] != "PASS":
                    row["status"] = "FAIL"
                    continue
            successful_result(result, before)
            write_report(checkpoint, {"attempt": attempt.name, "result_sha256": digest(result)})
            row["status"] = "PASS"
        except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as exc:
            row["error"] = str(exc)
            if (folder / "checkpoint.json").exists():
                row["status"] = "FAIL"
    states = {row["status"] for row in rows}
    status = "FAIL" if "FAIL" in states else "INCOMPLETE" if "INCOMPLETE" in states else "PASS"
    report = dict(phase=131, status=status, full_fresh_qualification=(
        status == "PASS" and args.profile == "extended" and selected == WORKLOADS),
        manifest=manifest, environment=metadata(), stages=rows, run_dir=str(run))
    write_report(args.out, report)
    print("RESULT PHASE131", status, flush=True)
    return int(status != "PASS")

if __name__ == "__main__":
    sys.exit(main())
