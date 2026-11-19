#!/usr/bin/env python3
"""Phase138: independently measure per-mode process/MLX memory on Mac M4.

No production changes. Every workload/mode/block runs in its own fresh pinned
Python process; GPU entry requires <=30% utilization and >=20% free memory.
Identical rendered frames are compared by full-image SHA256 across processes.
"""
import argparse
import gc
import fcntl
import hashlib
import json
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time

from qualification_common import (
    HERE, ROOT, command, digest, invariants, inventory, isolated_runtime,
    memory_pressure_gate, metadata, output_paths, write_report,
)
from phase138_checkpoint import (
    SCHEMA, completed_checkpoint, copy_verified_checkpoint, manifest_for,
    save_checkpoint, stable_gpu_gate, validate_manifest, write_once,
)

WORKLOADS = {
    "small": (4096, 640, 360),
    "fullhd": (700000, 1920, 1080),
    "highres": (700000, 2560, 1440),
}
MODES = ("auto", "selective_final3", "scatter512", "active", "adaptive")
CACHE_LIMIT_BYTES = 6 * 2**30
TOTAL_BUDGET_BYTES = 13 * 2**30


def run_worker(args):
    # Import GPU libraries only in an isolated, single-mode child process.
    sys.path.insert(0, str(ROOT))
    import mlx.core as mx
    import numpy as np
    from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
    from bench_phase49_breakdown import scene, camera
    from experimental_adaptive_sort import ExperimentalAdaptiveRenderer

    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    report = {
        "phase": 138, "status": "FAIL", "mode": args.mode,
        "workload": args.workload, "block": args.block,
        "invariants": invariants(), "environment": metadata(),
        "memory_scope": ("One renderer mode per fresh process; MLX and process "
                          "counters include scene, runtime, allocator and cache. "
                          "Not isolated kernel allocation or additive totals."),
        "timeline": [], "frames": [],
    }
    gs = None
    engine = None
    output = None

    def snap(label):
        record = {
            "label": label,
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "mlx_cache_bytes": int(mx.get_cache_memory()),
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        report["timeline"].append(record)
        if record["mlx_active_bytes"] + record["mlx_cache_bytes"] > TOTAL_BUDGET_BYTES:
            raise MemoryError("Active MLX plus cache exceeded 13 GiB hard budget")
        return record

    try:
        n, w, h = WORKLOADS[args.workload]
        report.update(resolution=[w, h], gaussians=n, scene_seed=12345,
                      cache_limit_bytes=CACHE_LIMIT_BYTES)
        snap("before_scene")
        gs = scene(n)  # uses numpy seed 12345 in bench_phase49_breakdown
        snap("after_scene")
        if args.mode == "adaptive":
            engine = ExperimentalAdaptiveRenderer(gs, supported_device=args.attest_m4)
        elif args.mode == "active":
            engine = PersistentCppCompatRenderer(gs, duplicate_sort_mode="active")
        else:
            native_sort = "selective_final3_scatter512" if args.mode == "scatter512" else args.mode
            engine = PersistentCppCompatRenderer(gs, tile_sort_impl=native_sort)
        snap("after_constructor")

        def frame(label, index):
            nonlocal output
            view = camera(w, h, index)
            start = time.perf_counter()
            output = engine.render(view)
            mx.eval(output)
            elapsed = (time.perf_counter() - start) * 1000
            snap(label)
            pixels = np.asarray(output)
            if pixels.shape != (h, w, 3) or pixels.dtype != np.float32:
                raise AssertionError(f"Unexpected output shape or dtype: {pixels.shape}, {pixels.dtype}")
            if not np.isfinite(pixels).all():
                raise AssertionError("Nonfinite RGB output")
            checksum = hashlib.sha256(np.ascontiguousarray(pixels).view(np.uint8)).hexdigest()
            row = {
                "label": label, "camera_index": index, "sha256_rgb": checksum,
                "elapsed_ms": elapsed,
                "last_frame_counts": dict(getattr(engine._renderer, "last_frame_counts", {}) or {}),
            }
            if args.mode == "adaptive":
                row["adaptive_decisions"] = list(engine.last_sort_decisions)
            report["frames"].append(row)
            output = None
            return row

        frame("first_frame", args.block + 3)
        for i in range(4):
            frame(f"warm_{i}", args.block + 3 + i)
        # Exercise one smaller replacement and a return to the original scene.
        # Keep original input alive for comparable after-release baseline.
        if args.workload != "small":
            smaller = scene(4096)
            engine.update(smaller)
            snap("after_small_scene_update")
            small_view = camera(640, 360, args.block + 3)
            output = engine.render(small_view)
            mx.eval(output)
            snap("small_scene_render")
            output = None
            del smaller
            engine.update(gs)
            snap("after_original_scene_restore")
            frame("restored_frame", args.block + 3)
        snap("before_release")
        del engine
        engine = None
        gc.collect()
        snap("after_release")
        report["status"] = "PASS"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_report(args.out, report)
    print("RESULT PHASE138 WORKER", report["status"], args.workload,
          args.mode, args.block, args.out, flush=True)
    return 0 if report["status"] == "PASS" else 1


def run_parent(args):
    # Validate resumed source BEFORE creating the new destination directory.
    before = invariants()
    packages = inventory()
    machine = metadata()
    matrix = [(w, b, m) for w in args.workload for b in range(args.blocks)
              for m in MODES]
    manifest = manifest_for(
        invariants=before, packages=packages,
        script_sha=digest(Path(__file__).resolve()),
        helper_sha=digest(HERE / "phase138_checkpoint.py"),
        adaptive_sha=digest(HERE / "experimental_adaptive_sort.py"),
        matrix=matrix, python=machine["python"], platform=machine["platform"],
        attested=args.attest_m4,
    )
    source = validate_manifest(args.resume_from, manifest) if args.resume_from else None
    out, logs = output_paths(args.out)
    write_once(logs / "phase138_manifest.json", manifest)
    # On macOS flock is released automatically after a crash or SIGKILL. The
    # retained lock file allows safe detection of a still-active source run.
    lock = (logs / "phase138_active.lock").open("x")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = {
        "phase": 138, "status": "INCOMPLETE", "environment": machine,
        "qualified_baseline": "M4 forward-only; experimental candidate branch",
        "invariants": before, "experiment_sha256": manifest["experiment_sha256"],
        "helper_sha256": manifest["helper_sha256"],
        "adaptive_sha256": manifest["adaptive_sha256"],
        "gpu_gate_attempts": [], "workers": [], "matrix_size": len(matrix),
        "checkpoint_schema": SCHEMA, "resumed_from": str(source) if source else None,
        "gpu_policy": {"maximum_utilization_pct": 30,
                       "required_consecutive_readings": 2,
                       "confirmation_gap_seconds": 2,
                       "cooldown_seconds": 10,
                       "post_worker_settle_seconds": 4},
        "artifacts": str(logs),
    }
    try:
        ran_new_worker = False
        with isolated_runtime(logs, with_mlx3d=True) as (py, _, env):
            if command([py, "-m", "pip", "check"], logs / "dependency_check.log", env=env):
                raise RuntimeError("Pinned runtime dependency check failed")
            for w in args.workload:
                for block in range(args.blocks):
                    shift = (block * 2) % len(MODES)
                    ordered = list(MODES[shift:] + MODES[:shift])
                    if block % 2:
                        ordered.reverse()
                    for mode in ordered:
                        label = f"{w}_{block}_{mode}"
                        target = logs / (label + ".json")
                        # Only reports with original dual-GPU gates, immutable
                        # worker files and matching runtime/source may be reused.
                        verified = (completed_checkpoint(source, label,
                                    mode=mode, workload=w, block=block,
                                    invariants=before) if source else None)
                        if verified is not None:
                            checkpoint, worker_report = verified
                            copied = copy_verified_checkpoint(source, logs, label, checkpoint)
                            report["gpu_gate_attempts"].append(
                                copied["gpu_gate"]["attempts"])
                            report["workers"].append({
                                "workload": w, "block": block, "mode": mode,
                                "free_memory_pct": copied["free_memory_pct"],
                                "gpu_gate": copied["gpu_gate"],
                                "artifact": target.name, "returncode": 0,
                                "reused_from": copied["reused_from"],
                                "report": worker_report,
                            })
                            print("REUSED VERIFIED PHASE138", label, flush=True)
                            continue
                        free = memory_pressure_gate()
                        gate = stable_gpu_gate(
                            report, attempts=12, cooldown_seconds=10,
                            confirm_seconds=2,
                            settle_seconds=4 if ran_new_worker else 0,
                        )
                        entry = {"workload": w, "block": block, "mode": mode,
                                 "free_memory_pct": free, "gpu_gate": gate,
                                 "artifact": target.name}
                        report["workers"].append(entry)
                        rc = command([py, Path(__file__).resolve(),
                                      "--worker", "--attest-m4", "--workload", w,
                                      "--block", block, "--mode", mode,
                                      "--out", target],
                                     logs / (label + ".log"), env=env, timeout=900)
                        ran_new_worker = True
                        entry["returncode"] = rc
                        if target.exists():
                            entry["report"] = json.loads(target.read_text())
                        if rc or entry.get("report", {}).get("status") != "PASS":
                            report["status"] = "FAIL"
                            raise RuntimeError(f"Worker {label} failed; inspect original log")
                        if entry["report"].get("invariants") != before or any(
                                entry["report"].get(k) != value for k, value in
                                (("phase", 138), ("workload", w),
                                 ("block", block), ("mode", mode))):
                            report["status"] = "FAIL"
                            raise AssertionError("Worker provenance/source/defaults changed")
                        save_checkpoint(logs, label, entry, before)
                        print("PASS CHECKPOINTED", label, flush=True)
            summary = {}
            for w in args.workload:
                grouped = {mode: [] for mode in MODES}
                for row in report["workers"]:
                    if row["workload"] == w:
                        grouped[row["mode"]].append(row["report"])
                for block in range(args.blocks):
                    rows = [r for r in report["workers"]
                            if r["workload"] == w and r["block"] == block]
                    expected = [r for r in rows if r["mode"] == "auto"][0]["report"]
                    expected_frames = {f["label"]: f["sha256_rgb"] for f in expected["frames"]}
                    for row in rows:
                        actual_frames = {f["label"]: f["sha256_rgb"] for f in row["report"]["frames"]}
                        if actual_frames != expected_frames:
                            raise AssertionError("Cross-process image mismatch: " + w + " " + row["mode"])
                summary[w] = {}
                for mode, blocks in grouped.items():
                    if len(blocks) != args.blocks:
                        raise AssertionError("Incomplete mode/block matrix")
                    def memory_at(row, label, key):
                        return next(e[key] for e in row["timeline"] if e["label"] == label)
                    summary[w][mode] = {
                        "blocks": len(blocks),
                        "median_first_frame_peak_mlx_bytes": statistics.median(
                            memory_at(row, "first_frame", "mlx_peak_bytes") for row in blocks),
                        "median_final_peak_mlx_bytes": statistics.median(
                            memory_at(row, "after_release", "mlx_peak_bytes") for row in blocks),
                        "median_final_cache_bytes": statistics.median(
                            memory_at(row, "after_release", "mlx_cache_bytes") for row in blocks),
                        "median_final_rss_hwm_bytes": statistics.median(
                            memory_at(row, "after_release", "process_peak_rss_bytes") for row in blocks),
                        "median_warm_frame_ms": statistics.median(
                            f["elapsed_ms"] for row in blocks for f in row["frames"]
                            if f["label"].startswith("warm_")),
                    }
            report["summary"] = summary
            report["status"] = "PASS"
    except KeyboardInterrupt:
        report["status"] = "INCOMPLETE"
        report["error"] = "Interrupted; completed worker checkpoints are preserved for --resume-from"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if invariants() != before or inventory() != packages:
            report["status"] = "FAIL"
            report["error"] = "Caller environment or protected source changed"
        report["completed_workers"] = sum(
            1 for row in report["workers"] if row.get("returncode") == 0
            and row.get("report", {}).get("status") == "PASS")
        try:
            write_report(out, report)
        finally:
            lock.close()
    print("RESULT PHASE138 MODE_ISOLATED_MEMORY", report["status"], out, flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--workload", action="append", choices=WORKLOADS,
                   help="Select workloads; omit to use all three")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--resume-from", type=Path,
                   help="Resume from a completed/checkpointed Phase138 _artifacts "
                        "directory; always supply a NEW --out and never edit the source")
    p.add_argument("--attest-m4", action="store_true",
                   help="Explicitly attest qualified Apple M4 hardware; mandatory for this experiment")
    args = p.parse_args()
    if args.out.exists():
        p.error("Output already exists; use a fresh filename")
    if not args.attest_m4:
        p.error("Explicit --attest-m4 required; this experiment is M4-only")
    if args.worker:
        if args.resume_from is not None:
            p.error("--resume-from applies only to the parent orchestrator")
        if len(args.workload or []) != 1 or args.mode is None or args.block < 0:
            p.error("Worker requires one --workload, one --mode, and nonnegative --block")
        args.workload = args.workload[0]
        return run_worker(args)
    if args.mode is not None:
        p.error("--mode is for worker use only")
    if not (1 <= args.blocks <= 4):
        p.error("--blocks must be between 1 and 4")
    args.workload = list(dict.fromkeys(args.workload or list(WORKLOADS)))
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
