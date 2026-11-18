#!/usr/bin/env python3
"""Qualify the Phase 128 host camera check, then benchmark against the frozen adapter."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

from qualification_common import (
    ADAPTER, HERE, ROOT, command, digest, invariants, inventory, isolated_runtime,
    memory_pressure_gate, metadata, output_paths, write_report,
)
from run_phase125_integration_performance import wait_for_gpu_gate

BASELINE = "c32ce63f45901fb6a1621452a9ab831fb90394a2"
BASELINE_SHA = "e8f9079908e0d08b28f4af125ed544c76d25b23ed572a273b8b20316db7192ec"
WORKLOADS = {"small": (4096, 640, 360), "fullhd": (700000, 1920, 1080),
             "highres": (700000, 2560, 1440)}

def run_worker(args):
    sys.path.insert(0, str(ROOT))
    import mlx.core as mx
    import numpy as np
    from mlx_gs.renderer import mlx3d_cpp_compat as candidate
    from bench_phase49_breakdown import camera, scene

    if digest(args.reference) != BASELINE_SHA:
        raise ValueError("Unexpected baseline adapter digest")
    spec = importlib.util.spec_from_file_location("_phase128_current_adapter", args.reference)
    current = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(current)
    current.__file__ = str(ADAPTER)
    current._MODULE_NAME = "mlx3d.splatting._phase128_current_fast"
    mx.set_cache_limit(6 * 2**30)
    n, width, height = WORKLOADS[args.workload]
    api = {"current": current, "host": candidate}

    def view(index, lazy):
        cam = camera(width, height, index)
        if lazy:
            # Create a fresh GPU-produced matrix each time, without pre-evaluation.
            cam.W2C = cam.W2C + mx.zeros((4, 4), stream=mx.gpu)
        return cam

    def bits(a, b):
        mx.eval(a, b)
        aa, bb = np.asarray(a), np.asarray(b)
        if (aa.shape != bb.shape or aa.dtype != bb.dtype or not np.isfinite(aa).all()
                or not np.array_equal(aa.view(np.uint8), bb.view(np.uint8))):
            raise AssertionError("Exact output parity failed")

    edge_checks = []
    dtype_names = ("float16", "bfloat16", "float32", "float64", "int8", "uint8",
                   "int16", "uint16", "int32", "uint32", "int64", "uint64")
    for name in dtype_names:
        dtype = getattr(mx, name, None)
        if dtype not in candidate._REAL_DTYPES:
            continue
        if name == "float64":
            # Metal does not evaluate float64. Keep the existing host dtype
            # acceptance while testing the matrix check on an explicit CPU stream.
            stream = mx.cpu
        else:
            stream = mx.gpu
        for value in (0.0, float("nan"), float("inf"), -float("inf")) if name.startswith(("float", "bfloat")) else (0,):
            cameras = {}
            for mode in api:
                cam = view(0, False)
                cam.W2C = mx.full((4, 4), value, dtype=dtype, stream=stream)
                cameras[mode] = cam
            with mx.stream(stream):
                for mode, module in api.items():
                    try:
                        converted = module._camera(cameras[mode])
                    except ValueError as exc:
                        if math.isfinite(value) or str(exc) != "Camera W2C must be finite":
                            raise
                    else:
                        if not math.isfinite(value):
                            raise AssertionError(f"{mode} accepted nonfinite {name}")
                        mx.eval(converted.R, converted.t)
            edge_checks.append({"dtype": name, "case": str(value), "status": "PASS"})

    gaussians = scene(n)
    engines = {mode: module.PersistentCppCompatRenderer(gaussians) for mode, module in api.items()}
    for index in range(3):
        bits(*(engine.render(view(index, False)) for engine in engines.values()))
    rows = []
    for index in range(4):
        for lazy in (False, True)[::1 if (args.block + index) % 2 == 0 else -1]:
            orders = [("current", "host"), ("host", "current")]
            if (args.block + index) % 2:
                orders.reverse()
            for order in orders:
                timings = {}
                images = {}
                for name in order:
                    cam = view(index, lazy)
                    start = time.perf_counter()
                    prepared = api[name]._camera(cam)
                    camera_ms = (time.perf_counter() - start) * 1000
                    mx.eval(prepared.R, prepared.t)
                    cam = view(index, lazy)
                    start = time.perf_counter()
                    images[name] = engines[name].render(cam)
                    mx.eval(images[name])
                    render_ms = (time.perf_counter() - start) * 1000
                    timings[name] = {"camera_ms": camera_ms, "render_ms": render_ms}
                bits(images["current"], images["host"])
                if mx.get_active_memory() + mx.get_cache_memory() > 13 * 2**30:
                    raise MemoryError("Worker exceeded allocator budget")
                rows.append({"lazy": lazy, "order": order, "measurements_ms": timings})
    one_shot = {}
    for lazy in (False, True):
        bits(*(module.render_cpp_compat(gaussians, view(3, lazy)) for module in api.values()))
        one_shot[str(lazy)] = "PASS"
    result = {"status": "PASS", "workload": args.workload, "block": args.block,
              "environment": metadata(), "invariants": invariants(), "rows": rows,
              "edge_checks": edge_checks, "one_shot_bitwise": one_shot,
              "memory": {"active": mx.get_active_memory(), "peak": mx.get_peak_memory(),
                         "cache": mx.get_cache_memory(), "scope": "process-wide"}}
    write_report(args.out, result)

def run_parent(args):
    out, logs = output_paths(args.out)
    before = invariants()
    packages = inventory()
    report = {"phase": 128, "status": "INCOMPLETE", "invariants": before,
              "environment": metadata(), "baseline_sha256": BASELINE_SHA,
              "gpu_gate_attempts": [], "workers": [], "artifacts": str(logs)}
    try:
        if before["adapter_sha256"] == BASELINE_SHA:
            raise RuntimeError("Stage the host-check candidate before running")
        paths = {}
        for name, rev in [("control", BASELINE), ("contract_reference", "6487258")]:
            rc, source, _ = command(["git", "show", f"{rev}:{ADAPTER.relative_to(ROOT)}"])
            if rc:
                raise RuntimeError(f"Failed to freeze {name} adapter")
            paths[name] = logs / (name + ".py")
            paths[name].write_text(source)
        if digest(paths["control"]) != BASELINE_SHA:
            raise RuntimeError("Frozen control adapter digest mismatch")
        with isolated_runtime(logs, with_mlx3d=True) as (py, _, env):
            if command([py, "-m", "pip", "check"], logs/"dependency_check.log", env=env):
                raise RuntimeError("Dependency check failed")

            def child(label, script, flags):
                memory_free = memory_pressure_gate()
                gate = wait_for_gpu_gate(report, attempts=12, delay_seconds=4)
                result_path = logs/(label+".json")
                row = dict(name=label, memory_free_pct=memory_free, gpu_gate=gate)
                report["workers"].append(row)
                rc = command([py, HERE/script, *flags, "--out", result_path],
                             logs/(label+".log"), env=env, timeout=900)
                row.update(returncode=rc, artifact=result_path.name)
                if result_path.exists():
                    row["result"] = json.loads(result_path.read_text())
                if rc or row.get("result", {}).get("status") != "PASS":
                    report["status"] = "FAIL"
                    raise RuntimeError(f"{label} failed. See {label}.log.")
                print("PASS", label, flush=True)
                return row["result"]

            suite = child("contracts", "run_phase124_api_contract.py",
                          ["--worker", "--reference", paths["contract_reference"]])
            if len(suite["checks"]) != 65 or any(row["status"] != "PASS" for row in suite["checks"]):
                report["status"] = "FAIL"
                raise RuntimeError("Required set of 65 passing contracts is incomplete")
            report["summary"] = {}
            for workload in WORKLOADS:
                blocks = []
                for block in range(4):
                    result = child(f"{workload}_{block}", Path(__file__).name,
                                   ["--worker", "--reference", paths["control"],
                                    "--workload", workload, "--block", block])
                    if result["invariants"] != before:
                        report["status"] = "FAIL"
                        raise RuntimeError("Candidate changed during benchmark")
                    blocks.append(result)
                summary = {}
                for lazy in (False, True):
                    for metric in ("camera_ms", "render_ms"):
                        deltas = [
                            statistics.median(r["measurements_ms"]["host"][metric]
                                              - r["measurements_ms"]["current"][metric]
                                              for r in result["rows"] if r["lazy"] == lazy)
                            for result in blocks
                        ]
                        summary[f"{'lazy' if lazy else 'eager'}_{metric}"] = {
                            "median_paired_host_minus_current_ms": statistics.median(deltas),
                            "process_median_deltas_ms": deltas,
                        }
                report["summary"][workload] = summary
            report["status"] = "PASS"
    except Exception as exc:
        report["error"] = str(exc)
    if invariants() != before or inventory() != packages:
        report["status"] = "FAIL"
        report["error"] = "Candidate, source or environment changed"
    write_report(out, report)
    print("RESULT PHASE128 HOST CAMERA", report["status"], out, flush=True)
    return int(report["status"] != "PASS")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--block", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return 0
    return run_parent(args)

if __name__ == "__main__":
    sys.exit(main())
