#!/usr/bin/env python3
"""Run bounded profiling and policy experiments in fresh pinned Python processes."""
import argparse
import importlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

from qualification_common import (HERE, ROOT, command, digest, invariants, isolated_runtime,
                                  memory_pressure_gate, metadata, output_paths, write_report)
from run_phase125_integration_performance import wait_for_gpu_gate

WORKLOADS = {"small": (4096, 640, 360), "fullhd": (700000, 1920, 1080),
             "highres": (700000, 2560, 1440)}
SCRIPTS = {128: "bench_phase128_adapter_overhead", 129: "bench_phase129_first_render",
           130: "bench_phase130_memory_lifecycle", 132: "bench_phase132_renderer_stages",
           133: "bench_phase133_active_sort", 137: "bench_phase137_adaptive"}
BUDGET = 13 * 2**30

def worker(args):
    start = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import mlx.core as mx
    import numpy as np
    from mlx_gs.renderer import mlx3d_cpp_compat as api
    from bench_phase49_breakdown import camera, scene
    imports_ms = 1000 * (time.perf_counter() - start)
    n, w, h = WORKLOADS[args.workload]
    mx.set_cache_limit(6 * 2**30)
    gs = scene(n)
    cam = camera(w, h, 3)

    def timed(fn, evaluate=False):
        start = time.perf_counter()
        result = fn()
        if evaluate:
            mx.eval(result)
        elapsed = 1000 * (time.perf_counter() - start)
        if mx.get_active_memory() + mx.get_cache_memory() > BUDGET:
            raise MemoryError("Experiment exceeded allocator budget")
        return result, elapsed

    def equal(a, b):
        mx.eval(a, b)
        aa, bb = np.asarray(a), np.asarray(b)
        if (aa.shape != bb.shape or aa.dtype != bb.dtype or not np.isfinite(aa).all()
                or not np.array_equal(aa.view(np.uint8), bb.view(np.uint8))):
            raise AssertionError("Failed RGB/alpha bitwise parity")

    report = dict(phase=args.phase[0], workload=args.workload, block=args.block,
                  environment=metadata(), import_ms=imports_ms, scene_seed=12345,
                  invariants=invariants(), status="FAIL")
    try:
        program = importlib.import_module(SCRIPTS[args.phase[0]])
        extra = (args.block,) if args.phase[0] in (133, 137) else ()
        report["result"] = program.run((mx,np,api,gs,cam,timed,equal), *extra)
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = str(exc)
        traceback.print_exc()
    report["memory"] = dict(mlx_peak=mx.get_peak_memory(), mlx_active=mx.get_active_memory(),
                            mlx_cache=mx.get_cache_memory(),
                            process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            cache_limit_bytes=6*2**30)
    write_report(args.out, report)
    print("RESULT", args.phase[0], args.workload, args.block, report["status"], flush=True)
    return int(report["status"] != "PASS")

def parent(args):
    out, logs = output_paths(args.out)
    report = dict(status="INCOMPLETE", environment=metadata(), invariants=invariants(),
                  gpu_gate_attempts=[], workers=[], phases=args.phase, scope="experimental",
                  script_digests={name: digest(HERE/(name+".py")) for name in
                                  [*SCRIPTS.values(), "experimental_adaptive_sort"]})
    try:
        with isolated_runtime(logs, with_mlx3d=True) as (py, _, env):
            if command([py, "-m", "pip", "check"], logs/"dependency_check.log", env=env):
                raise RuntimeError("Pinned environment dependency check failed")
            for phase in args.phase:
                for workload in (args.workload,) if args.workload else WORKLOADS:
                    blocks = 4 if phase in (129, 133, 137) else 1
                    for block in range(blocks):
                        label = f"{phase}_{workload}_{block}"
                        memory = memory_pressure_gate()
                        gate = wait_for_gpu_gate(report, attempts=4, delay_seconds=10)
                        result = logs/(label+".json")
                        row = dict(phase=phase, workload=workload, block=block,
                                   memory_free_pct=memory, gpu_gate=gate, artifact=result.name)
                        report["workers"].append(row)
                        rc = command([py, HERE/"run_overnight_renderer.py", "--worker",
                                      "--phase", phase, "--workload", workload, "--block", block,
                                      "--out", result], logs/(label+".log"), env=env, timeout=600)
                        row["returncode"] = rc
                        if result.exists():
                            row["report"] = json.loads(result.read_text())
                        if rc or row.get("report", {}).get("status") != "PASS":
                            report["status"] = "FAIL"
                            row["error"] = f"Worker failed: {label}; inspect retained log"
                            if "memory" in row.get("report", {}).get("error", "").lower():
                                raise MemoryError(row["error"])
                            print(row["error"], flush=True)
                            continue
                        if row["report"]["invariants"] != report["invariants"]:
                            report["status"] = "FAIL"
                            raise RuntimeError("Worker source/default invariants changed")
                        print("PASS", label, flush=True)
            if report["status"] != "FAIL":
                report["status"] = "PASS"
    except Exception as exc:
        report["error"] = str(exc)
    write_report(out,report)
    print("RESULT OVERNIGHT",report["status"],out,flush=True)
    return int(report["status"] != "PASS")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=SCRIPTS, action="append", required=True)
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a fresh report filename")
    if args.worker and (len(args.phase) != 1 or args.workload is None):
        parser.error("worker requires one phase and a workload")
    return worker(args) if args.worker else parent(args)

if __name__ == "__main__":
    sys.exit(main())
