#!/usr/bin/env python3
"""Run one isolated matrix at a time; resume with a new report and verified artifacts."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys

from phase138_checkpoint import stable_gpu_gate, utc_now, hash_inventory
from phase140_gpu_lease import (
    AdmissionBusy, atomic_json, checkpoint, lease, require_inherited_lease, run_child, verify,
)
from qualification_common import (
    HERE, ROOT, command, digest, invariants, inventory, isolated_runtime,
    memory_pressure_gate, metadata, output_paths, write_report,
)

PROFILES = {
    "small": (640, 360, 4096),
    "fullhd": (1920, 1080, 700000),
    "highres": (2560, 1440, 700000),
}
DENSITIES = {
    "sparse": (0.72, 0.61, 1.20),
    "moderate": (1.0, 1.0, 1.0),
    "dense": (1.0, 1.14, 0.78),
}
MODES = ("auto", "scatter512", "active", "adaptive", "combined")
MEMORY_CAP = 13 * 2**30

def jobs_for(phase):
    workloads = ("small",) if phase == "smoke" else tuple(PROFILES)
    densities = tuple(DENSITIES) if phase == "140" else ("moderate",)
    blocks = 2 if phase == "140" else 4 if phase == "143" else 1
    modes = MODES if phase == "143" else ("auto", "active")
    jobs = []
    if phase == "143":
        jobs.append(dict(kind="contracts", workload="small", mode="combined",
                         density="moderate", block=0, seed=14301))
    for workload in workloads:
        for density in densities:
            for block in range(blocks):
                shift = block % len(modes)
                order = modes[shift:] + modes[:shift]
                if len(modes) > 2 and block % 2:
                    order = order[::-1]
                for mode in order:
                    jobs.append(dict(kind="render", workload=workload, density=density,
                                     block=block, mode=mode, seed=12345 + block))
    return jobs

def label(job):
    return "_".join(str(job[key]) for key in ("kind", "workload", "density", "block", "mode"))

def worker(args):
    require_inherited_lease()
    if args.out.exists():
        raise FileExistsError(args.out)
    job = json.loads(args.worker)
    sys.path.insert(0, str(ROOT))
    import gc
    import resource
    import time
    from unittest.mock import patch
    from contextlib import ExitStack
    import numpy as np
    import mlx.core as mx
    from mlx_gs.core.gaussians import Gaussians
    from mlx_gs.renderer import mlx3d_cpp_compat as api
    from bench_phase49_breakdown import camera
    from experimental_adaptive_sort import ExperimentalAdaptiveRenderer

    if job["kind"] == "contracts":
        from phase142_combined import run_contracts
        result = run_contracts(args.out, args.reference, job)
        return 0 if result else 1
    mx.set_cache_limit(6 * 2**30)
    w, h, n = PROFILES[job["workload"]]
    count_factor, size_factor, extent = DENSITIES[job["density"]]
    n = int(n * count_factor)
    rng = np.random.default_rng(job["seed"])
    gs = Gaussians(
        means=mx.array(np.column_stack([rng.uniform(-1.8,1.8,n)*extent,
                     rng.uniform(-1.2,1.2,n)*extent,rng.uniform(2.5,7,n)]).astype(np.float32)),
        scales=mx.array((rng.uniform(-3.9,-2.7,(n,3)) + np.log(size_factor)).astype(np.float32)),
        quaternions=mx.array(np.tile([1.,0.,0.,0.],(n,1)).astype(np.float32)),
        opacities=mx.array(rng.uniform(-1.,1.,(n,1)).astype(np.float32)),
        sh_coeffs=mx.array(rng.uniform(-.5,.5,(n,1,3)).astype(np.float32)),
    )
    mode = job["mode"]
    if mode == "combined":
        from phase142_combined import CombinedRenderer
        engine = CombinedRenderer(gs, supported_device=True)
    elif mode == "adaptive":
        engine = ExperimentalAdaptiveRenderer(gs, supported_device=True)
    else:
        engine = api.PersistentCppCompatRenderer(
            gs, duplicate_sort_mode="active" if mode == "active" else "auto",
            tile_sort_impl="selective_final3_scatter512" if mode == "scatter512" else "auto")
    module = sys.modules[api._renderer_class().__module__]
    frames = []
    def snapshot():
        row = dict(active=mx.get_active_memory(), peak=mx.get_peak_memory(),
                   cache=mx.get_cache_memory(),
                   rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if row["active"] + row["cache"] > MEMORY_CAP:
            raise MemoryError("13 GiB combined allocator budget exceeded")
        return row
    def run(tag, width, height, index, trace=False):
        cam = camera(width, height, index)
        routes = []
        def wrap(fn, route):
            def traced(keys, tiles, **kwargs):
                routes.append(dict(route=route, tile_count=tiles, sorted_slots=int(keys.size),
                                   scatter_group=kwargs.get("scatter_group")))
                return fn(keys, tiles, **kwargs)
            return traced
        with ExitStack() as stack:
            if trace:
                for name, route in (("_radix_tile_argsort", "radix4"),
                                    ("_selective_final3_argsort", "final3")):
                    stack.enter_context(patch.object(module, name, wrap(getattr(module, name), route)))
            start = time.perf_counter()
            output = engine.render(cam)
            mx.eval(output)
            ms = 1000 * (time.perf_counter() - start)
        pixels = np.asarray(output)
        if pixels.shape != (height, width, 3) or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
            raise AssertionError("Invalid RGB image")
        counts = dict(engine._renderer.last_frame_counts)
        if trace and not routes:
            routes.append(dict(route="mlx", sorted_slots=counts["sorted"]))
        if trace and routes[-1]["sorted_slots"] != counts["sorted"]:
            raise AssertionError("Actual sort lengths differ from renderer counts")
        frames.append(dict(label=tag, camera=[width,height,index], ms=ms, counts=counts,
                           routes=routes, selection=list(getattr(engine, "last_sort_decisions", [])),
                           policy_ms=getattr(engine, "last_policy_ms", None),
                           rgb=hashlib.sha256(np.ascontiguousarray(pixels)).hexdigest(), memory=snapshot()))
    run("first", w, h, 3 + job["block"])
    for i in range(3):
        run(f"warm_{i}", w, h, 3 + job["block"] + i)
    run("trace", w, h, 3 + job["block"], trace=True)
    small = Gaussians(**{key: getattr(gs, key)[:1024] for key in
                         ("means","scales","quaternions","opacities","sh_coeffs")})
    engine.update(small)
    run("replacement", 640, 360, 5, trace=True)
    engine.update(gs)
    run("restore", w, h, 4 + job["block"], trace=True)
    del engine
    gc.collect()
    write_report(args.out, dict(job=job, status="PASS", invariants=invariants(),
                 runtime=metadata(), dependency_inventory_sha=hash_inventory(inventory()),
                 gaussians=n, size_factor=size_factor, extent=extent, frames=frames,
                 output_checksums={f["label"]: f["rgb"] for f in frames},
                 after_release=snapshot()))
    return 0

def summarize(rows, phase):
    result = {}
    for row in rows:
        job = row["result"]["job"]
        if job["kind"] != "render":
            continue
        result.setdefault((job["workload"], job["density"], job["block"]), {})[job["mode"]] = row["result"]
    matrix = []
    for (workload, density, block), modes in sorted(result.items()):
        if "auto" not in modes:
            continue
        reference = modes["auto"]["output_checksums"]
        control = modes["auto"]
        warm = lambda r: statistics.median(f["ms"] for f in r["frames"] if f["label"].startswith("warm"))
        for mode, report in modes.items():
            if report["output_checksums"] != reference:
                raise AssertionError(f"RGB parity failure: {workload}/{density}/{block}/{mode}")
            frame = next(f for f in report["frames"] if f["label"] == "trace")
            base_frame = next(f for f in control["frames"] if f["label"] == "trace")
            count = base_frame["counts"]
            matrix.append(dict(workload=workload, density=density, block=block, mode=mode,
                 gaussians=report["gaussians"], seed=report["job"]["seed"], camera=frame["camera"],
                 base_count=count, unused_fraction=(count["sorted"]-count["active"])/count["sorted"],
                 warm_ms=warm(report), warm_delta_ms=warm(report)-warm(control),
                 first_ms=report["frames"][0]["ms"], first_delta_ms=report["frames"][0]["ms"]-control["frames"][0]["ms"],
                 memory=report["after_release"], baseline_memory=control["after_release"],
                 actual=frame["routes"], selection=frame["selection"], counts=frame["counts"]))
    return matrix

def parent(args):
    out, logs = output_paths(args.out)
    report = dict(phase=args.phase, status="INCOMPLETE", lock_events=[], gpu_gate_attempts=[],
                  workers=[], artifacts=str(logs), environment=metadata())
    try:
        with lease(report["lock_events"]) as fd, (logs/"run.lock").open("xb") as own:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            jobs = jobs_for(args.phase)
            files = ["run_phase140_143.py","phase140_gpu_lease.py","phase138_checkpoint.py",
                     "qualification_common.py","experimental_adaptive_sort.py",
                     "bench_phase49_breakdown.py","run_phase124_api_contract.py"]
            if args.phase == "143":
                files.append("phase142_combined.py")
            manifest = dict(phase=args.phase, jobs=jobs, invariants=invariants(),
                            scripts={name:digest(HERE/name) for name in files},
                            machine=platform.node(), platform=platform.platform(),
                            python=sys.version, caller_dependencies=hash_inventory(inventory()))
            source = args.resume_from.resolve(strict=True) if args.resume_from else None
            if source:
                with (source/"run.lock").open("rb") as previous:
                    fcntl.flock(previous, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if json.loads((source/"manifest.json").read_text()) != manifest:
                        raise ValueError("Source manifest differs; evidence retained")
            atomic_json(logs/"manifest.json", manifest)
            report["manifest"] = manifest
            runnable = []
            for job in jobs:
                name = label(job)
                reused = verify(source, name, job, manifest) if source else None
                if reused:
                    checkpoint_row, result = reused
                    for suffix in (".json", ".log"):
                        with (logs/(name+suffix)).open("xb") as target:
                            target.write((source/(name+suffix)).read_bytes())
                    # Copy the original checkpoint contents, including original timestamp/path.
                    atomic_json(logs/(name+".checkpoint.json"), checkpoint_row)
                    verify(logs, name, job, manifest)
                    report["workers"].append(dict(name=name, reused_from=str(source),
                                                  checkpoint=checkpoint_row, result=result))
                else:
                    runnable.append(job)
            if runnable:
                with isolated_runtime(logs, with_mlx3d=True) as (py, _, env):
                    if command([py,"-m","pip","check"], logs/"dependency_check.log", env=env):
                        raise RuntimeError("Dependency check failed")
                    rc, runtime_json, _ = command([py,"-c","import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"],env=env)
                    if rc:
                        raise RuntimeError("Cannot inventory pinned worker environment")
                    worker_inventory = hash_inventory(json.loads(runtime_json))
                    report["runtime_dependency_sha256"] = worker_inventory
                    if any(row["result"].get("dependency_inventory_sha") != worker_inventory
                           for row in report["workers"]):
                        raise ValueError("Isolated runtime dependency mismatch on resume")
                    reference = logs/"phase124_reference.py"
                    if args.phase == "143":
                        rc, source_text, _ = command(["git","show","6487258:mlx_gs/renderer/mlx3d_cpp_compat.py"])
                        if rc:
                            raise RuntimeError("Cannot freeze contract reference")
                        reference.write_text(source_text)
                    finished = 0
                    for job in runnable:
                        if args.stop_after is not None and finished >= args.stop_after:
                            raise AdmissionBusy("Requested checkpoint pause")
                        try:
                            gate = stable_gpu_gate(report, settle_seconds=4 if finished else 0)
                            free = memory_pressure_gate()
                        except RuntimeError as exc:
                            raise AdmissionBusy(str(exc)) from exc
                        name = label(job)
                        target = logs/(name+".json")
                        argv = [py, HERE/"run_phase140_143.py", "--worker", json.dumps(job),
                                "--out", target, "--reference", reference]
                        rc = run_child(argv, logs/(name+".log"), fd, env, ROOT)
                        report["last_worker_exit"] = dict(name=name, returncode=rc)
                        if rc or not target.exists():
                            report["status"] = "FAIL"
                            raise RuntimeError(f"Worker {name} failed. See retained log.")
                        result = json.loads(target.read_text())
                        if (result.get("job") != job or result.get("status") != "PASS"
                                or result.get("invariants") != manifest["invariants"]
                                or result.get("dependency_inventory_sha") != worker_inventory
                                or not result.get("output_checksums")):
                            report["status"] = "FAIL"
                            raise ValueError(f"Invalid child result: {name}")
                        row = checkpoint(logs, name, job, manifest, gate, free)
                        report["workers"].append(dict(name=name, checkpoint=row, result=result))
                        finished += 1
                        print("CHECKPOINT", name, flush=True)
            report["matrix"] = summarize(report["workers"], args.phase)
            if len(report["workers"]) != len(jobs):
                raise AssertionError("Incomplete matrix")
            report["status"] = "PASS"
    except (AdmissionBusy, KeyboardInterrupt) as exc:
        report["error"] = str(exc) or "Interrupted"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["reused"] = sum("reused_from" in w for w in report["workers"])
        report["executed_verified"] = len(report["workers"]) - report["reused"]
        report["cooldown_events"] = sum(r["status"] == "BUSY" for group in report["gpu_gate_attempts"] for r in group)
        report["peak_admitted_pct"] = max((r["utilization_pct"] for w in report["workers"]
            for r in w["checkpoint"]["gate"]["confirmations"]), default=None)
        write_report(out,report)
    print("RESULT", args.phase, report["status"], out, report.get("error",""), flush=True)
    return int(report["status"] != "PASS")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("smoke","140","143"))
    p.add_argument("--attest-m4", action="store_true")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--out", required=True,type=Path)
    p.add_argument("--resume-from",type=Path)
    p.add_argument("--reference",type=Path,help=argparse.SUPPRESS)
    p.add_argument("--stop-after",type=int)
    args = p.parse_args()
    if args.worker:
        return worker(args)
    if not args.phase or not args.attest_m4 or platform.system() != "Darwin" or platform.machine() != "arm64":
        p.error("Choose a phase and explicitly attest the Apple M4 machine")
    return parent(args)

if __name__ == "__main__":
    sys.exit(main())
