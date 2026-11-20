#!/usr/bin/env python3
"""Phase146: qualify explicitly opt-in runtime reserve-reset guard on the Mac M4."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

from phase138_checkpoint import hash_inventory
from phase140_gpu_lease import require_inherited_lease
from qualification_common import ROOT, invariants, inventory, metadata, write_report
from run_phase140_143 import parent

def jobs(_):
    result = [dict(kind="contracts", workload="small", density="moderate", mode="guard", block=0, seed=14301)]
    for block in range(4):
        for mode in ("keep","guard")[::1 if block in (0,3) else -1]:
            result.append(dict(kind="lifecycle", workload="transition", density="moderate",
                               mode=mode, block=block, seed=12345))
    return result

def summarize(rows, _):
    pairs = []
    contracts = [r["result"] for r in rows if r["result"]["job"]["kind"] == "contracts"]
    if len(contracts) != 1 or contracts[0]["contracts"].get("count") != 65 or contracts[0]["status"] != "PASS":
        raise AssertionError("Phase146 guard contracts are missing or incomplete")
    for block in range(4):
        keep, reset = [next(r["result"] for r in rows if r["result"]["job"]["block"] == block
                            and r["result"]["job"]["mode"] == mode) for mode in ("keep","guard")]
        if keep["output_checksums"] != reset["output_checksums"]:
            raise AssertionError(f"Transition bitwise parity failed in process block {block}")
        control = {r["tag"]:r for r in keep["frames"]}
        candidate = {r["tag"]:r for r in reset["frames"]}
        def delta(prefix):
            return statistics.median(candidate[k]["wall_ms"]-control[k]["wall_ms"]
                                   for k in control if k.startswith(prefix))
        pairs.append(dict(block=block, small_first_delta_ms=delta("small_first_"),
            small_warm_delta_ms=delta("small_warm_"), fullhd_restore_delta_ms=delta("fullhd_restore_"),
            highres_return_delta_ms=delta("highres_return_"), highres_warm_delta_ms=delta("highres_warm_"),
            first_render_delta_ms=candidate["initial"]["wall_ms"]-control["initial"]["wall_ms"],
            constructor_delta_ms=reset["initialization_ms"]["construct"]-keep["initialization_ms"]["construct"],
            peak_delta_bytes=reset["after_release"]["peak"]-keep["after_release"]["peak"],
            retained_cache_delta_bytes=reset["after_release"]["cache"]-keep["after_release"]["cache"]))
    return pairs

def worker(args):
    require_inherited_lease()
    if args.out.exists():
        raise FileExistsError(args.out)
    import gc
    import resource
    import time
    from contextlib import ExitStack
    from unittest.mock import patch
    sys.path.insert(0, str(ROOT))
    started = time.perf_counter()
    import mlx.core as mx
    import numpy as np
    from mlx_gs.renderer import mlx3d_cpp_compat as api
    from bench_phase49_breakdown import scene, camera
    from phase142_combined import CombinedRenderer
    from phase146_reserve_guard import ExperimentalReserveResetRenderer
    import_ms = (time.perf_counter()-started)*1000
    job = json.loads(args.worker)
    if job["seed"] != 12345:
        raise ValueError("Scene helper has a fixed deterministic seed")
    if job["kind"] == "contracts":
        from unittest.mock import patch
        import subprocess
        import phase142_combined as module
        from phase142_combined import run_contracts
        reference = args.out.with_name(args.out.stem + "_reference.py")
        if reference.exists():
            raise FileExistsError(reference)
        source = subprocess.check_output(
            ["git", "show", "6487258:mlx_gs/renderer/mlx3d_cpp_compat.py"],
            cwd=ROOT)
        with reference.open("xb") as stream:
            stream.write(source)
        # The existing 65-case contract suite and five native boundaries
        # now exercise ExperimentalReserveResetRenderer via the opt-in API.
        with patch.object(module, "CombinedRenderer", ExperimentalReserveResetRenderer):
            return 0 if run_contracts(args.out, reference, job) else 1
    mx.set_cache_limit(6*2**30)
    large, small = scene(700000), scene(1024)
    start = time.perf_counter()
    native = api._renderer_class()
    verify_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    engine = (ExperimentalReserveResetRenderer if job["mode"] == "guard"
              else CombinedRenderer)(large, supported_device=True)
    construct_ms = (time.perf_counter()-start)*1000
    module = sys.modules[native.__module__]
    frames = []
    def memory():
        m = dict(active=mx.get_active_memory(),peak=mx.get_peak_memory(),cache=mx.get_cache_memory(),
                 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if m["active"]+m["cache"] > 13*2**30:
            raise MemoryError("Lifecycle worker exceeded 13 GiB allocator ceiling")
        return m
    def frame(tag, w, h, index, trim=False):
        inherited = engine._renderer._capacity
        routes = []
        def wrap(fn,route):
            def wrapped(keys,tiles,**opts):
                routes.append(dict(route=route,slots=int(keys.size),tiles=tiles,
                                   scatter_group=opts.get("scatter_group")))
                return fn(keys,tiles,**opts)
            return wrapped
        cam=camera(w,h,index)
        with ExitStack() as stack:
            for name,route in (("_radix_tile_argsort","radix4"),("_selective_final3_argsort","final3")):
                stack.enter_context(patch.object(module,name,wrap(getattr(module,name),route)))
            tick=time.perf_counter()
            out=engine.render(cam)
            mx.eval(out)
            ms=(time.perf_counter()-tick)*1000
        a=np.asarray(out)
        if a.shape!=(h,w,3) or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError("Invalid image")
        counts=dict(engine._renderer.last_frame_counts)
        if not routes:
            routes.append(dict(route="mlx",slots=counts["sorted"]))
        if routes[-1]["slots"] != counts["sorted"]:
            raise ValueError("Recorded route and slot count diverged")
        frames.append(dict(tag=tag,camera=[w,h,index],wall_ms=ms,counts=counts,
                           inherited=inherited,active_after_frame_reserve=engine._renderer._capacity,
                           guard_decision=getattr(engine, "last_reserve_reset", None),
                           routes=routes,memory=memory(),
                           rgb=hashlib.sha256(np.ascontiguousarray(a)).hexdigest()))
    index=3+job["block"]
    frame("initial",2560,1440,index)
    # Four threshold/shape cases, then repeat to observe sustained reserve/cache.
    small_counts = (1024, 4096, 4097, 1024, 1024, 4096)
    for cycle in range(6):
        count = small_counts[cycle]
        replacement = small if count == 1024 else scene(count)
        engine.update(replacement)
        width = 641 if cycle == 4 else 640
        frame(f"small_first_{cycle}",width,360,index,trim=False)
        if job["mode"] == "guard":
            decision = frames[-1]["guard_decision"]
            expected = (count <= 4096 and width == 640 and frames[-1]["inherited"] >= 2**22)
            if decision is None or decision["applied"] != expected:
                raise AssertionError(f"Unexpected guarded reset {cycle}: {decision}")
        frame(f"small_warm_{cycle}",width,360,index)
        engine.update(large)
        frame(f"fullhd_restore_{cycle}",1920,1080,index)
        frame(f"highres_return_{cycle}",2560,1440,index)
        frame(f"highres_warm_{cycle}",2560,1440,index)
    del engine
    gc.collect()
    write_report(args.out,dict(status="PASS",job=job,invariants=invariants(),
        runtime=metadata(),dependency_inventory_sha=hash_inventory(inventory()),
        initialization_ms=dict(import_libs=import_ms,verify_load=verify_ms,construct=construct_ms),
        frames=frames,after_release=memory(),output_checksums={r["tag"]:r["rgb"] for r in frames}))
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",help=argparse.SUPPRESS)
    parser.add_argument("--reference",help=argparse.SUPPRESS)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--resume-from",type=Path)
    parser.add_argument("--stop-after",type=int)
    parser.add_argument("--attest-m4",action="store_true")
    args=parser.parse_args()
    if args.worker:
        return worker(args)
    if not args.attest_m4 or platform.system()!="Darwin" or platform.machine()!="arm64":
        parser.error("Apple M4 attestation required")
    args.phase="146"
    return parent(args,worker_script=Path(__file__).name,job_factory=jobs,summary_factory=summarize,
                  extra_sources=("phase142_combined.py","phase146_reserve_guard.py",))

if __name__=="__main__":
    sys.exit(main())
