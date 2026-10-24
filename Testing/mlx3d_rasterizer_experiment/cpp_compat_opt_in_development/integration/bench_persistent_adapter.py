#!/usr/bin/env python3
"""Paired runtime experiment: persistent versus one-shot CPP-compatible renderer.

Run from the feature worktree with gs-mps active. Self-contained synthetic
scene, no COLMAP or torch_gs._C. Reports correctness and median wall time;
the one-shot API is intentionally left unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from mlx_gs.core.gaussians import Gaussians
from mlx_gs.renderer.renderer import render
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer

def make_scene(n):
    rng = np.random.default_rng(12345)
    means = mx.array(np.column_stack((
        rng.uniform(-1.0, 1.0, n),
        rng.uniform(-0.8, 0.8, n),
        rng.uniform(3.0, 6.0, n),
    )).astype(np.float32))
    quats = mx.array(np.tile([1., 0., 0., 0.], (n, 1)).astype(np.float32))
    return Gaussians(
        means=means, scales=mx.full((n,3), -3.0),
        quaternions=quats, opacities=mx.full((n,1), 0.0),
        sh_coeffs=mx.zeros((n,1,3)),
    )

def cam(w, h, step=0):
    w2c = np.eye(4, dtype=np.float32)
    w2c[0,3] = 0.05 * step
    return SimpleNamespace(W=w, H=h, fx=float(min(w,h)), fy=float(min(w,h)),
                           cx=w/2., cy=h/2., W2C=mx.array(w2c))

def timed(call):
    start=time.perf_counter()
    arr=call()
    mx.eval(arr)
    return (time.perf_counter()-start)*1000.,np.asarray(arr)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",type=int,default=4096)
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--height",type=int,default=96)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if min(a.n,a.rounds,a.width,a.height)<1:p.error("all dimensions and rounds must be positive")
    scene=make_scene(a.n)
    persistent=PersistentCppCompatRenderer(scene)
    # Warm both paths and enforce equality at several camera positions.
    for step in (0,1,2):
        camera=cam(a.width,a.height,step)
        _,one=timed(lambda:render(scene,camera,rasterizer_type="mlx3d_cpp_compat"))
        _,persist=timed(lambda:persistent.render(camera))
        if not np.array_equal(one,persist):
            raise AssertionError(f"Persistent output differs at camera step {step}")
    # Verify explicit scene update changes output and matches new one-shot.
    moved=make_scene(a.n)
    moved.means=scene.means+mx.array([.04,0.,0.],dtype=mx.float32)
    persistent.update(moved)
    camera=cam(a.width,a.height,1)
    _,one=timed(lambda:render(moved,camera,rasterizer_type="mlx3d_cpp_compat"))
    _,persist=timed(lambda:persistent.render(camera))
    if not np.array_equal(one,persist):raise AssertionError("Scene update mismatch")
    # Interleaved trials at the same camera; alternate order to limit drift bias.
    times={"one_shot":[],"persistent":[]}
    for i in range(a.rounds):
        camera=cam(a.width,a.height,i%3)
        calls={"one_shot":lambda:render(moved,camera,rasterizer_type="mlx3d_cpp_compat"),
               "persistent":lambda:persistent.render(camera)}
        order=("one_shot","persistent") if i%2==0 else ("persistent","one_shot")
        for name in order:
            elapsed,arr=timed(calls[name])
            if not np.isfinite(arr).all():raise AssertionError(f"Non-finite {name} frame")
            times[name].append(elapsed)
    med={k:float(np.median(v)) for k,v in times.items()}
    report={"n":a.n,"width":a.width,"height":a.height,"rounds":a.rounds,
            "camera_parity_steps":[0,1,2],"update_parity":True,"times_ms":times,
            "median_ms":med,"persistent_over_one_shot":med["persistent"]/med["one_shot"]}
    print("PASS: 3 camera views match bitwise; explicit update matches bitwise")
    print("Median wall ms:",med,"persistent/one-shot:",report["persistent_over_one_shot"])
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        if a.out.exists():p.error(f"Refusing to overwrite {a.out}")
        a.out.write_text(json.dumps(report,indent=2)+"\n")
        print("Saved:",a.out)

if __name__=="__main__":
    main()
