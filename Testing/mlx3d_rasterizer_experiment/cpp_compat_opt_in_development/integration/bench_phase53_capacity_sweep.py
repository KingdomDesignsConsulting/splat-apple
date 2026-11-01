#!/usr/bin/env python3
"""Phase 53: capacity-sized sort overhead sweep with moving-camera parity.

All capacities are derived from a safely measured warm-frame capacity and
multiplied upward only. Every output is compared bitwise to the baseline.
Uses separate persistent instances and randomized/rotating call order.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera

def timed(fn):
    start=time.perf_counter()
    image=fn()
    mx.eval(image)
    return (time.perf_counter()-start)*1000,np.asarray(image)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",type=int,default=700000)
    p.add_argument("--width",type=int,default=1920)
    p.add_argument("--height",type=int,default=1080)
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--multipliers",default="1,1.1,1.25,1.5,2")
    p.add_argument("--out",type=Path,default=Path("phase53_capacity_sweep.json"))
    a=p.parse_args()
    if a.out.exists():p.error("Output already exists")
    if a.n<1 or a.n>800000 or min(a.width,a.height,a.rounds)<1:p.error("invalid inputs")
    multipliers=[float(x) for x in a.multipliers.split(",")]
    if not multipliers or min(multipliers)<1 or max(multipliers)>3:
        p.error("multipliers must be between 1 and 3")
    gs=scene(a.n)
    base_cam=camera(a.width,a.height,0)
    baseline=PersistentCppCompatRenderer(gs)
    _,base_img=timed(lambda:baseline.render(base_cam))
    initial=baseline._renderer._capacity
    configs=[(f"{v:g}x",v) for v in multipliers]
    engines={}
    for name,v in configs:
        renderer=PersistentCppCompatRenderer(gs)
        renderer._renderer._capacity=int(np.ceil(initial*v))
        ms,im=timed(lambda r=renderer:r.render(base_cam))
        if not np.array_equal(base_img,im):raise AssertionError("baseline mismatch "+name)
        engines[name]=renderer
    data={name:[] for name,_ in configs}
    samples=[]
    rng=np.random.default_rng(53)
    for i in range(a.rounds):
        c=camera(a.width,a.height,i+1)
        outputs={}
        for name in rng.permutation(list(engines)):
            ms,im=timed(lambda r=engines[name]:r.render(c))
            outputs[name]=im
            data[name].append(ms)
            samples.append({"frame":i,"condition":name,"ms":ms,
                            "capacity":engines[name]._renderer._capacity})
        first=outputs[configs[0][0]]
        for name,im in outputs.items():
            if not np.array_equal(im,first):
                raise AssertionError(f"Parity mismatch frame={i} condition={name}")
        print(f"PASS {i+1}/{a.rounds}: "+", ".join(
            f"{name}={data[name][-1]:.2f}ms" for name,_ in configs),flush=True)
    med={name:float(np.median(data[name])) for name,_ in configs}
    report={"phase":53,"n":a.n,"resolution":[a.width,a.height],
            "initial_capacity":initial,"multipliers":multipliers,"rounds":a.rounds,
            "medians_ms":med,"samples":samples,
            "note":"Private _capacity override is benchmark-only. Larger capacity means more sorting work."}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print("Medians:",med,"Saved:",a.out)

if __name__=="__main__":main()
