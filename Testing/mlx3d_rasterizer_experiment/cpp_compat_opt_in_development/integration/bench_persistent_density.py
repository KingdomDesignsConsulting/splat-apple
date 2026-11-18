#!/usr/bin/env python3
"""Phase 45: high-density camera-motion acceptance for persistent CPP renderer.

Run in gs-mps from the separate feature worktree. Synthetic dataset, no COLMAP
or torch_gs._C. Compares rendered pixels bitwise on moving cameras; alternates
timed one-shot and persistent frames to control ordering bias. Results to JSON.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import mlx.core as mx
from mlx_gs.core.gaussians import Gaussians
from mlx_gs.renderer.renderer import render
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer

def scene(n, seed=12345):
    r=np.random.default_rng(seed)
    points=np.column_stack((r.uniform(-1.8,1.8,n),
                             r.uniform(-1.2,1.2,n),
                             r.uniform(2.5,7.0,n))).astype(np.float32)
    return Gaussians(means=mx.array(points),
        scales=mx.array(r.uniform(-3.9,-2.7,(n,3)).astype(np.float32)),
        quaternions=mx.array(np.tile([1.,0.,0.,0.],(n,1)).astype(np.float32)),
        opacities=mx.array(r.uniform(-1.,1.,(n,1)).astype(np.float32)),
        sh_coeffs=mx.array(r.uniform(-.5,.5,(n,1,3)).astype(np.float32)))

def camera(w,h,i):
    w2c=np.eye(4,dtype=np.float32)
    w2c[0,3]=0.035*((i%7)-3)
    w2c[1,3]=0.02*((i%5)-2)
    f=float(min(w,h))*1.2
    return SimpleNamespace(W=w,H=h,fx=f,fy=f,cx=w/2.,cy=h/2.,
                           W2C=mx.array(w2c))

def execute(fn):
    start=time.perf_counter()
    out=fn()
    mx.eval(out)
    return (time.perf_counter()-start)*1000.,np.asarray(out)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--counts',default='200000,400000,700000')
    ap.add_argument('--width',type=int,default=640)
    ap.add_argument('--height',type=int,default=480)
    ap.add_argument('--rounds',type=int,default=8)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    counts=[int(s) for s in a.counts.split(',')]
    if not counts or min(counts)<1 or max(counts)>800000:ap.error('counts must be in 1..800000')
    if min(a.width,a.height,a.rounds)<1:ap.error('dimensions/rounds must be positive')
    if a.out.exists():ap.error('Output exists: '+str(a.out))
    print('Phase 45: high-density parity and paired timing; CUDA/PyTorch not required',flush=True)
    results={'phase':45,'args':vars(a),'cases':[]}
    for n in counts:
        gauss=scene(n)
        persist=PersistentCppCompatRenderer(gauss)
        # Prime both modes and verify parity for camera translation.
        for i in range(3):
            c=camera(a.width,a.height,i)
            _,ref=execute(lambda:render(gauss,c,rasterizer_type='mlx3d_cpp_compat'))
            _,got=execute(lambda:persist.render(c))
            if not np.array_equal(ref,got):
                diff=np.max(np.abs(ref-got))
                raise AssertionError(f'n={n} camera={i} parity failure, max abs {diff}')
        timings={'one_shot':[],'persistent':[]}
        frames=[]
        # Random deterministic order at each viewpoint.
        rng=np.random.default_rng(12345+n)
        for i in range(a.rounds):
            c=camera(a.width,a.height,i+3)
            fns={'one_shot':lambda:render(gauss,c,rasterizer_type='mlx3d_cpp_compat'),
                 'persistent':lambda:persist.render(c)}
            order=list(rng.permutation(['one_shot','persistent']))
            values={}
            for key in order:
                ms,im=execute(fns[key])
                if not np.isfinite(im).all():raise AssertionError(f'n={n} frame={i} {key} nonfinite')
                timings[key].append(ms)
                values[key]=im
            if not np.array_equal(values['one_shot'],values['persistent']):
                raise AssertionError(f'n={n} frame={i} timed moving-camera mismatch')
            frames.append({'camera_step':i+3,'order':order})
        ratios=np.array(timings['persistent'])/np.array(timings['one_shot'])
        med={k:float(np.median(v)) for k,v in timings.items()}
        case={'n':n,'parity_views':3,'timed_parity_frames':a.rounds,
              'median_ms':med,'paired_ratio_median':float(np.median(ratios)),
              'paired_ratio_p10':float(np.percentile(ratios,10)),
              'paired_ratio_p90':float(np.percentile(ratios,90)),
              'times_ms':timings,'frames':frames}
        results['cases'].append(case)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(results,indent=2,default=str)+'\n')
        print(f'PASS n={n} moving-camera parity, one_shot={med["one_shot"]:.3f} ms '
              f'persistent={med["persistent"]:.3f} ms paired_ratio={case["paired_ratio_median"]:.4f}',flush=True)
    print('Saved:',a.out)

if __name__=='__main__':main()
