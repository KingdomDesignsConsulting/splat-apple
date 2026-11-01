#!/usr/bin/env python3
"""Phase 49: decompose persistent vs one-shot cost without changing kernels.

Synthetic 700K 1080p scene; paired camera frames, initialized vs steady state,
and optional MLX active/peak memory counters. Every paired frame bitwise-equal.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import mlx.core as mx
from mlx_gs.core.gaussians import Gaussians
from mlx_gs.renderer.renderer import render
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer

def scene(n):
    rng=np.random.default_rng(12345)
    return Gaussians(
        means=mx.array(np.column_stack([rng.uniform(-1.8,1.8,n),
                                        rng.uniform(-1.2,1.2,n),
                                        rng.uniform(2.5,7,n)]).astype(np.float32)),
        scales=mx.array(rng.uniform(-3.9,-2.7,(n,3)).astype(np.float32)),
        quaternions=mx.array(np.tile([1.,0.,0.,0.],(n,1)).astype(np.float32)),
        opacities=mx.array(rng.uniform(-1.,1.,(n,1)).astype(np.float32)),
        sh_coeffs=mx.array(rng.uniform(-.5,.5,(n,1,3)).astype(np.float32)))

def camera(w,h,i):
    w2c=np.eye(4,dtype=np.float32)
    w2c[0,3]=.035*((i%7)-3)
    w2c[1,3]=.02*((i%5)-2)
    f=float(min(w,h))*1.2
    return SimpleNamespace(W=w,H=h,fx=f,fy=f,cx=w/2,cy=h/2,W2C=mx.array(w2c))

def memory():
    result={}
    for key in ('get_active_memory','get_peak_memory','get_cache_memory'):
        fn=getattr(mx,key,None)
        if callable(fn):
            try: result[key]=int(fn())
            except Exception: pass
    return result

def timed(fn):
    t=time.perf_counter()
    obj=fn()
    construction_ms=(time.perf_counter()-t)*1e3
    mx.eval(obj)
    total_ms=(time.perf_counter()-t)*1e3
    return construction_ms,total_ms,np.asarray(obj)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--rounds',type=int,default=12)
    p.add_argument('--out',type=Path,default=Path('phase49_breakdown.json'))
    a=p.parse_args()
    if a.n<1 or a.n>800000 or min(a.width,a.height,a.rounds)<1:p.error('invalid arguments')
    if a.out.exists():p.error(f'refusing to overwrite {a.out}')
    gs=scene(a.n); c=camera(a.width,a.height,0)
    t=time.perf_counter(); pr=PersistentCppCompatRenderer(gs)
    constructor_ms=(time.perf_counter()-t)*1e3
    first_p=timed(lambda:pr.render(c)); first_o=timed(lambda:render(gs,c,rasterizer_type='mlx3d_cpp_compat'))
    if not np.array_equal(first_p[2],first_o[2]):raise AssertionError('initial parity')
    print('PASS initial parity; persistent construction',round(constructor_ms,3),'ms',flush=True)
    # Warm both paths (do not count in timed round measurements).
    for i in range(3):
        c=camera(a.width,a.height,i)
        _,_,pimg=timed(lambda:pr.render(c))
        _,_,oimg=timed(lambda:render(gs,c,rasterizer_type='mlx3d_cpp_compat'))
        if not np.array_equal(pimg,oimg):raise AssertionError(f'warm parity step {i}')
    times={'one_shot':[],'persistent':[]}
    sampled=[]
    rng=np.random.default_rng(4949)
    for i in range(a.rounds):
        c=camera(a.width,a.height,i+3)
        calls={'one_shot':lambda:render(gs,c,rasterizer_type='mlx3d_cpp_compat'),
               'persistent':lambda:pr.render(c)}
        order=list(rng.permutation(['one_shot','persistent']))
        outputs={}
        for key in order:
            pre=memory(); construction,total,out=timed(calls[key]); post=memory()
            outputs[key]=out
            rec={'camera_step':i+3,'mode':key,'construction_ms':construction,
                 'eval_ms':total-construction,'total_ms':total,'memory_before':pre,'memory_after':post}
            sampled.append(rec);times[key].append(total)
        if not np.array_equal(outputs['one_shot'],outputs['persistent']):
            raise AssertionError(f'frame parity failure {i}')
        print(f"PASS {i+1}/{a.rounds}: one-shot={times['one_shot'][-1]:.2f}ms "
              f"persistent={times['persistent'][-1]:.2f}ms",flush=True)
    median={key:float(np.median(value)) for key,value in times.items()}
    report={'phase':49,'n':a.n,'resolution':[a.width,a.height],
            'persistent_constructor_ms':constructor_ms,
            'initial_render_ms':{'persistent':first_p[1],'one_shot':first_o[1]},
            'initial_construction_ms':{'persistent':first_p[0],'one_shot':first_o[0]},
            'median_steady_ms':median,'paired_ratio_median':float(np.median(np.asarray(times['persistent'])/np.asarray(times['one_shot']))),
            'samples':sampled,'correctness_frames':a.rounds}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('Median steady-state ms:',median,'paired ratio',report['paired_ratio_median'])
    print('Saved:',a.out)

if __name__=='__main__':main()
