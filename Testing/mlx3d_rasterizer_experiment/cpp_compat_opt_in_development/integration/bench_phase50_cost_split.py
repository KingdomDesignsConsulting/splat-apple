#!/usr/bin/env python3
"""Phase 50: instrument construction, render and allocation timing separately.

Run in gs-mps from the feature worktree. Uses same deterministic synthetic
scene across modes, moving cameras, bitwise parity, no site-packages writes.
This is wall-clock decomposition, not GPU kernel profiling.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer, _renderer_class
from bench_phase49_breakdown import scene, camera

def snapshots():
    out={}
    for name in ('get_active_memory','get_peak_memory','get_cache_memory'):
        fn=getattr(mx,name,None)
        if callable(fn):
            try:out[name]=int(fn())
            except Exception:pass
    return out

def make(gaussians):
    p={'means':gaussians.means,'quats':gaussians.quaternions,
       'scales':mx.exp(gaussians.scales),
       'opacities':mx.sigmoid(gaussians.opacities).reshape((-1,)),
       'colors':mx.clip(gaussians.sh_coeffs[:,0,:]*0.28209479177387814+0.5,0,1),
       'sh_degree':0,'geometry_policy':'cpp_compat','alpha_mode':'exponential',
       't_min':0.0001}
    return _renderer_class()(**p)

def timed(fn):
    t=time.perf_counter();v=fn();ms=(time.perf_counter()-t)*1000
    return v,ms

def fastcam(c):
    from mlx3d.cameras import Camera
    return Camera(R=c.W2C[:3,:3],t=c.W2C[:3,3],fx=c.fx,fy=c.fy,
                  cx=c.cx+.5,cy=c.cy+.5,width=c.W,height=c.H)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--rounds',type=int,default=12)
    p.add_argument('--out',type=Path,default=Path('phase50_cost_split.json'))
    a=p.parse_args()
    if a.n<1 or a.n>800000 or min(a.width,a.height,a.rounds)<1:p.error('invalid parameters')
    if a.out.exists():p.error('output already exists')
    gs=scene(a.n)
    keep,init_ms=timed(lambda:PersistentCppCompatRenderer(gs))
    camera0=fastcam(camera(a.width,a.height,0))
    # Prewarm kernels then force an initial parity check.
    ephemeral,_=timed(lambda:make(gs))
    image1,_=timed(lambda:keep._renderer.render(camera0)['image'])
    image2,_=timed(lambda:ephemeral.render(camera0)['image'])
    mx.eval(image1,image2)
    if not np.array_equal(np.asarray(image1),np.asarray(image2)):raise AssertionError('initial parity')
    rows=[]
    for i in range(a.rounds):
        c=fastcam(camera(a.width,a.height,i+3))
        baseline={}
        # Construct anew on every frame, matching the one-shot API; record GPU
        # waits that the internal constructor may trigger, and render separately.
        for mode in (('fresh','persistent') if i%2==0 else ('persistent','fresh')):
            before=snapshots()
            if mode=='fresh':
                renderer,construction_ms=timed(lambda:make(gs))
            else:
                renderer=keep._renderer;construction_ms=0.0
            output,render_ms=timed(lambda:renderer.render(c)['image'])
            mx.eval(output)
            after=snapshots()
            baseline[mode]=np.asarray(output)
            rows.append({'frame':i,'mode':mode,'construct_ms':construction_ms,
                         'render_ms':render_ms,'total_ms':construction_ms+render_ms,
                         'memory_before':before,'memory_after':after})
        if not np.array_equal(baseline['fresh'],baseline['persistent']):
            raise AssertionError(f'frame {i} parity')
        print('PASS frame',i+1,flush=True)
    summary={}
    for mode in ('fresh','persistent'):
        subset=[r for r in rows if r['mode']==mode]
        summary[mode]={k:float(np.median([r[k] for r in subset]))
                       for k in ('construct_ms','render_ms','total_ms')}
    report={'phase':50,'n':a.n,'resolution':[a.width,a.height],
            'persistent_initialization_ms':init_ms,
            'medians_ms':summary,'rows':rows,'parity_frames':a.rounds,
            'notes':['render() in this MLX3D Fast implementation internally calls mx.eval; render_ms is not enqueue-only',
                     'temporary GPU memory and allocator cache may affect fresh construction costs']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('Medians:',summary,'Saved:',a.out)

if __name__=='__main__':main()
