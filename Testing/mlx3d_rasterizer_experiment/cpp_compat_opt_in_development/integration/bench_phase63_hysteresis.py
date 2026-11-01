#!/usr/bin/env python3
"""Phase 63: benchmark conservative capacity growth and shrink hysteresis.

Compare default automatic renderer to an opt-in *test wrapper* that:
- restores the previous post-render reserve (undoes eager shrinking);
- grows reserve to >= 1.2x last observed active demand after overflow;
- only shrinks if active demand is below 35% of current capacity.
No changes to the rendering kernel or installed packages.
Bitwise parity is required frame by frame. This is a policy prototype;
preallocating capacity never makes overflow impossible.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera,memory
from bench_phase57_active_sort import pose

def sample(renderer,cam,policy=None):
    r=renderer._renderer
    old=int(r._capacity)
    t=time.perf_counter()
    img=renderer.render(cam);mx.eval(img)
    ms=(time.perf_counter()-t)*1000
    counts=dict(r.last_frame_counts)
    active=int(counts['active'])
    used=int(counts['capacity'])
    after=int(r._capacity)
    if policy=='hysteresis':
        # The underlying renderer may have already shrunk its post-frame
        # capacity. Restore what worked and choose a larger reserve after
        # growth, preserving its overflow and rerender safety path.
        if old and active*100>=old*35:
            r._capacity=max(old,after)
        if old and active>old:
            r._capacity=max(r._capacity,active*120//100+1024)
    return img,{'ms':ms,'before':old,'used':used,'after_internal':after,
                'after_policy':int(r._capacity),'active':active,
                'overflow_rerender_likely':bool(old and active>old)}

def stats(rows,name):
    a=np.asarray([x[name]['ms'] for x in rows],dtype=float)
    return dict(p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),
                p99=float(np.percentile(a,99)),max=float(a.max()),
                total=float(a.sum()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase63_hysteresis.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height)<1:p.error('bad args')
    gs=scene(a.n)
    engines={name:PersistentCppCompatRenderer(gs) for name in ('default','hysteresis')}
    for e in engines.values():sample(e,camera(a.width,a.height,0))
    sequence=[0,1,14,15,16,17,18,19,20,29,30,31,32,33,34,35,44,0,1,44,0,1]
    rows=[]
    for k,i in enumerate(sequence):
        cam=pose(a.width,a.height,i)
        images={};d={}
        order=('default','hysteresis') if k%2==0 else ('hysteresis','default')
        for name in order:
            images[name],d[name]=sample(engines[name],cam,name if name=='hysteresis' else None)
        if not np.array_equal(np.asarray(images['default']),np.asarray(images['hysteresis'])):
            raise AssertionError(f'parity failure step {k} pose {i}')
        rows.append(dict(step=k,pose=i,**d))
        print(f"PASS {k+1}/{len(sequence)} pose={i} default={d['default']['ms']:.2f}ms "
              f"hysteresis={d['hysteresis']['ms']:.2f}ms "
              f"caps={d['hysteresis']['before']}->{d['hysteresis']['after_policy']} "
              f"active={d['hysteresis']['active']}",flush=True)
    result={'phase':63,'n':a.n,'resolution':[a.width,a.height],
            'stats_ms':{name:stats(rows,name) for name in engines},
            'samples':rows,'process_memory':memory(),
            'notes':['Prototype modifies capacity only between render calls, never kernels.',
                     'All images must match bitwise. Timing includes rerender and sync.',
                     'Policy may lose on stable frames because of increased sort capacity.',
                     'No guarantee of zero growth on unknown scenes.']}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',result['stats_ms'],'saved',a.out,flush=True)

if __name__=='__main__':main()
