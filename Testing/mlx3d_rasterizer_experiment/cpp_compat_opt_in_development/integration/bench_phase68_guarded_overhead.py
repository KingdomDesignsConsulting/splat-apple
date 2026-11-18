#!/usr/bin/env python3
"""Phase 68: isolate guarded overhead on stable and camera-change paths.

Two independent renderers, both using the same optimized radix and binary
range selection. Automatic vs guarded: paired bitwise parity, fresh warmup,
steady fixed-pose frames and camera jumps. Count composite calls; report
pre/post-capacity, timing percentiles, and memory. No production changes.
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

def instrument(engine):
    r=engine._renderer
    real=r._composite
    calls=[0]
    def proxy(*args,**kwargs):
        calls[0]+=1
        return real(*args,**kwargs)
    r._composite=proxy
    return calls

def measure(engine,cam,calls):
    r=engine._renderer
    calls[0]=0
    cap0=int(r._capacity)
    t=time.perf_counter()
    im=engine.render(cam)
    mx.eval(im)
    ms=(time.perf_counter()-t)*1000
    c=dict(r.last_frame_counts)
    return im,dict(ms=ms,passes=calls[0],capacity_before=cap0,
                   capacity_used=int(c['capacity']),capacity_after=int(r._capacity),
                   active=int(c['active']),sorted=int(c['sorted']))

def summarize(rows,names):
    out={}
    for name in names:
        a=np.array([r[name]['ms'] for r in rows])
        out[name]=dict(median_ms=float(np.median(a)),
             p95_ms=float(np.percentile(a,95)),max_ms=float(a.max()),
             total_ms=float(a.sum()),
             retry_count=sum(r[name]['passes']>1 for r in rows))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--rounds',type=int,default=24)
    p.add_argument('--out',type=Path,default=Path('phase68_guarded_overhead.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height,a.rounds)<1:p.error('invalid input')
    gs=scene(a.n);names=('auto','guarded')
    engines={n:PersistentCppCompatRenderer(gs,duplicate_sort_mode=n,
                 tile_sort_impl='auto',tile_range_mode='auto') for n in names}
    count={n:instrument(engines[n]) for n in names}
    rng=np.random.default_rng(68)
    def paired(cam,label):
        data={};images={}
        for n in rng.permutation(names):
            images[n],data[n]=measure(engines[n],cam,count[n])
        if not np.array_equal(np.asarray(images['auto']),np.asarray(images['guarded'])):
            raise AssertionError('bitwise mismatch '+label)
        if data['guarded']['passes']!=1:raise AssertionError('guarded retry '+label)
        print('PASS',label,
              ' '.join(f"{n}={data[n]['ms']:.2f}ms/{data[n]['passes']}pass cap={data[n]['capacity_used']}" for n in names),flush=True)
        return dict(frame=label,**data)
    fixed=camera(a.width,a.height,3)
    for j in range(6):paired(fixed,'warmup'+str(j))
    steady=[paired(fixed,'steady'+str(j)) for j in range(a.rounds)]
    jumps=[0,1,14,15,16,17,18,19,20,29,30,31,32,33,34,35,44,0,1,44,0,1]
    stress=[paired(pose(a.width,a.height,i),f'jump{j}_pose{i}') for j,i in enumerate(jumps)]
    result=dict(phase=68,n=a.n,resolution=[a.width,a.height],rounds=a.rounds,
                steady_stats=summarize(steady,names),jump_stats=summarize(stress,names),
                steady=steady,jumps=stress,process_memory=memory(),
                notes=['Same optimized sort/ranges in each policy; only capacity sizing differs.',
                       'Camera fixed identically for measured steady rounds.',
                       'Paired output bitwise parity enforced.',
                       'Timing includes synchronization; does not isolate hardware GPU timestamp.',
                       'Both renderers share allocator, report is not per-renderer memory.'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT steady',result['steady_stats'],'jumps',result['jump_stats'],
          'saved',a.out,flush=True)

if __name__=='__main__':main()
