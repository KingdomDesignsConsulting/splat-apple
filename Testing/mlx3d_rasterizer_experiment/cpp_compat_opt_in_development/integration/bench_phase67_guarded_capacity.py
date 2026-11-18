#!/usr/bin/env python3
"""Phase 67: evaluate opt-in guarded capacity against auto and active.

Guarded performs one count readback before expansion, grows on demand and
keeps an unchanged radix-sort shape if existing capacity suffices.
It should never composite twice due to overflow. Test bitwise fidelity and
latency on 22 stress poses plus 12 repeated steady frames.
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

def attach(e):
    r=e._renderer;original=r._composite;count=[0]
    def wrapped(*args,**kwargs):
        count[0]+=1
        return original(*args,**kwargs)
    r._composite=wrapped
    return count

def timed(e,cam,count):
    count[0]=0;r=e._renderer;before=int(r._capacity)
    t=time.perf_counter();img=e.render(cam);mx.eval(img);ms=(time.perf_counter()-t)*1000
    c=dict(r.last_frame_counts)
    return img,{'ms':ms,'passes':count[0],'before':before,
                'active':int(c['active']),'used_capacity':int(c['capacity']),
                'after':int(r._capacity),'sorted':int(c['sorted'])}

def summary(rows,names):
    out={}
    for name in names:
        vals=np.array([r[name]['ms'] for r in rows])
        out[name]={'median':float(np.median(vals)),'p95':float(np.percentile(vals,95)),
                   'maximum':float(vals.max()),'total':float(vals.sum()),
                   'retry_frames':sum(x[name]['passes']>1 for x in rows)}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase67_guarded_capacity.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height)<1:p.error('invalid args')
    gs=scene(a.n);names=('auto','guarded','active')
    engines={name:PersistentCppCompatRenderer(gs,duplicate_sort_mode=name,
             tile_sort_impl='auto',tile_range_mode='auto') for name in names}
    counters={name:attach(engine) for name,engine in engines.items()}
    for name in names:timed(engines[name],camera(a.width,a.height,0),counters[name])
    rng=np.random.default_rng(67)
    def step(cam,label):
        images={};d={}
        for name in rng.permutation(names):
            images[name],d[name]=timed(engines[name],cam,counters[name])
        baseline=np.asarray(images['auto'])
        for name in names:
            if not np.array_equal(baseline,np.asarray(images[name])):
                raise AssertionError(f'parity {label}: {name}')
        if d['guarded']['passes']!=1:
            raise AssertionError(f'guarded unexpectedly retried: {label}')
        print('PASS',label,' '.join(f"{name}={d[name]['ms']:.2f}ms/{d[name]['passes']}pass"
                                    for name in names),flush=True)
        return {'frame':label,**d}
    seq=[0,1,14,15,16,17,18,19,20,29,30,31,32,33,34,35,44,0,1,44,0,1]
    stress=[step(pose(a.width,a.height,i),f'pose{i}#{j}') for j,i in enumerate(seq)]
    for j in range(3):step(camera(a.width,a.height,j+3),'steadywarm'+str(j))
    steady=[step(camera(a.width,a.height,j+6),'steady'+str(j)) for j in range(12)]
    report={'phase':67,'n':a.n,'resolution':[a.width,a.height],
            'stress_stats':summary(stress,names),'steady_stats':summary(steady,names),
            'stress':stress,'steady':steady,'process_memory':memory(),
            'notes':['Independent persistent instances; randomized execution order.',
                     'Every frame checked bitwise across three policies.',
                     'No new GPU instrumentation sync: only wrapper pass counts.',
                     'Policy remains opt-in; do not promote from one synthetic run.']}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT stress',report['stress_stats'],'steady',report['steady_stats'],
          'Saved:',a.out,flush=True)
if __name__=='__main__':main()
