#!/usr/bin/env python3
"""Phase 69: compare two guarded capacity reserves with bitwise parity.

Both paths use guarded preflight and the identical sorter. One retains the
native 5% growth reserve. Experimental reserve path expands capacity to 20%
after *observed growth*, avoids shrink to preserve shape. This does not
predict unknown demand and is not a production configuration.
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

def measured(e,cam,retain=False):
    r=e._renderer;before=int(r._capacity)
    t=time.perf_counter();image=e.render(cam);mx.eval(image)
    ms=(time.perf_counter()-t)*1000
    counts=dict(r.last_frame_counts);active=int(counts['active'])
    used=int(counts['capacity'])
    if retain and before>0 and active>before:
        # A future reserve setting; deliberately applied *after* current frame.
        r._capacity=max(int(r._capacity),active*120//100+1024)
    return image,dict(ms=ms,before=before,used=used,after=int(r._capacity),
                      active=active,grew=bool(before and active>before))

def stats(rows,mode):
    arr=np.asarray([x[mode]['ms'] for x in rows])
    return dict(p50=float(np.percentile(arr,50)),p95=float(np.percentile(arr,95)),
                max=float(arr.max()),total=float(arr.sum()),
                growth_events=sum(x[mode]['grew'] for x in rows))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase69_guarded_reserve.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    gs=scene(a.n)
    modes=('guarded','reserve20')
    es={k:PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded',
        tile_sort_impl='auto',tile_range_mode='auto') for k in modes}
    for k,e in es.items():measured(e,camera(a.width,a.height,0),k=='reserve20')
    rng=np.random.default_rng(69)
    def paired(cam,label):
        data={};images={}
        for k in rng.permutation(modes):
            images[k],data[k]=measured(es[k],cam,k=='reserve20')
        if not np.array_equal(np.asarray(images['guarded']),np.asarray(images['reserve20'])):
            raise AssertionError('parity '+label)
        print('PASS',label,' '.join(f"{k}={data[k]['ms']:.2f}ms cap={data[k]['before']}->{data[k]['after']}" for k in modes),flush=True)
        return dict(frame=label,**data)
    seq=[0,1,14,15,16,17,18,19,20,29,30,31,32,33,34,35,44,0,1,44,0,1]
    stress=[paired(pose(a.width,a.height,i),f'pose{i}#{j}') for j,i in enumerate(seq)]
    cam=camera(a.width,a.height,3)
    for j in range(6):paired(cam,'warm'+str(j))
    steady=[paired(cam,'steady'+str(j)) for j in range(24)]
    result=dict(phase=69,n=a.n,resolution=[a.width,a.height],
        stress_stats={k:stats(stress,k) for k in modes},
        steady_stats={k:stats(steady,k) for k in modes},
        stress=stress,steady=steady,process_memory=memory(),
        notes=['Both guarded paths preflight every frame and preserve bitwise parity.',
               'Reserve20 retains post-growth 20% headroom; not a production default.',
               'Process memory is shared and cannot isolate per-policy peak.',
               'One stress sequence; stochastic GPU contention remains possible.'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT stress',result['stress_stats'],'steady',result['steady_stats'],'saved',a.out,flush=True)
if __name__=='__main__':main()
