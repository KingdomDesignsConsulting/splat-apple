#!/usr/bin/env python3
"""Phase 70: distinguish one-time sort-shape changes from recurring slot cost.

Use the opt-in guarded renderer at a fixed camera. Alternate independently
prewarmed renderers at capacity multipliers, including exact same repeated
capacities. Test image equality for every frame. Report first-frame vs
steady-frame timing per capacity and process-wide memory (not per renderer).
No changes to production source.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera,memory

def render(e,cam):
    t=time.perf_counter();im=e.render(cam);mx.eval(im)
    return im,(time.perf_counter()-t)*1000

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--out',type=Path,default=Path('phase70_capacity_transition.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height,a.rounds)<1:p.error('bad args')
    gs=scene(a.n);cam=camera(a.width,a.height,3)
    baseline=PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded')
    image,_=render(baseline,cam)
    needed=int(baseline._renderer.last_frame_counts['active'])
    capacities=[max(needed+1024,1)*m//100 for m in (105,110,125,150)]
    capacities=[max(c,needed+1024) for c in capacities]
    rng=np.random.default_rng(70)
    rows=[]
    for capacity in rng.permutation(capacities):
        eng=PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded')
        _,_ =render(eng,cam)   # warm geometry and first kernel allocations
        eng._renderer._capacity=int(capacity)
        trial=[]
        for i in range(a.rounds):
            im,ms=render(eng,cam)
            if not np.array_equal(np.asarray(im),np.asarray(image)):
                raise AssertionError('bitwise parity failure at '+str(capacity))
            if int(eng._renderer.last_frame_counts['capacity']) != int(capacity):
                raise AssertionError('unanticipated growth')
            trial.append(ms)
            print(f"PASS cap={capacity} round={i} ms={ms:.3f}",flush=True)
        rows.append(dict(capacity=int(capacity),first_ms=trial[0],
                         median_repeated_ms=float(np.median(trial[1:])) if len(trial)>1 else None,
                         samples_ms=trial))
    result=dict(phase=70,n=a.n,resolution=[a.width,a.height],
                active_duplicates=needed,capacities=capacities,rows=rows,
                process_memory_after=memory(),
                notes=['Fixed camera, guarded preflight, same image every frame.',
                       'First frame after shape change may include lazy allocation or shader compile.',
                       'Subsequent frames show persistent sort work for capacity, not GPU kernel timestamps.',
                       'Run order randomized; memory measured for shared process.'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',[(r['capacity'],r['first_ms'],r['median_repeated_ms']) for r in rows],
          'saved',a.out,flush=True)
if __name__=='__main__':main()
