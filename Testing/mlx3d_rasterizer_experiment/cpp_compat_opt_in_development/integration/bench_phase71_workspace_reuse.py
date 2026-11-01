#!/usr/bin/env python3
"""Phase 71: test reuse of GPU workspaces after revisiting a sort capacity.

Fixed camera, guarded sizing, two capacities. Measures first exposure to the
larger capacity, second exposure after returning to small, and recurring cost.
Every image must match. This is a diagnostic: order/allocator effects remain.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera,memory

def timed(e,cam,cap,reference):
    e._renderer._capacity=cap
    t=time.perf_counter()
    img=e.render(cam);mx.eval(img)
    ms=(time.perf_counter()-t)*1000
    if not np.array_equal(np.asarray(img),reference):raise AssertionError('image parity')
    if e._renderer.last_frame_counts['capacity']!=cap:raise AssertionError('unexpected growth')
    return ms

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase71_workspace_reuse.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height)<1:p.error('bad args')
    gs=scene(a.n);cam=camera(a.width,a.height,3)
    e=PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded')
    img=e.render(cam);mx.eval(img)
    reference=np.asarray(img)
    active=e._renderer.last_frame_counts['active']
    small=max(active*105//100+1024,active+1)
    large=max(active*150//100+1024,small+1)
    schedule=[('small_first',small,4),('large_first',large,4),
              ('small_return',small,4),('large_return',large,4),
              ('small_again',small,4),('large_again',large,4)]
    rows=[]
    for phase,cap,count in schedule:
        times=[]
        for j in range(count):
            ms=timed(e,cam,cap,reference);times.append(ms)
            print('PASS',phase,j,f'{ms:.3f}ms',flush=True)
        rows.append({'phase':phase,'capacity':cap,'first_ms':times[0],
                     'repeat_median_ms':float(np.median(times[1:])),
                     'samples_ms':times})
    result={'phase':71,'n':a.n,'resolution':[a.width,a.height],
            'active_duplicates':int(active),'small_capacity':int(small),
            'large_capacity':int(large),'rows':rows,'process_memory_after':memory(),
            'notes':['Fixed camera and scene, one persistent guarded renderer.',
                     'Repeated shape revisits test allocator/workspace reuse, not GPU timestamps.',
                     'Every image checked bitwise; no production changes.']}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',[(r['phase'],r['first_ms'],r['repeat_median_ms']) for r in rows],flush=True)
if __name__=='__main__':main()
