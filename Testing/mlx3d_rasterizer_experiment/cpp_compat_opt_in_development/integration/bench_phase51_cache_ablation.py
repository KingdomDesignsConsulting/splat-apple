#!/usr/bin/env python3
"""Phase 51: isolate reusable duplicate-buffer capacity versus precomputed scene data.

Compares steady-state persistent renderer with a persistent renderer whose tile
capacity is reset before each frame. Both share scene arrays and covariance,
so any delta principally reflects buffer sizing/synchronization and allocation.
Preserves bitwise comparisons. No site-packages changes.
"""
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera

def measure(fn):
    t=time.perf_counter();image=fn();mx.eval(image)
    return (time.perf_counter()-t)*1000,np.asarray(image)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n',type=int,default=700000)
    ap.add_argument('--width',type=int,default=1920)
    ap.add_argument('--height',type=int,default=1080)
    ap.add_argument('--rounds',type=int,default=12)
    ap.add_argument('--out',type=Path,default=Path('phase51_cache_ablation.json'))
    a=ap.parse_args()
    if a.out.exists():ap.error('Output exists')
    if a.n<1 or a.n>800000 or min(a.width,a.height,a.rounds)<1:ap.error('Invalid arguments')
    gs=scene(a.n)
    cached=PersistentCppCompatRenderer(gs)
    reset=PersistentCppCompatRenderer(gs)
    times={'capacity_cached':[],'capacity_reset':[]}
    records=[]
    for i in range(a.rounds+3):
        c=camera(a.width,a.height,i)
        # Reinitializing capacity on a separate object leaves geometry/covariance
        # precomputed but forces fresh duplicate buffer sizing on every frame.
        calls={'capacity_cached':lambda:cached.render(c),
               'capacity_reset':lambda:reset.render(c)}
        order=['capacity_cached','capacity_reset'] if i%2 else ['capacity_reset','capacity_cached']
        images={}
        for mode in order:
            if mode=='capacity_reset':reset._renderer._capacity=0
            ms,image=measure(calls[mode])
            images[mode]=image
            if i>=3:times[mode].append(ms)
            if i>=3:records.append({'frame':i-3,'mode':mode,'ms':ms})
        if not np.array_equal(images['capacity_cached'],images['capacity_reset']):
            raise AssertionError(f'Bitwise mismatch at step {i}')
        if i>=3:print(f"PASS {i-2}/{a.rounds} cache={times['capacity_cached'][-1]:.3f}ms "
                      f"reset={times['capacity_reset'][-1]:.3f}ms",flush=True)
    med={k:float(np.median(v)) for k,v in times.items()}
    result={'phase':51,'n':a.n,'resolution':[a.width,a.height],'rounds':a.rounds,
            'medians_ms':med,'capacity_reset_over_cached':med['capacity_reset']/med['capacity_cached'],
            'samples':records,'notes':[
                'Both paths use persistent precomputed arrays; reset path only invalidates private _capacity.',
                'A capacity reset triggers internal first-frame sizing synchronization; timing is end-to-end wall clock.',
                'Private _capacity is experiment-only, not a proposed production API.']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('Medians:',med,'Saved:',a.out)

if __name__=='__main__':main()
