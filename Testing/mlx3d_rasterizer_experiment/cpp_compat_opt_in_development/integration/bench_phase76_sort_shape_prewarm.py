#!/usr/bin/env python3
"""Phase 76: test whether radix first-use preparation is shape-specific.

Run --mode cold, smaller, exact in separate Python processes. Each starts
with the same small-shape warmup. 'smaller' prepares an intermediate radix
shape; 'exact' prepares the target shape. All prewarm time is included in
total work. This is sort-only, not a full render. Separate process runs
control allocator sharing, but not machine thermal/system shader caches.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class

def timed(fn,keys,tiles):
    t=time.perf_counter()
    order=fn(keys,tiles)
    mx.eval(order)
    return order,(time.perf_counter()-t)*1000

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['cold','smaller','exact'],required=True)
    p.add_argument('--small',type=int,default=33154238)
    p.add_argument('--intermediate',type=int,default=39469331)
    p.add_argument('--target',type=int,default=47363197)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not (0<a.small<a.intermediate<a.target<2**31 and a.rounds>=2):
        p.error('invalid sizes or rounds')
    cls=_renderer_class()
    fn=sys.modules[cls.__module__]._radix_tile_argsort
    tiles=8192
    def keys(n):
        x=mx.arange(n,dtype=mx.uint32)
        return (x*1664525+1013904223)%tiles
    small=keys(a.small)
    for _ in range(3):timed(fn,small,tiles)
    prewarm_ms=0.0
    if a.mode!='cold':
        length=a.intermediate if a.mode=='smaller' else a.target
        _,prewarm_ms=timed(fn,keys(length),tiles)
        timed(fn,small,tiles)
    target=keys(a.target)
    samples=[]
    for i in range(a.rounds):
        order,ms=timed(fn,target,tiles)
        if tuple(order.shape)!=(a.target,):raise AssertionError('shape mismatch')
        samples.append(ms)
        print('PASS',a.mode,'round',i,'ms',round(ms,3),flush=True)
    rng=np.random.default_rng(76)
    source=rng.integers(0,tiles,4096,dtype=np.uint32)
    test,_=timed(fn,mx.array(source),tiles)
    np.testing.assert_array_equal(np.asarray(test),
                                  np.argsort(source,kind='stable').astype(np.int32))
    metrics={k:int(getattr(mx,k)()) for k in
             ('get_active_memory','get_peak_memory','get_cache_memory')}
    result=dict(phase=76,mode=a.mode,small=a.small,
                intermediate=a.intermediate,target=a.target,
                prewarm_ms=prewarm_ms,first_target_ms=samples[0],
                repeated_target_median_ms=float(np.median(samples[1:])),
                total_including_prewarm_ms=float(prewarm_ms+sum(samples)),
                samples_ms=samples,memory=metrics,
                notes=['Run modes as separate fresh Python processes.',
                       'Smaller prewarm tests whether scratch cache benefits other sizes.',
                       'Sort-only benchmark, not a whole-render timing.',
                       'Exact stable order checked on independent 4096-key case; target shape only on large input.',
                       'No modification to production renderer.'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',a.mode,result['first_target_ms'],
          result['repeated_target_median_ms'],
          result['total_including_prewarm_ms'],flush=True)

if __name__=='__main__':main()
