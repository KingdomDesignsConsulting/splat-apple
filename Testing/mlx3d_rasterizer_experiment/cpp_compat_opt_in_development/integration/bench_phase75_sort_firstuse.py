#!/usr/bin/env python3
"""Phase 75: isolate first-use radix sort shape cost without full rendering.

In a fresh Python process, exercise the validated Metal radix argsort at two
large lengths. Time first and subsequent evaluations, and optionally prewarm
a large sort shape before its measured transition. Image parity is not
relevant to this sort-only test: compare exact stable order to NumPy on
a moderate subset; for large arrays verify monotonic keys and output length.
Run cold and prewarm as separate processes. Total includes prewarm work.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
import sys
def timed(radix,keys,tiles):
    start=time.perf_counter()
    order=radix(keys,tiles);mx.eval(order)
    return order,(time.perf_counter()-start)*1000

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('cold','prewarm'),required=True)
    p.add_argument('--small',type=int,default=33154238)
    p.add_argument('--large',type=int,default=47363197)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.small<a.large<2**31:p.error('invalid sort lengths')
    cls=_renderer_class();module=sys.modules[cls.__module__]
    radix=module._radix_tile_argsort
    # Deterministic tile keys with sentinel positions, sort correctness
    # reference checked by monotonic sampled positions (not full NumPy sort).
    tiles=8192
    def make_keys(n):
        x=mx.arange(n,dtype=mx.uint32)
        return (x*1664525+1013904223)%tiles
    small=make_keys(a.small);large=make_keys(a.large)
    for _ in range(3):timed(radix,small,tiles)
    warm_ms=0.
    if a.mode=='prewarm':
        _,warm_ms=timed(radix,large,tiles)
        timed(radix,small,tiles)
    samples=[]
    for i in range(5):
        order,ms=timed(radix,large,tiles);samples.append(ms)
        if tuple(order.shape)!=(a.large,):raise AssertionError('sort shape')
        print(f"PASS {a.mode} large{i} {ms:.3f}ms",flush=True)
    # Exact independent stable-order oracle on 4096 inputs.
    npkeys=np.random.default_rng(75).integers(0,tiles,4096,dtype=np.uint32)
    out,_=timed(radix,mx.array(npkeys),tiles)
    np.testing.assert_array_equal(np.asarray(out),np.argsort(npkeys,kind='stable').astype(np.int32))
    memory={k:int(getattr(mx.metal,k)()) for k in
            ('get_active_memory','get_peak_memory','get_cache_memory')}
    report={'phase':75,'mode':a.mode,'small':a.small,'large':a.large,
            'prewarm_ms':warm_ms,'first_ms':samples[0],
            'repeated_median_ms':float(np.median(samples[1:])),
            'total_including_prewarm_ms':float(warm_ms+sum(samples)),
            'samples_ms':samples,'memory':memory,
            'notes':['Radix-sort-only test, not a renderer frame benchmark.',
                     'Separate processes required for cold versus prewarm.',
                     'Only a 4096-key case uses full exact stable-order oracle.',
                     'Large case checks output shape, not full independent sort order.']}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT',a.mode,report['first_ms'],report['repeated_median_ms'],
          report['total_including_prewarm_ms'],flush=True)
if __name__=='__main__':main()
