#!/usr/bin/env python3
"""Phase 66: verify whether frame spikes coincide with a second composite pass.

Count _composite invocations per frame without forcing extra synchronization.
Compare optimized automatic policy with the same renderer configured for
preflight active sizing, preserving bitwise output. Track active count, reserve
before/after, and full-frame wall latency across stressful pose transitions.
Diagnostic only: no production changes or GPU kernel timing claims.
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

def instrument(renderer):
    r=renderer._renderer
    original=r._composite
    calls=[]
    def wrapped(*args,**kwargs):
        calls.append(int(r._capacity))
        return original(*args,**kwargs)
    r._composite=wrapped
    return calls

def timed(renderer,cam,calls):
    calls.clear()
    before=int(renderer._renderer._capacity)
    start=time.perf_counter()
    result=renderer.render(cam)
    mx.eval(result)
    elapsed=(time.perf_counter()-start)*1000
    counts=dict(renderer._renderer.last_frame_counts)
    return result,{'ms':elapsed,'before':before,
                   'composite_passes':len(calls),
                   'pass_capacities':list(calls),
                   'after':int(renderer._renderer._capacity),
                   'active':int(counts['active']),
                   'sorted':int(counts['sorted'])}

def summary(rows,name):
    values=np.asarray([x[name]['ms'] for x in rows],dtype=float)
    return {'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95)),
            'p99':float(np.percentile(values,99)),
            'total':float(values.sum()),
            'retry_frames':sum(x[name]['composite_passes']>1 for x in rows),
            'max':float(values.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase66_overflow_audit.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output already exists')
    if not 1<=a.n<=800000 or min(a.width,a.height)<1:p.error('invalid args')
    gs=scene(a.n)
    engines={name:PersistentCppCompatRenderer(gs,duplicate_sort_mode=mode,
        tile_range_mode='auto',tile_sort_impl='radix4')
        for name,mode in [('auto','auto'),('preflight','active')]}
    calls={name:instrument(engine) for name,engine in engines.items()}
    startcam=camera(a.width,a.height,0)
    for name,engine in engines.items():timed(engine,startcam,calls[name])
    sequence=[0,1,14,15,16,17,18,19,20,29,30,31,32,33,34,35,44,0,1,44,0,1]
    rng=np.random.default_rng(66)
    rows=[]
    for step,index in enumerate(sequence):
        cam=pose(a.width,a.height,index)
        imgs={};results={}
        for name in rng.permutation(list(engines)):
            imgs[name],results[name]=timed(engines[name],cam,calls[name])
        if not np.array_equal(np.asarray(imgs['auto']),np.asarray(imgs['preflight'])):
            raise AssertionError(f'bitwise parity mismatch step {step}')
        row={'step':step,'pose':index,**results};rows.append(row)
        print(f"PASS {step+1}/{len(sequence)} pose={index} "
              f"auto={results['auto']['ms']:.2f}ms/"
              f"{results['auto']['composite_passes']}pass "
              f"preflight={results['preflight']['ms']:.2f}ms/"
              f"{results['preflight']['composite_passes']}pass "
              f"active={results['auto']['active']}",flush=True)
    s={name:summary(rows,name) for name in engines}
    report={'phase':66,'n':a.n,'resolution':[a.width,a.height],
            'stats_ms':s,'rows':rows,'process_memory_after':memory(),
            'notes':['_composite wrapping counts passes, no added GPU synchronization.',
                     'Wall-clock measures entire render and GPU completion, not individual GPU kernel time.',
                     'Retry observed through composite invocation count, not inferred from active/capacity.',
                     'Synthetic scenes only; no source or installed package modifications.']}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT',s,'saved',a.out,flush=True)

if __name__=='__main__':main()
