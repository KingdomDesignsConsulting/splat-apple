#!/usr/bin/env python3
"""Phase 55: fixed-capacity sweep with bitwise parity and no timed instrumentation.

The active count observed in Phase 54 was 31.54–31.57M. This experiment uses
a 32M lower bound and the observed 39,428,667 default. Fixed per-instance
capacity means no repeated sizing or CPU-side total.item() hook during timing.
Do not modify installed packages. Synthetic scene only.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera

def timed(fn):
    t=time.perf_counter(); image=fn(); mx.eval(image)
    return (time.perf_counter()-t)*1000,np.asarray(image)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--rounds',type=int,default=10)
    p.add_argument('--capacities',default='32000000,34000000,36000000,39428667')
    p.add_argument('--out',type=Path,default=Path('phase55_fixed_capacity.json'))
    a=p.parse_args()
    caps=[int(x) for x in a.capacities.split(',')]
    if a.out.exists():p.error('output already exists')
    if a.n<1 or a.n>800000 or min(a.width,a.height,a.rounds)<1 or not caps or min(caps)<1:p.error('invalid inputs')
    if len(set(caps))!=len(caps):p.error('duplicate capacity')
    gs=scene(a.n)
    names=[str(c) for c in caps]
    engines={}
    for name,capacity in zip(names,caps):
        eng=PersistentCppCompatRenderer(gs)
        eng._renderer._capacity=capacity
        engines[name]=eng
    samples={name:[] for name in names}
    rng=np.random.default_rng(55)
    # Three untimed warm-ups; no instrumentation inside the render path.
    for i in range(-3,0):
        c=camera(a.width,a.height,i+3)
        images={}
        for name in names:
            _,img=timed(lambda r=engines[name]:r.render(c))
            images[name]=img
        ref=images[names[-1]]
        if any(not np.array_equal(ref,images[name]) for name in names):
            raise AssertionError('warmup parity failure; possible capacity overflow')
    for i in range(a.rounds):
        c=camera(a.width,a.height,i+3)
        images={}
        for name in rng.permutation(names):
            ms,img=timed(lambda r=engines[name]:r.render(c))
            if engines[name]._renderer._capacity!=int(name):
                raise RuntimeError(f'capacity auto-grew for {name}: initial size was insufficient')
            images[name]=img;samples[name].append(ms)
        ref=images[names[-1]]
        for name in names:
            if not np.array_equal(ref,images[name]):
                raise AssertionError(f'frame {i} capacity {name} differs bitwise')
        print('PASS',i+1,'/'+str(a.rounds),', '.join(
            f'{name}={samples[name][-1]:.2f}ms' for name in names),flush=True)
    med={name:float(np.median(values)) for name,values in samples.items()}
    report={'phase':55,'n':a.n,'resolution':[a.width,a.height],
            'capacities':caps,'rounds':a.rounds,'medians_ms':med,
            'raw_ms':samples,
            'note':'Fixed capacities; any mismatch or auto-growth aborts. '
                   'Derived from Phase 54 counts; do not reuse these sizes for other scenes.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('Medians:',med,'Saved:',a.out)

if __name__=='__main__':main()
