#!/usr/bin/env python3
"""Phase 54: inspect capacity utilization via identical warm-camera sweeps.

Compares standard persistent capacity, exact active-duplicate-sized capacity,
and a 10% headroom policy. Active duplicate counts are read by wrapping the
existing geometry kernel, without modifying installed packages or kernels.
Parity is checked on every frame; under-capacity cases are disallowed.
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
    t=time.perf_counter();image=fn();mx.eval(image)
    return (time.perf_counter()-t)*1000,np.asarray(image)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",type=int,default=700000)
    p.add_argument("--width",type=int,default=1920)
    p.add_argument("--height",type=int,default=1080)
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--out",type=Path,default=Path("phase54_capacity_utilization.json"))
    a=p.parse_args()
    if a.out.exists():p.error("Output exists")
    if a.n<1 or a.n>800000 or min(a.width,a.height,a.rounds)<1:p.error("invalid input")
    gs=scene(a.n)
    standard=PersistentCppCompatRenderer(gs)
    exact=PersistentCppCompatRenderer(gs)
    margin=PersistentCppCompatRenderer(gs)
    modes={'default':standard,'exact':exact,'headroom_10pct':margin}
    measurements=[]
    # Derive active count from standard renderer's per-camera geometry invocation.
    # Hooking _composite observes the scalar total after geometry and before use.
    original=standard._renderer._composite
    observed={}
    def capture(*args,**kwargs):
        total=args[8]  # composite signature: xy,co,rgb,order,offsets,counts,bbox,background,total,...
        observed['active']=int(total.item())
        return original(*args,**kwargs)
    standard._renderer._composite=capture
    # Use same ordered camera sequence, and avoid underallocation by running
    # standard first to observe each frame's active duplicates.
    for i in range(a.rounds+2):
        c=camera(a.width,a.height,i)
        observed.clear()
        t0,baseline=timed(lambda:standard.render(c))
        if 'active' not in observed:raise RuntimeError("active-duplicate observation failed")
        active=observed['active']
        # Expansion requires a nonzero capacity; reserve at least one slot.
        exact._renderer._capacity=max(active,1)
        margin._renderer._capacity=max(int(np.ceil(active*1.10)),1)
        values={'default':baseline}
        timings={'default':t0}
        for mode,renderer in (('exact',exact),('headroom_10pct',margin)):
            ms,image=timed(lambda r=renderer:r.render(c))
            timings[mode]=ms;values[mode]=image
            if not np.array_equal(image,baseline):
                raise AssertionError(f"Mismatch frame={i} mode={mode}")
        if i>=2:
            row={'camera':i,'active_duplicates':active,
                 'default_capacity':standard._renderer._capacity,
                 'exact_capacity':exact._renderer._capacity,
                 'headroom_capacity':margin._renderer._capacity,
                 'timings_ms':timings}
            measurements.append(row)
            print(f"PASS {i-1}/{a.rounds}: active={active} default={row['default_capacity']} "
                  f"utilization={active/row['default_capacity']:.3f} "
                  f"default={t0:.2f} exact={timings['exact']:.2f} "
                  f"headroom={timings['headroom_10pct']:.2f} ms",flush=True)
    med={mode:float(np.median([r['timings_ms'][mode] for r in measurements]))
         for mode in modes}
    report={'phase':54,'n':a.n,'resolution':[a.width,a.height],
            'rounds':a.rounds,'medians_ms':med,'samples':measurements,
            'notes':['Diagnostic uses private _composite instrumentation to count duplicates.',
                     'total.item() causes additional synchronization in default measured path; timing is not a fair speed comparison.',
                     'Exact/margin mode override capacity on every frame; intended to characterize size, not propose an unsafe production policy.']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print("Saved:",a.out,"Medians (instrumented; not comparable):",med)

if __name__=='__main__':main()
