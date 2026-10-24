#!/usr/bin/env python3
"""Phase 62: isolate adaptive-capacity latency spikes at camera transitions.

Compare default adaptive/radix to fixed high-water capacity/radix on the same
700K synthetic scene. Both use identical kernels; check bitwise output.
The high-water renderer is initially sized by a full pose survey; its steadier
capacity costs sorting extra slots. Reports capacity before/after, active
duplicates, overflow/growth signals, per-pose latency, p50/p95/p99, and memory.
No source/site-packages changes or frame-time instrumentation.
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

def measured(renderer,cam):
    before=renderer._renderer._capacity
    start=time.perf_counter()
    img=renderer.render(cam); mx.eval(img)
    elapsed=(time.perf_counter()-start)*1000
    details=dict(renderer._renderer.last_frame_counts)
    after=renderer._renderer._capacity
    return img,{'ms':elapsed,'before':before,'after':after,
                'active':details.get('active'),'sorted':details.get('sorted'),
                'used_capacity':details.get('capacity'),
                'growth_expected':details.get('active',0)>before and before>0,
                'capacity_changed':after!=before}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase62_growth_spikes.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height)<1:p.error('invalid scene')
    gs=scene(a.n)
    dynamic=PersistentCppCompatRenderer(gs)
    highwater=PersistentCppCompatRenderer(gs)
    # Prime both; intentionally survey high-water needs without interfering
    # with dynamic state. Compare dynamically measured active counts and
    # reject if any high-water capacity proves insufficient.
    required=0
    for i in range(45):
        _img,_row=measured(highwater,pose(a.width,a.height,i))
        required=max(required,int(_row['active']))
    # Fixed preallocation safely exceeds max active count in this survey.
    reserve=required*105//100+1024
    highwater._renderer._capacity=reserve
    measured(dynamic,camera(a.width,a.height,0))
    rows=[]
    # Repeat known stressful jumps to expose grow/shrink cost and hot steady
    # frames. Keep frames identical across paths and alternate call order.
    sequence=[0,1,14,15,16,17,18,19,20,29,30,31,32,33,34,35,44,0,1,44,0,1]
    for k,i in enumerate(sequence):
        cam=pose(a.width,a.height,i)
        names=('dynamic','highwater') if k%2==0 else ('highwater','dynamic')
        result={}
        images={}
        for name in names:
            eng=dynamic if name=='dynamic' else highwater
            images[name],result[name]=measured(eng,cam)
        if not np.array_equal(np.asarray(images['dynamic']),np.asarray(images['highwater'])):
            raise AssertionError(f'frame parity failure at sequence step {k}, pose {i}')
        if highwater._renderer._capacity!=reserve:
            raise RuntimeError('highwater unexpectedly grew; survey bound violated')
        rows.append({'step':k,'pose':i,'dynamic':result['dynamic'],
                     'highwater':result['highwater']})
        print(f"PASS step={k} pose={i} dynamic={result['dynamic']['ms']:.2f}ms "
              f"highwater={result['highwater']['ms']:.2f}ms "
              f"active={result['dynamic']['active']} "
              f"cap={result['dynamic']['before']}->{result['dynamic']['after']}",flush=True)
    def stats(mode):
        values=[r[mode]['ms'] for r in rows]
        return {'p50':float(np.percentile(values,50)),
                'p95':float(np.percentile(values,95)),
                'p99':float(np.percentile(values,99)),
                'max':float(max(values)),
                'total':float(sum(values))}
    report={'phase':62,'n':a.n,'resolution':[a.width,a.height],
            'highwater_capacity':reserve,'highwater_required_max':required,
            'sequence':sequence,'stats_ms':{m:stats(m) for m in ('dynamic','highwater')},
            'samples':rows,'process_memory':memory(),
            'notes':['All frames bitwise equal across the two optimized policies.',
                     'Fixed highwater policy is a diagnostic, not an arbitrary-scene safe default.',
                     'Growth_expected is inferred from active count and pre-frame capacity, not a kernel-level GPU timestamp.',
                     'GPU kernel attribution requires Metal trace.']}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT',report['stats_ms'],'saved',a.out,flush=True)

if __name__=='__main__':main()
