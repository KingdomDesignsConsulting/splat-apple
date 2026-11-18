#!/usr/bin/env python3
"""Two-renderer paired final validation against the frozen pre-change source."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from mlx_gs.renderer import mlx3d_cpp_compat as adapter
from bench_phase49_breakdown import scene,camera,memory
from bench_phase57_active_sort import pose,breakdown


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--survey',type=int,default=45)
    p.add_argument('--candidate-sort',choices=['capacity','adaptive','active','bucketed','auto'],default='adaptive')
    p.add_argument('--candidate-algorithm', choices=['mlx','radix4','auto'], default='mlx')
    p.add_argument('--candidate-ranges',choices=['scan','binary','auto'],default='auto')
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1 <= a.n <= 800000 or min(a.width,a.height,a.rounds,a.survey)<1:p.error('invalid inputs')
    source=a.reference.read_bytes()
    reference_sha=hashlib.sha256(source).hexdigest()
    assert reference_sha=='b6aa0444116920d1ae5e526eaf5821e4970bf681c74ce04096fb7f57ae9c11d1'
    spec=importlib.util.spec_from_file_location('mlx3d.splatting._phase59_reference',a.reference)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    gs=scene(a.n)
    reference=adapter.PersistentCppCompatRenderer(gs,duplicate_sort_mode='capacity',tile_range_mode='scan')
    old=reference._renderer
    reference._renderer=module.FastGaussianRenderer(means=old.means,quats=gs.quaternions,
        scales=mx.exp(gs.scales),opacities=old.opacities,colors=old._colors_static,
        sh_degree=0,geometry_policy='cpp_compat',alpha_mode='exponential',t_min=0.0001)
    start=time.perf_counter()
    candidate=adapter.PersistentCppCompatRenderer(gs,duplicate_sort_mode=a.candidate_sort,tile_range_mode=a.candidate_ranges,tile_sort_impl=a.candidate_algorithm)
    setup_ms=(time.perf_counter()-start)*1000
    engines={'reference':reference,'candidate':candidate}
    rng=np.random.default_rng(5959)
    rows=[]
    def paired(cam,label):
        times,images={},{}
        for mode in rng.permutation(list(engines)):
            t=time.perf_counter()
            images[mode]=engines[mode].render(cam)
            mx.eval(images[mode])
            times[mode]=(time.perf_counter()-t)*1000
        if not np.array_equal(np.asarray(images['reference']),np.asarray(images['candidate'])):
            raise AssertionError(('parity',label))
        row={'frame':label,'ms':times,'reference_capacity':reference._renderer._capacity,
             'candidate_counts':dict(candidate._renderer.last_frame_counts)}
        print('PASS',label,times,flush=True)
        return row
    for i in range(6):paired(camera(a.width,a.height,i),'warmup'+str(i))
    for i in range(a.rounds):rows.append(paired(camera(a.width,a.height,i+6),i))
    survey=[]
    reference._renderer._capacity=1
    candidate._renderer._capacity=1
    for i in range(a.survey):survey.append(paired(pose(a.width,a.height,i),'pose'+str(i)))
    # Revisit wide view after close zoom; checks shrink then abrupt regrowth.
    for i in (0,1,44,0,1):survey.append(paired(pose(a.width,a.height,i),'revisit'+str(i)))
    # Profile on a common camera after both engines have settled there.
    cam=camera(a.width,a.height,3)
    for _ in range(3):paired(cam,'profilewarm')
    phases={}
    for mode,engine in engines.items():
        with breakdown(engine) as samples:
            for _ in range(3):engine.render(cam)
        phases[mode]={k:float(np.median(v)) for k,v in samples.items()}
    def summary(data):
        return {'median_ms':{m:float(np.median([r['ms'][m] for r in data])) for m in engines},
                'paired_ratio':float(np.median([r['ms']['candidate']/r['ms']['reference'] for r in data]))}
    report={'phase':59,'n':a.n,'resolution':[a.width,a.height],'rounds':a.rounds,
            'candidate_sort':a.candidate_sort,'candidate_ranges':a.candidate_ranges,'candidate_algorithm':a.candidate_algorithm,
            'reference_sha256':reference_sha,'candidate_sha256':adapter._VALIDATED_SOURCE_SHA,
            'candidate_constructor_cached_module_ms':setup_ms,'steady':summary(rows),
            'varied_camera':summary(survey),'samples':rows,'survey':survey,
            'synchronized_breakdown_ms':phases,'process_memory_after':memory(),
            'notes':['Bitwise parity against exact pre-change validated source for every frame.',
                     'Timings include render and GPU completion; NumPy readback/parity is outside timers.',
                     'Synchronized breakdown includes dispatch; not pure GPU timestamps.',
                     'Memory is process-wide and includes both renderers and allocator cache.',
                     'Synthetic scenes; no new equivalence claim against original C++ renderer.']}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT steady',report['steady'],'varied',report['varied_camera'],flush=True)


if __name__=='__main__':main()
