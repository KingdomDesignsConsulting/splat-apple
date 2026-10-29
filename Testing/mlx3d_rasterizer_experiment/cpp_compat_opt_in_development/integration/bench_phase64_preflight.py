#!/usr/bin/env python3
"""Phase 64: compare preflight sizing to adaptive rerender on abrupt camera paths.

Three pre-existing policy modes: auto adaptive; active exact sizing; bucketed
(1M quantization). All force radix4 and automatic binary ranges, compare
bitwise each frame, and record p50/p95/p99 and frame-specific counts.
No production code or site-packages changes.
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

def measure(engine,cam):
    r=engine._renderer;before=int(r._capacity)
    start=time.perf_counter()
    image=engine.render(cam);mx.eval(image)
    ms=(time.perf_counter()-start)*1000
    counts=dict(r.last_frame_counts)
    return image,{'ms':ms,'before':before,'after':int(r._capacity),
                  'active':int(counts['active']),'sorted':int(counts['sorted']),
                  'used_capacity':int(counts['capacity']),
                  'overflow_expected':bool(before and int(counts['active'])>before)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase64_preflight.json'))
    a=p.parse_args()
    if a.out.exists():p.error('Output file already exists')
    if not 1<=a.n<=800000 or min(a.width,a.height)<1:p.error('Invalid parameters')
    gs=scene(a.n)
    modes={'auto':'auto','active':'active','bucketed':'bucketed'}
    engines={k:PersistentCppCompatRenderer(gs,duplicate_sort_mode=mode,
        tile_range_mode='auto',tile_sort_impl='radix4') for k,mode in modes.items()}
    for e in engines.values():measure(e,camera(a.width,a.height,0))
    poses=[0,1,14,15,16,17,18,19,20,29,30,31,32,33,34,35,44,0,1,44,0,1]
    rng=np.random.default_rng(64)
    rows=[]
    for step,index in enumerate(poses):
        cam=pose(a.width,a.height,index)
        images={};results={}
        for name in rng.permutation(list(modes)):
            images[name],results[name]=measure(engines[name],cam)
        baseline=np.asarray(images['auto'])
        for name in modes:
            if not np.array_equal(baseline,np.asarray(images[name])):
                raise AssertionError(f'bitwise image mismatch step {step} pose {index} mode {name}')
        rows.append({'step':step,'pose':index,**results})
        print(f"PASS {step+1}/{len(poses)} pose={index} "+' '.join(
            f"{name}={results[name]['ms']:.2f}ms" for name in modes),flush=True)
    stats={}
    for name in modes:
        times=np.array([x[name]['ms'] for x in rows])
        stats[name]={k:float(v) for k,v in {
            'p50':np.percentile(times,50),'p95':np.percentile(times,95),
            'p99':np.percentile(times,99),'maximum':times.max(),
            'total':times.sum()}.items()}
    result={'phase':64,'n':a.n,'resolution':[a.width,a.height],
            'modes':modes,'stats_ms':stats,'samples':rows,
            'process_memory_after':memory(),
            'notes':['Identical geometry, radix4 and auto binary ranges; only sizing policy differs.',
                     'Active/bucketed perform scalar readback before expansion.',
                     'Bitwise image parity enforced on every step.',
                     'Single-process shared allocator history is not per-mode memory usage.']}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',stats,'Saved:',a.out,flush=True)

if __name__=='__main__':main()
