#!/usr/bin/env python3
"""Phase 73: separate-process cold vs prewarmed shape transition.

Run this file twice as distinct Python processes with --mode cold / prewarm.
Cold never evaluates large-capacity sort before the measured transition;
prewarm evaluates it, returns to small, then measures. No concurrent renderer
instances or allocator sharing between modes. Check bitwise output parity
within each run. Do not compare output time against a different thermal/load
state as a controlled GPU timestamp.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera,memory

def timed(engine,cam,capacity):
    engine._renderer._capacity=int(capacity)
    start=time.perf_counter()
    img=engine.render(cam);mx.eval(img)
    dt=(time.perf_counter()-start)*1000
    if int(engine._renderer.last_frame_counts['capacity']) != capacity:
        raise AssertionError('unexpected capacity growth')
    return img,dt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('cold','prewarm'),required=True)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height)<1:p.error('invalid scene')
    gaussians=scene(a.n);cam=camera(a.width,a.height,3)
    engine=PersistentCppCompatRenderer(gaussians,duplicate_sort_mode='guarded')
    image=engine.render(cam);mx.eval(image)
    oracle=np.asarray(image).copy()
    active=int(engine._renderer.last_frame_counts['active'])
    small=max(active*105//100+1024,active+1)
    large=max(active*150//100+1024,small+1)
    for _ in range(4):
        img,_=timed(engine,cam,small)
        if not np.array_equal(np.asarray(img),oracle):raise AssertionError('small parity')
    prewarm_ms=None
    if a.mode=='prewarm':
        img,prewarm_ms=timed(engine,cam,large)
        if not np.array_equal(np.asarray(img),oracle):raise AssertionError('prewarm parity')
        img,_=timed(engine,cam,small)
        if not np.array_equal(np.asarray(img),oracle):raise AssertionError('return parity')
    rows=[]
    for i in range(5):
        img,ms=timed(engine,cam,large)
        if not np.array_equal(np.asarray(img),oracle):raise AssertionError('large parity')
        rows.append(ms)
        print(f"PASS {a.mode} large{i}={ms:.3f}ms",flush=True)
    report=dict(phase=73,mode=a.mode,n=a.n,resolution=[a.width,a.height],
                small_capacity=small,large_capacity=large,active=active,
                prewarm_ms=prewarm_ms,transition_ms=rows[0],
                repeated_median_ms=float(np.median(rows[1:])),
                samples_ms=rows,memory=memory(),
                notes=['Run opposite modes as separate Python processes.',
                       'MLX allocator cache not shared across processes.',
                       'System shader caches and device contention may still persist.',
                       'Prewarm costs GPU time outside measured transition.',
                       'Every frame checked against fixed-camera image oracle.'])
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT',a.mode,report['transition_ms'],report['repeated_median_ms'],
          'prewarm',prewarm_ms,flush=True)
if __name__=='__main__':main()
