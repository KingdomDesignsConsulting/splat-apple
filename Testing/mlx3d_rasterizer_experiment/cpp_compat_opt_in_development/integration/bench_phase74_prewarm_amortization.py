#!/usr/bin/env python3
"""Phase 74: include the GPU cost of prewarming over short/long use horizons.

Launch each mode as a separate process, preferably alternate run order across
replications. Fixed scene/camera: initial small capacity, then one transition
to a larger sort shape used for --frames frames. 'prewarm' invokes the large
shape before the measured sequence, returning to small first; its warmup
time is INCLUDED in end-to-end total. Compares against within-process image
oracle bitwise. Report active and cached MLX memory snapshots.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera,memory

def draw(renderer,cam,capacity,oracle):
    renderer._renderer._capacity=int(capacity)
    t=time.perf_counter()
    image=renderer.render(cam);mx.eval(image)
    ms=(time.perf_counter()-t)*1000
    if not np.array_equal(np.asarray(image),oracle):raise AssertionError('bitwise parity')
    if int(renderer._renderer.last_frame_counts['capacity'])!=int(capacity):
        raise AssertionError('unexpected buffer growth')
    return ms

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['cold','prewarm'],required=True)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height,a.frames)<1:p.error('invalid parameters')
    gs=scene(a.n);cam=camera(a.width,a.height,3)
    engine=PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded')
    first=engine.render(cam);mx.eval(first)
    oracle=np.asarray(first).copy()
    active=int(engine._renderer.last_frame_counts['active'])
    small=max(active*105//100+1024,active+1)
    large=max(active*150//100+1024,small+1)
    for _ in range(4):draw(engine,cam,small,oracle)
    before=memory()
    warm_ms=0.
    if a.mode=='prewarm':
        warm_ms=draw(engine,cam,large,oracle)
        draw(engine,cam,small,oracle)
    after_prewarm=memory()
    seq=[draw(engine,cam,large,oracle) for _ in range(a.frames)]
    report=dict(phase=74,mode=a.mode,n=a.n,resolution=[a.width,a.height],
       frames=a.frames,active=active,small_capacity=small,large_capacity=large,
       prewarm_ms=warm_ms,first_transition_ms=seq[0],
       sequence_total_ms=float(sum(seq)),
       total_including_prewarm_ms=float(warm_ms+sum(seq)),
       steady_median_ms=float(np.median(seq[1:])) if len(seq)>1 else None,
       frame_times_ms=seq,memory_before=before,memory_after_prewarm=after_prewarm,
       memory_after_sequence=memory(),
       notes=['Independent Python processes required for clean comparison.',
              'Totals include only large-shape prewarm plus measurement frames; initial fixed setup excluded.',
              'Memory metrics are process-wide MLX allocator counters, not OS peak RSS.',
              'Prewarm before transition reduces latency spike but costs GPU time and memory.',
              'One fixed camera; does not establish benefit for arbitrary camera motion.'])
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT',a.mode,'prewarm_ms',round(warm_ms,3),
          'first_ms',round(seq[0],3),'total_including_prewarm_ms',
          round(report['total_including_prewarm_ms'],3),
          'memory_after_prewarm',after_prewarm,'saved',a.out,flush=True)
if __name__=='__main__':main()
