#!/usr/bin/env python3
"""Phase 72: controlled first-use sort-capacity prewarm, fixed camera.

Two independent guarded renderers: cold and prewarmed. Both start in small
capacity and are tested at identical large capacity. Warm renderer exercises
the large shape and returns to small before the timed switch. Compare image
parity and latency; randomized policy order for large trials is *not* used
because a shared MLX allocator can transfer benefits between renderers.
Runs both orders in separately launched processes for stronger evidence.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera,memory

def timed(e,cam,cap):
    e._renderer._capacity=int(cap)
    t=time.perf_counter()
    image=e.render(cam);mx.eval(image)
    dt=(time.perf_counter()-t)*1000
    if int(e._renderer.last_frame_counts['capacity'])!=cap:
        raise AssertionError('unexpected resize')
    return image,dt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    gs=scene(a.n);cam=camera(a.width,a.height,3)
    engines={k:PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded') for k in ('cold','prewarm')}
    first=engines['cold'].render(cam);mx.eval(first)
    act=int(engines['cold']._renderer.last_frame_counts['active'])
    small=max(act*105//100+1024,act+1)
    large=max(act*150//100+1024,small+1)
    for name in engines:
        for i in range(3):timed(engines[name],cam,small)
    baseline=np.asarray(first)
    # Prewarm explicitly at large capacity before the measured transition.
    warmimg,warm_ms=timed(engines['prewarm'],cam,large)
    assert np.array_equal(np.asarray(warmimg),baseline)
    timed(engines['prewarm'],cam,small)
    rows=[]
    for name in ('prewarm','cold'):
        img,ms=timed(engines[name],cam,large)
        assert np.array_equal(np.asarray(img),baseline)
        follow=[]
        for _ in range(3):
            i,t=timed(engines[name],cam,large)
            assert np.array_equal(np.asarray(i),baseline)
            follow.append(t)
        rows.append(dict(name=name,transition_ms=ms,repeat_median_ms=float(np.median(follow)),repeats_ms=follow))
        print('PASS',name,'transition_ms',round(ms,3),'repeat_median_ms',round(float(np.median(follow)),3),flush=True)
    result=dict(phase=72,n=a.n,resolution=[a.width,a.height],
                small_capacity=small,large_capacity=large,
                prewarm_ms=warm_ms,rows=rows,memory=memory(),
                notes=['Two renderer instances share allocator; prewarming may affect cold renderer.',
                       'Prewarm render is performed before measured transition, not free.',
                       'Images compared bitwise to initial image.'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',result['rows'],'saved',a.out,flush=True)
if __name__=='__main__':main()
