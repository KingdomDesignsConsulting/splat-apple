#!/usr/bin/env python3
"""Phase 81: locate next renderer algorithm bottleneck after prewarm experiments.

Compare auto and guarded at matched capacities on a fixed camera, preserving
bitwise equality. Report uninstrumented per-frame wall timings and separately
instrumented synchronized Python-stage timings (not hardware GPU timestamps).
Do not sum stage numbers to predict frame time.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera,memory
from bench_phase57_active_sort import breakdown

def sample(e,cam):
 start=time.perf_counter()
 image=e.render(cam);mx.eval(image)
 return image,(time.perf_counter()-start)*1000

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--n',type=int,default=700000)
 p.add_argument('--width',type=int,default=1920)
 p.add_argument('--height',type=int,default=1080)
 p.add_argument('--rounds',type=int,default=16)
 p.add_argument('--profile-rounds',type=int,default=4)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if not 1<=a.n<=800000 or min(a.width,a.height,a.rounds,a.profile_rounds)<1:p.error('invalid args')
 gs=scene(a.n);cam=camera(a.width,a.height,3)
 names=('auto','guarded')
 engines={name:PersistentCppCompatRenderer(gs,duplicate_sort_mode=name,
    tile_sort_impl='auto',tile_range_mode='auto') for name in names}
 for name in names:
  for _ in range(3):sample(engines[name],cam)
 cap=max(int(engines[name]._renderer.last_frame_counts['active']) for name in names)
 cap=cap*105//100+1024
 for name in names:engines[name]._renderer._capacity=cap
 rng=np.random.default_rng(81)
 samples={name:[] for name in names}
 for i in range(a.rounds):
  images={}
  for name in rng.permutation(names):
   images[name],ms=sample(engines[name],cam)
   samples[name].append(ms)
  if not np.array_equal(np.asarray(images['auto']),np.asarray(images['guarded'])):
   raise AssertionError('image parity failure on round '+str(i))
  print('PASS steady',i,' '.join(f'{name}={samples[name][-1]:.3f}ms' for name in names),flush=True)
 profiles={}
 for name in names:
  e=engines[name]
  with breakdown(e) as stages:
   for _ in range(a.profile_rounds):
    image=e.render(cam);mx.eval(image)
  profiles[name]={key:dict(median_ms=float(np.median(values)),samples_ms=[float(x) for x in values])
                  for key,values in stages.items() if values}
 result=dict(phase=81,n=a.n,resolution=[a.width,a.height],
  matched_capacity=int(cap),rounds=a.rounds,
  normal={name:dict(median_ms=float(np.median(samples[name])),
                   p95_ms=float(np.percentile(samples[name],95)),
                   samples_ms=samples[name]) for name in names},
  synchronized_stage_profile=profiles,memory_after=memory(),
  notes=['Synthetic scene, same camera and capacity, bitwise frame parity.',
         'Stage timings include Python dispatch, pending dependency evaluation and GPU sync.',
         'Not Metal per-kernel timestamps; never sum to obtain normal frame time.',
         'No source or installed package modifications.'])
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT normal', {k:v['median_ms'] for k,v in result['normal'].items()},
       'stages', {m:{k:round(v['median_ms'],3) for k,v in d.items()} for m,d in profiles.items()},flush=True)
if __name__=='__main__':main()
