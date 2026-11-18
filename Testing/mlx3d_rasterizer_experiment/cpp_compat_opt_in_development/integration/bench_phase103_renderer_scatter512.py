#!/usr/bin/env python3
"""Phase 103: renderer-level A/B for 512-thread intermediate radix scatter.

Scoped temporary experiment patch of module-level scatter dispatcher. This
benchmark is serial-only; it restores the original kernel on every frame.
Compare native instance-local selective_final3 at 256 vs 512 thread scatter,
random paired order, guarded capacity, bitwise RGB and alpha parity.
Production source and defaults remain unmodified.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer,_renderer_class
from bench_phase49_breakdown import scene,camera
from bench_phase57_active_sort import pose
from bench_phase85_final3 import fast_camera,assert_render_equal

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--rounds',type=int,default=2)
 p.add_argument('--seed',type=int,default=103)
 p.add_argument('--growth-order',choices=['random','256-first','512-first'],default='random')
 a=p.parse_args()
 if a.out.exists():p.error(f'output exists: {a.out}')
 if a.rounds<2:p.error('--rounds must be >=2')
 module=sys.modules[_renderer_class().__module__]
 original=module._radix_scatter_kernel
 rng=np.random.default_rng(a.seed)
 gs=scene(700000)
 engines={name:PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded',
  tile_sort_impl='selective_final3',tile_range_mode='auto') for name in ('256','512')}
 positions=[3,3,0,1,14,15,16,29,30,31,34,37,40,44,3]
 rows=[]
 def patched(*args,**kwargs):
  kwargs['threadgroup']=(512,1,1)
  return original(*args,**kwargs)
 try:
  for round_id in range(a.rounds):
   for step,k in enumerate(positions):
    cam=fast_camera(camera(1920,1080,3) if step<2 else pose(1920,1080,k))
    frames={};times={};capacity={}
    order=rng.permutation(['256','512']).tolist()
    if round_id==0 and step in (9,10) and a.growth_order!='random':
     order=['256','512'] if a.growth_order=='256-first' else ['512','256']
    for mode in order:
     module._radix_scatter_kernel=patched if mode=='512' else original
     t=time.perf_counter()
     try:
      frame=engines[mode]._renderer.render(cam)
      mx.eval(frame['image'],frame['alpha'])
      times[mode]=(time.perf_counter()-t)*1000
      frames[mode]=frame
      capacity[mode]=dict(engines[mode]._renderer.last_frame_counts)
     finally:
      module._radix_scatter_kernel=original
    assert_render_equal({'radix4':frames['256'],'final3':frames['512']},
                        (round_id,step,k))
    if capacity['256']!=capacity['512']:
     raise AssertionError(('different workload',round_id,step,capacity))
    row={'round':round_id,'step':step,'pose':k,'ms':times,
         'counts':capacity['256']}
    rows.append(row)
    print('PASS',round_id,step,k,'slots',capacity['256'].get('sorted'),
          'group256',round(times['256'],3),
          'group512',round(times['512'],3),flush=True)
 finally:
  module._radix_scatter_kernel=original
 selected=[r for r in rows if r['step']>=2]
 stats={mode:{'median_ms':float(np.median([r['ms'][mode] for r in selected])),
     'p95_ms':float(np.percentile([r['ms'][mode] for r in selected],95)),
     'total_ms':sum(r['ms'][mode] for r in selected),
     'max_ms':max(r['ms'][mode] for r in selected)}
        for mode in ('256','512')}
 data={'phase':103,'rounds':a.rounds,'stats_excluding_first_two_frames_each_round':stats,
       'rows':rows,'parity':'exact RGB and alpha PASS',
       'seed':a.seed,'growth_order':a.growth_order,
       'peak_shared_process_mlx_bytes':int(mx.get_peak_memory()),
       'caveats':['Experimental process-global hook is serial-only; no concurrent use.',
                  'Both renderers use native selective_final3; only scatter group differs.',
                  'The final specialized indices-only scatter still uses its original group.',
                  'No production source/default mutation.']}
 a.out.write_text(json.dumps(data,indent=2)+'\n')
 print('RESULT phase103 PASS',json.dumps(stats),flush=True)
if __name__=='__main__':main()
