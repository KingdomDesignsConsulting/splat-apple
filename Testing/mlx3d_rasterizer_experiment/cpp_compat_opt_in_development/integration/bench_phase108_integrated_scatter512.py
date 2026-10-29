#!/usr/bin/env python3
"""Phase108: instance-local integrated opt-in scatter512 acceptance.

Exercises validated source-hash loading and two independent renderer instances
without the temporary process-global benchmark hook. Checks bitwise frame
parity across capacity transitions plus warmed camera poses. No default changes.
"""
import argparse,json,time
from pathlib import Path

# Direct script execution sets sys.path[0] to this integration directory.
# Add the repository root before importing the local mlx_gs package.
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer,_renderer_class
from bench_phase49_breakdown import scene,camera
from bench_phase57_active_sort import pose
from bench_phase85_final3 import fast_camera,assert_render_equal

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--rounds',type=int,default=3)
 a=p.parse_args()
 if a.out.exists():p.error(f'output already exists: {a.out}')
 if a.rounds<2:p.error('--rounds must be at least 2')
 cls=_renderer_class()  # enforce validated repository-local source guard
 for invalid in ('selective_final3_scatter1024',''):
  try:
   cls(tile_sort_impl=invalid)
  except ValueError:
   pass
  else:raise AssertionError(('accepted unsupported sort mode',invalid))
 gs=scene(700000)
 engines={
  '256':PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded',
    tile_sort_impl='selective_final3',tile_range_mode='auto'),
  '512':PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded',
    tile_sort_impl='selective_final3_scatter512',tile_range_mode='auto')}
 assert engines['256']._renderer.tile_sort_impl=='selective_final3'
 assert engines['512']._renderer.tile_sort_impl=='selective_final3_scatter512'
 positions=[3,3,0,1,14,15,16,29,30,31,34,37,40,44,3]
 rows=[]
 for round_id in range(a.rounds):
  for step,k in enumerate(positions):
   cam=fast_camera(camera(1920,1080,3) if step<2 else pose(1920,1080,k))
   times={};frames={};counts={}
   for name in (('256','512') if (round_id+step)%2==0 else ('512','256')):
    t=time.perf_counter()
    frame=engines[name]._renderer.render(cam)
    mx.eval(frame['image'],frame['alpha'])
    times[name]=(time.perf_counter()-t)*1000
    frames[name]=frame
    counts[name]=dict(engines[name]._renderer.last_frame_counts)
   assert_render_equal({'radix4':frames['256'],'final3':frames['512']},
                      ('phase108',round_id,step,k))
   if counts['256']!=counts['512']:
    raise AssertionError(('count mismatch',round_id,step,counts))
   rows.append({'round':round_id,'step':step,'pose':k,'ms':times,
                'counts':counts['256']})
   print('PASS',round_id,step,k,'slots',counts['256'].get('sorted'),
         '256',round(times['256'],3),'512',round(times['512'],3),flush=True)
 warm=[r for r in rows if r['round']>=1 and r['step']>=2]
 if not warm:raise AssertionError('no warmed frames')
 med={name:float(np.median([r['ms'][name] for r in warm])) for name in ('256','512')}
 report={'phase':108,'parity':'PASS exact RGB and alpha',
   'source_guard':'PASS _renderer_class instantiated',
   'instances':'two independent local modes, no global hooks',
   'rounds':a.rounds,'warmed_median_ms':med,
   'warmed_relative_percent':100*(med['512']/med['256']-1),
   'warmed_win_rate_512':float(np.mean([r['ms']['512']<r['ms']['256'] for r in warm])),
   'rows':rows,'peak_mlx_memory_bytes':int(mx.get_peak_memory()),
   'notes':['New mode is explicit opt-in and previous defaults unchanged.',
            'When tile capacity <8M, tile count <8000 or sentinel categories ineligible, existing radix4 fallback is used.',
            'This script verifies real render path, not just isolated sorter.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase108 PASS',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
