#!/usr/bin/env python3
"""Phase 87: audit actual selective radix dispatch during renderer camera changes.

Opt-in only; production code and defaults unchanged.  Uses Phase85's camera,
parity, and scene setup. Count the actual calls to final3 and fallback radix4,
and check bitwise image+alpha parity against unmodified renderer.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera
from bench_phase57_active_sort import pose
from bench_phase85_final3 import fast_camera,assert_render_equal
import selective_final3_experiment as policy
import radix_final3_experiment as experiment

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--n',type=int,default=700000)
 p.add_argument('--width',type=int,default=1920)
 p.add_argument('--height',type=int,default=1080)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if not 1<=a.n<=800000:p.error('n out of range')
 gs=scene(a.n)
 engines={x:PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded',
           tile_sort_impl='radix4',tile_range_mode='auto') for x in ('baseline','selective')}
 base=experiment.backend()[0]
 original=base._radix_tile_argsort
 chosen=[]
 def traced(keys,tiles):
  slots=int(keys.size);tiles=int(tiles)
  final=policy.select_final3(slots,tiles)
  chosen.append({'slots':slots,'tiles':tiles,'selected':'final3' if final else 'radix4'})
  if final:return experiment.stable_final3_argsort(keys,tiles,variant='indices_only')
  return original(keys,tiles)
 poses=[3,3,0,1,14,15,16,29,30,31,34,37,40,44,3]
 rows=[]
 for i,index in enumerate(poses):
  cam=fast_camera(pose(a.width,a.height,index) if i>=2 else camera(a.width,a.height,index))
  images={};times={}
  for name in ('baseline','selective'):
   begin=time.perf_counter()
   if name=='selective':
    if base._radix_tile_argsort is not original:raise AssertionError('sort hook not clean')
    base._radix_tile_argsort=traced
   try:
    images[name]=engines[name]._renderer.render(cam)
    mx.eval(images[name]['image'],images[name]['alpha'])
   finally:
    base._radix_tile_argsort=original
   times[name]=(time.perf_counter()-begin)*1000
  assert_render_equal(images,('phase87',index,i))
  if not chosen:raise AssertionError('dispatch not invoked')
  info=chosen[-1]
  row={'pose':index,'step':i,'dispatch':dict(info),
       'baseline_ms':times['baseline'],'selective_ms':times['selective'],
       'counts':dict(engines['selective']._renderer.last_frame_counts)}
  rows.append(row)
  print('PASS',i,'pose',index,info['selected'],'slots',info['slots'],
        'baseline',round(times['baseline'],3),'selective',round(times['selective'],3),flush=True)
 if base._radix_tile_argsort is not original:raise AssertionError('monkey patch leaked')
 report={'phase':87,'n':a.n,'resolution':[a.width,a.height],
         'policy':{'min_slots':policy.MIN_SLOTS,'min_tiles':policy.MIN_TILES},
         'rows':rows,'selected_counts':{x:sum(r['dispatch']['selected']==x for r in rows)
                                      for x in ('radix4','final3')},
         'median_baseline_ms':float(np.median([r['baseline_ms'] for r in rows])),
         'median_selective_ms':float(np.median([r['selective_ms'] for r in rows])),
         'image_alpha_bitwise_parity':'PASS',
         'caveats':['Sequential unrandomized comparison; use for dispatch audit, not definitive speedup.',
                    'Guarded capacity may grow after camera jumps.',
                    'Experimental process-global patch is serial-only.',
                    'No production changes.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase87',report['selected_counts'],
       'median',report['median_baseline_ms'],report['median_selective_ms'],flush=True)
if __name__=='__main__':main()
