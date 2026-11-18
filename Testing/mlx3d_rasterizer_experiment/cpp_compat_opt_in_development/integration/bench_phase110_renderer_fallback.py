#!/usr/bin/env python3
"""Phase110: actual renderer fallback-dispatch acceptance for scatter512.

Run small-capacity FullHD and small-tile 640x360 scenes, checking actual
radix4 sorter calls and 256-thread scatters. Instrumentation uses scoped
serial-only wrappers restored in finally; the opt-in production modes are
unchanged. Exact RGB+alpha parity with existing selective_final3 mode.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer,_renderer_class
from bench_phase49_breakdown import scene,camera
from bench_phase85_final3 import fast_camera,assert_render_equal

def case(gaussians,width,height,splats,label):
 cls=_renderer_class()
 module=sys.modules[cls.__module__]
 base=PersistentCppCompatRenderer(gaussians,duplicate_sort_mode='guarded',
     tile_range_mode='auto',tile_sort_impl='selective_final3')
 test=PersistentCppCompatRenderer(gaussians,duplicate_sort_mode='guarded',
     tile_range_mode='auto',tile_sort_impl='selective_final3_scatter512')
 cam=fast_camera(camera(width,height,3))
 original_radix=module._radix_tile_argsort
 original_final=module._selective_final3_argsort
 original_scatter=module._radix_scatter_kernel
 record={'label':label,'splats':splats,'width':width,'height':height,
         'calls':{'radix4':0,'final3':0},'scatter_groups':[]}
 def radix(*args,**kwargs):
  record['calls']['radix4']+=1
  return original_radix(*args,**kwargs)
 def final3(*args,**kwargs):
  record['calls']['final3']+=1
  return original_final(*args,**kwargs)
 def scatter(*args,**kwargs):
  record['scatter_groups'].append(kwargs.get('threadgroup'))
  return original_scatter(*args,**kwargs)
 try:
  module._radix_tile_argsort=radix
  module._selective_final3_argsort=final3
  module._radix_scatter_kernel=scatter
  frame=test._renderer.render(cam)
  mx.eval(frame['image'],frame['alpha'])
 finally:
  module._radix_tile_argsort=original_radix
  module._selective_final3_argsort=original_final
  module._radix_scatter_kernel=original_scatter
 ref=base._renderer.render(cam)
 mx.eval(ref['image'],ref['alpha'])
 assert_render_equal({'radix4':ref,'final3':frame},('phase110',label))
 if record['calls']['radix4']!=1 or record['calls']['final3']!=0:
  raise AssertionError(('unexpected fallback dispatch',record))
 if not record['scatter_groups'] or any(tuple(g)!=(256,1,1) for g in record['scatter_groups']):
  raise AssertionError(('unexpected scatter group',record))
 record['sort_slots']=test._renderer.last_frame_counts.get('sorted')
 record['reference_counts']=base._renderer.last_frame_counts
 if record['reference_counts']!=test._renderer.last_frame_counts:
  raise AssertionError(('counts differ',record))
 if record['sort_slots']>=8_000_000 and ((width+15)//16)*((height+15)//16)>=8000:
  raise AssertionError(('case did not exercise fallback eligibility',record))
 record['parity']='PASS exact RGB and alpha'
 print('PASS fallback renderer',label,'slots',record['sort_slots'],
       'radix_calls',record['calls']['radix4'],'scatter',record['scatter_groups'],flush=True)
 return record

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error(f'output already exists: {a.out}')
 tests=[
  ('small_capacity_fullhd',4096,1920,1080),
  ('small_tile_domain',700000,640,360),
 ]
 result=[]
 for label,n,w,h in tests:
  result.append(case(scene(n),w,h,n,label))
 report={'phase':110,'cases':result,'parity':'PASS','dispatch':'PASS radix4 fallback and 256-thread scatter',
  'caveats':['Instrumented scoped Python hooks, restored before reference rendering.',
             'No production code/default changes.',
             'Two renderer fallback cases; does not exhaust all tile geometries.']}
 a.out.write_text(json.dumps(report,indent=2,default=str)+'\n')
 print('RESULT phase110 PASS',json.dumps({'cases':[x['label'] for x in result]}),flush=True)
if __name__=='__main__':main()
