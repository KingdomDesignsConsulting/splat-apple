#!/usr/bin/env python3
"""Phase 94: rank/scatter stage audit at 33M, 42M and 53M sort capacities.

Instrument native radix4 and final3 rank/scatter calls without modifying
production source. Input keys are evaluated first. Forced intermediate GPU
evaluation changes scheduling, so stage times are diagnostics, NOT hardware
kernel timestamps. Compare first exposure and revisit in fresh mode processes.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
CAPS=(33154187,42477817,53283577)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=('radix4','final3'),required=True)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 module=sys.modules[_renderer_class().__module__]
 tiles=8160
 keys=[]
 for n in CAPS:
  x=mx.arange(n,dtype=mx.uint32)
  k=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
             (x*1664525+1013904223)%tiles)
  mx.eval(k);keys.append(k)
 old={}
 stages=[]
 for attr,label in [('_radix_rank_hist_kernel','rank_hist'),
                    ('_radix_scatter_kernel','scatter'),
                    ('_FINAL_RANK','final_rank'),
                    ('_FINAL_INDICES','final_scatter')]:
  old[attr]=getattr(module,attr)
  def wrapper(*args,_original=old[attr],_label=label,**kwargs):
   t=time.perf_counter()
   value=_original(*args,**kwargs);mx.eval(value)
   stages.append({'stage':_label,'ms':(time.perf_counter()-t)*1000})
   return value
  setattr(module,attr,wrapper)
 rows=[]
 try:
  for step,index in enumerate((0,0,1,1,2,2,1,0,2)):
   stages.clear()
   start=time.perf_counter()
   order=(module._radix_tile_argsort(keys[index],tiles) if a.mode=='radix4'
          else module._selective_final3_argsort(keys[index],tiles))
   mx.eval(order)
   elapsed=(time.perf_counter()-start)*1000
   row={'step':step,'capacity':CAPS[index],
        'profiled_total_ms':elapsed,'stages':list(stages)}
   rows.append(row)
   print('PASS',a.mode,step,CAPS[index],round(elapsed,3),
         [(x['stage'],round(x['ms'],2)) for x in stages],flush=True)
 finally:
  for name,value in old.items():setattr(module,name,value)
 result={'phase':94,'mode':a.mode,'rows':rows,
         'peak_memory_bytes':int(mx.get_peak_memory()),
         'cache_memory_bytes':int(mx.get_cache_memory()),
         'notes':['All keys evaluated before timed sorting.',
                  'Stage instrumentation changes dependencies and synchronization.',
                  'Do not sum instrumented stages to infer uninstrumented latency.',
                  'Execute modes in separate fresh processes.',
                  'No production changes.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase94',a.mode,
       [(row['capacity'],round(row['profiled_total_ms'],2)) for row in rows],flush=True)
if __name__=='__main__':main()
