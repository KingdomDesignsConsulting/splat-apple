#!/usr/bin/env python3
"""Phase 82: measure each stable radix digit pass separately.

Use production _radix_tile_argsort with temporary in-process wrappers around
rank/histogram and scatter. Compare uninstrumented sort timing against pass
diagnostics at the Full-HD tile count. No source, kernel, or defaults changed.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--n',type=int,default=33154238)
 p.add_argument('--tiles',type=int,default=8160)
 p.add_argument('--rounds',type=int,default=6)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if a.n<1 or not 1<=a.tiles<65536 or a.rounds<2:p.error('bad inputs')
 cls=_renderer_class();mod=sys.modules[cls.__module__]
 # Tile and sentinel key distribution approximate masked renderer slots.
 x=mx.arange(a.n,dtype=mx.uint32)
 keys=mx.where((x%7)==0,mx.array(0xffffffff,dtype=mx.uint32),
               (x*1664525+1013904223)%a.tiles)
 def run():
  t=time.perf_counter();r=mod._radix_tile_argsort(keys,a.tiles);mx.eval(r)
  return r,(time.perf_counter()-t)*1000
 for _ in range(3):run()
 normal=[run()[1] for _ in range(a.rounds)]
 original_rank=mod._radix_rank_hist_kernel
 original_scatter=mod._radix_scatter_kernel
 stages=[]
 def wrap(original,name):
  def call(*args,**kwargs):
   t=time.perf_counter()
   value=original(*args,**kwargs)
   mx.eval(value)
   stages.append({'operation':name,'shift':int(np.asarray(kwargs['inputs'][-1])[2]),
                  'ms':(time.perf_counter()-t)*1000})
   return value
  return call
 try:
  mod._radix_rank_hist_kernel=wrap(original_rank,'rank_hist')
  mod._radix_scatter_kernel=wrap(original_scatter,'scatter')
  _out,profiled=run()
 finally:
  mod._radix_rank_hist_kernel=original_rank
  mod._radix_scatter_kernel=original_scatter
 # Independent exact stable oracle on manageable input.
 raw=np.random.default_rng(82).integers(0,a.tiles,4096,dtype=np.uint32)
 raw[::17]=np.uint32(0xffffffff)
 observed=mod._radix_tile_argsort(mx.array(raw),a.tiles);mx.eval(observed)
 np.testing.assert_array_equal(np.asarray(observed),np.argsort(raw,kind='stable').astype(np.int32))
 result={'phase':82,'slots':a.n,'tiles':a.tiles,'rounds':a.rounds,
         'normal_sort_median_ms':float(np.median(normal)),
         'normal_sort_samples_ms':normal,'profiled_total_ms':profiled,
         'stage_samples':stages,
         'memory':{k:int(getattr(mx,k)()) for k in
                   ('get_active_memory','get_peak_memory','get_cache_memory')},
         'notes':['The mask includes sentinels as in capacity-sorted rendering.',
                  'Profile forces evaluation between stages; not GPU kernel timestamps.',
                  'Profiled stages cannot be summed to estimate normal latency.',
                  'Exact stable-sort oracle checked on 4096 independent keys.',
                  'No production code changes.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT normal_median',result['normal_sort_median_ms'],
       'profiled',profiled,'stages',stages,flush=True)
if __name__=='__main__':main()
