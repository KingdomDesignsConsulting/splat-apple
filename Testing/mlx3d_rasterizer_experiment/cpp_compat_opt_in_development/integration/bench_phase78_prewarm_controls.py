#!/usr/bin/env python3
"""Phase 78: compare no prep, unrelated GPU work, and nearby/exact radix prep.

Run each --mode in a separate Python process, preferably across repeated
rounds with rotated mode order. Includes preparation in totals. Fixed target
47,363,197 keys, five measured repeats. Exact stable order checked on 4096
independent keys. No production modifications.
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
 p.add_argument('--mode',choices=['cold','generic','near','exact'],required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--target',type=int,default=47363197)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if not 10000<=a.target<2**31:p.error('invalid target')
 cls=_renderer_class();radix=sys.modules[cls.__module__]._radix_tile_argsort
 tiles=8192
 def keys(n):
  x=mx.arange(n,dtype=mx.uint32)
  return (x*1664525+1013904223)%tiles
 def run(n):
  t=time.perf_counter();order=radix(keys(n),tiles);mx.eval(order)
  if tuple(order.shape)!=(n,):raise AssertionError('size')
  return (time.perf_counter()-t)*1000
 for _ in range(3):run(33154238)
 prep=0.
 if a.mode=='generic':
  t=time.perf_counter()
  x=mx.arange(a.target,dtype=mx.float32);mx.eval(mx.sum(x*x))
  prep=(time.perf_counter()-t)*1000
 elif a.mode=='near':prep=run(a.target+636803)
 elif a.mode=='exact':prep=run(a.target)
 rows=[run(a.target) for _ in range(5)]
 rng=np.random.default_rng(78);x=rng.integers(0,tiles,4096,dtype=np.uint32)
 y=radix(mx.array(x),tiles);mx.eval(y)
 np.testing.assert_array_equal(np.asarray(y),np.argsort(x,kind='stable').astype(np.int32))
 result={'phase':78,'mode':a.mode,'target':a.target,
         'prep_ms':prep,'first_ms':rows[0],
         'repeat_median_ms':float(np.median(rows[1:])),
         'total_including_prep_ms':float(prep+sum(rows)),
         'samples_ms':rows,
         'memory':{k:int(getattr(mx,k)()) for k in ('get_active_memory','get_peak_memory','get_cache_memory')},
         'notes':['One fresh process per mode; repeat across mode-order rotations.',
                  'generic runs a non-radix large GPU operation control.',
                  'near prepares a different radix shape; exact prepares identical radix shape.',
                  'All preparation time included in totals; not renderer timing.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT',a.mode,'prep_ms',round(prep,3),'first_ms',round(rows[0],3),
       'total_ms',round(result['total_including_prep_ms'],3),flush=True)
if __name__=='__main__':main()
