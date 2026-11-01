#!/usr/bin/env python3
"""Phase 96: measure duplicate-key radix scatter traffic at fixed capacities.

Benchmarks production's individual rank/histogram and scatter operations for
four LSD digits on evaluated deterministic key buffers. Each timed operation
forces completion; these are *software-synchronized stage times*, not pure GPU
timestamps. No source modifications or automatic prewarming.
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
 p.add_argument('--capacity',type=int,choices=CAPS,required=True)
 p.add_argument('--repeats',type=int,default=6)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output already exists')
 if a.repeats<3:p.error('repeats >=3')
 module=sys.modules[_renderer_class().__module__]
 n=a.capacity;blocks=(n+255)//256;tiles=8160
 x=mx.arange(n,dtype=mx.uint32)
 keys=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
               (x*1664525+1013904223)%tiles)
 indices=mx.arange(n,dtype=mx.int32)
 mx.eval(keys,indices)
 rows=[]
 for shift in (0,4,8,12):
  params=mx.array([n,blocks,shift],dtype=mx.int32)
  # Repeated runs; first counted separately as potentially shape cold.
  for attempt in range(a.repeats):
   t=time.perf_counter()
   ranks,hist=module._radix_rank_hist_kernel(
     inputs=[keys,params],output_shapes=[(n,),(16,blocks)],
     output_dtypes=[mx.int32,mx.int32],
     grid=(blocks*256,1,1),threadgroup=(256,1,1))
   mx.eval(ranks,hist)
   rank_ms=(time.perf_counter()-t)*1000
   t=time.perf_counter()
   prefix=mx.cumsum(hist,axis=1)-hist
   totals=mx.sum(hist,axis=1)
   starts=mx.cumsum(totals)-totals
   mx.eval(prefix,starts)
   prefix_ms=(time.perf_counter()-t)*1000
   t=time.perf_counter()
   out_keys,out_indices=module._radix_scatter_kernel(
     inputs=[keys,indices,ranks,prefix,starts,params],
     output_shapes=[(n,),(n,)],output_dtypes=[mx.uint32,mx.int32],
     grid=(n,1,1),threadgroup=(256,1,1))
   mx.eval(out_keys,out_indices)
   scatter_ms=(time.perf_counter()-t)*1000
   rows.append({'digit_shift':shift,'attempt':attempt,'rank_ms':rank_ms,
                'prefix_ms':prefix_ms,'scatter_ms':scatter_ms})
   print('PASS',n,shift,attempt,
         round(rank_ms,3),round(prefix_ms,3),round(scatter_ms,3),flush=True)
 summary=[]
 for shift in (0,4,8,12):
  group=[x for x in rows if x['digit_shift']==shift]
  summary.append({'shift':shift,
     'first_scatter_ms':group[0]['scatter_ms'],
     'repeat_scatter_median_ms':float(np.median([x['scatter_ms'] for x in group[1:]])),
     'repeat_rank_median_ms':float(np.median([x['rank_ms'] for x in group[1:]])),
     'repeat_prefix_median_ms':float(np.median([x['prefix_ms'] for x in group[1:]]))})
 result={'phase':96,'capacity':n,'repeats':a.repeats,'summaries':summary,'rows':rows,
         'peak_memory_bytes':int(mx.get_peak_memory()),
         'notes':['Single digits are benchmarked independently on the same pre-evaluated input.',
                  'Scatter outputs are materialized for fair completion accounting.',
                  'Cannot derive end-to-end full stable-sort timing by summing these stages.',
                  'No algorithm mutation, no prewarming, no production default changes.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase96',json.dumps(summary),flush=True)
if __name__=='__main__':main()
