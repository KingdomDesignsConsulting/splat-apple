#!/usr/bin/env python3
"""Phase 98: assess reduced-write scatter variants without production mutation.

Compare native radix scatter (keys and indices) against indices-only scatter
when only the permutation is needed. IMPORTANT: indices-only cannot replace
intermediate LSD passes, which must carry keys forward; it is legal only for
the last pass. This microbenchmark quantifies isolated write savings and
does not establish a complete-sort improvement.
"""
import argparse,json,sys,time
from pathlib import Path
import mlx.core as mx
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
CAPS=(33154187,42477817,53283577)
INDICES_ONLY=mx.fast.metal_kernel(
 name="phase98_scatter_indices_only",
 input_names=["keys","indices","ranks","prefix","starts","params"],
 output_names=["out_indices"],
 source=r"""
    uint i=thread_position_in_grid.x;
    if (i >= (uint)params[0]) return;
    uint block=i>>8, blocks=(uint)params[1];
    uint digit=(keys[i]>>params[2])&15u;
    uint dst=(uint)(starts[digit]+prefix[digit*blocks+block]+ranks[i]);
    out_indices[dst]=indices[i];
 """,
)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--capacity',type=int,choices=CAPS,required=True)
 p.add_argument('--rounds',type=int,default=12)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.rounds<5 or a.out.exists():p.error('rounds>=5 and unused out required')
 n=a.capacity;tiles=8160;blocks=(n+255)//256
 m=sys.modules[_renderer_class().__module__]
 x=mx.arange(n,dtype=mx.uint32)
 keys=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
              (x*1664525+1013904223)%tiles)
 idx=mx.arange(n,dtype=mx.int32);mx.eval(keys,idx)
 # Last full 4-bit pass, with sentinels still legal.
 params=mx.array([n,blocks,12],dtype=mx.int32)
 ranks,hist=m._radix_rank_hist_kernel(
  inputs=[keys,params],output_shapes=[(n,),(16,blocks)],
  output_dtypes=[mx.int32,mx.int32],
  grid=(blocks*256,1,1),threadgroup=(256,1,1))
 prefix=mx.cumsum(hist,axis=1)-hist
 totals=mx.sum(hist,axis=1);starts=mx.cumsum(totals)-totals
 mx.eval(ranks,prefix,starts,params)
 inputs=[keys,idx,ranks,prefix,starts,params]
 random=np.random.default_rng(98)
 records=[]
 for r in range(a.rounds):
  results={};time_ms={}
  for mode in random.permutation(('native','indices_only')).tolist():
   t=time.perf_counter()
   if mode=='native':
    result=m._radix_scatter_kernel(inputs=inputs,output_shapes=[(n,),(n,)],
      output_dtypes=[mx.uint32,mx.int32],grid=(n,1,1),threadgroup=(256,1,1))
    mx.eval(*result);results[mode]=result[1]
   else:
    result=INDICES_ONLY(inputs=inputs,output_shapes=[(n,)],
      output_dtypes=[mx.int32],grid=(n,1,1),threadgroup=(256,1,1))
    mx.eval(*result);results[mode]=result[0]
   time_ms[mode]=(time.perf_counter()-t)*1000
  # Exact full-array parity verified outside both timed regions.
  if not bool(mx.all(results['native']==results['indices_only']).item()):
   raise AssertionError(('permutation mismatch',r))
  records.append({'round':r,'times_ms':time_ms})
  print('PASS',n,r,round(time_ms['native'],3),
        round(time_ms['indices_only'],3),flush=True)
 steady=records[1:]
 med={mode:float(np.median([r['times_ms'][mode] for r in steady]))
      for mode in ('native','indices_only')}
 result={'phase':98,'capacity':n,'rounds':a.rounds,'median_ms':med,
         'relative_change_percent':100*(med['indices_only']/med['native']-1),
         'saved_final_key_output_bytes':4*n,
         'exact_permutation_parity':'PASS','records':records,
         'peak_memory_bytes':int(mx.get_peak_memory()),
         'limitations':['Final LSD pass only; intermediate passes must scatter keys.',
                        'Final3 already uses an indices-only output for its specialized final pass.',
                        'Isolated scatter comparison, not a proposed production replacement.',
                        'Synchronized software timing, no GPU hardware timestamps.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase98 PASS',n,med,result['relative_change_percent'],flush=True)
if __name__=='__main__':main()
