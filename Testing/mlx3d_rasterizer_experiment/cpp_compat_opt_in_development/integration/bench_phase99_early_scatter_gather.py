#!/usr/bin/env python3
"""Phase99: early-pass indices-only scatter followed by key gather.

Tests whether separating indices scatter and key gather beats the native
two-output scatter, for first LSD pass with identity input indices only.
Not a production-ready algorithm for later LSD passes. No source mutations.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
CAPS=(33154187,42477817,53283577)
INDICES_ONLY=mx.fast.metal_kernel(
 name="phase99_early_scatter_indices_only",
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
 p.add_argument('--rounds',type=int,default=8)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists() or a.rounds<4:p.error('unused out and rounds >=4 required')
 m=sys.modules[_renderer_class().__module__]
 n=a.capacity;tiles=8160;blocks=(n+255)//256
 x=mx.arange(n,dtype=mx.uint32)
 keys=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
               (x*1664525+1013904223)%tiles)
 ids=mx.arange(n,dtype=mx.int32);mx.eval(keys,ids)
 rng=np.random.default_rng(99)
 records=[]
 for shift in (0,4,8):
  params=mx.array([n,blocks,shift],dtype=mx.int32)
  ranks,hist=m._radix_rank_hist_kernel(
   inputs=[keys,params],output_shapes=[(n,),(16,blocks)],
   output_dtypes=[mx.int32,mx.int32],grid=(blocks*256,1,1),threadgroup=(256,1,1))
  prefix=mx.cumsum(hist,axis=1)-hist
  totals=mx.sum(hist,axis=1);starts=mx.cumsum(totals)-totals
  mx.eval(ranks,prefix,starts)
  inp=[keys,ids,ranks,prefix,starts,params]
  for rep in range(a.rounds):
   times={};outputs={}
   for mode in rng.permutation(['native','index_gather']).tolist():
    start=time.perf_counter()
    if mode=='native':
     out=m._radix_scatter_kernel(
      inputs=inp,output_shapes=[(n,),(n,)],
      output_dtypes=[mx.uint32,mx.int32],grid=(n,1,1),threadgroup=(256,1,1))
     mx.eval(*out)
    else:
     perm=INDICES_ONLY(inputs=inp,output_shapes=[(n,)],
      output_dtypes=[mx.int32],grid=(n,1,1),threadgroup=(256,1,1))[0]
     gathered=mx.take(keys,perm,axis=0)
     mx.eval(perm,gathered)
     out=(gathered,perm)
    times[mode]=(time.perf_counter()-start)*1000
    outputs[mode]=out
   for j in (0,1):
    if not bool(mx.all(outputs['native'][j]==outputs['index_gather'][j]).item()):
     raise AssertionError((shift,rep,j))
   records.append({'shift':shift,'repeat':rep,'ms':times})
   print('PASS',n,shift,rep,round(times['native'],3),
         round(times['index_gather'],3),flush=True)
 summary=[]
 for shift in (0,4,8):
  rows=[r for r in records if r['shift']==shift and r['repeat']>0]
  med={k:float(np.median([r['ms'][k] for r in rows]))
       for k in ('native','index_gather')}
  summary.append({'shift':shift,'medians_ms':med,
      'relative_change_percent':100*(med['index_gather']/med['native']-1)})
 report={'phase':99,'capacity':n,'summary':summary,'records':records,
   'parity':'exact keys and indices PASS',
   'peak_memory_bytes':int(mx.get_peak_memory()),
   'limitations':['First-pass identity indices only; subsequent passes require care.',
                  'Index-only scatter plus gather has more launches and read traffic.',
                  'No renderer integration or production changes.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase99',json.dumps(summary),flush=True)
if __name__=='__main__':main()
