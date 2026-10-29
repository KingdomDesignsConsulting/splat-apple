#!/usr/bin/env python3
"""Phase101: direct-layout two-output radix scatter threadgroup-size sweep.

One Metal kernel, no packed format or conversion pass. Compare native 256
threadgroup size with 128 and 512 for the same scatter source, using exact
key/index equality and paired timings. This isolates dispatch shape without
modifying any validated renderer source. First round excluded from medians.
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
 p.add_argument('--rounds',type=int,default=10)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists() or a.rounds<4:p.error('unused output required; rounds>=4')
 m=sys.modules[_renderer_class().__module__]
 n=a.capacity;blocks=(n+255)//256;tiles=8160
 x=mx.arange(n,dtype=mx.uint32)
 keys=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
               (x*1664525+1013904223)%tiles)
 indices=mx.arange(n,dtype=mx.int32)
 mx.eval(keys,indices)
 rng=np.random.default_rng(101)
 records=[]
 for shift in (0,4,8):
  params=mx.array([n,blocks,shift],dtype=mx.int32)
  ranks,hist=m._radix_rank_hist_kernel(inputs=[keys,params],
    output_shapes=[(n,),(16,blocks)],output_dtypes=[mx.int32,mx.int32],
    grid=(blocks*256,1,1),threadgroup=(256,1,1))
  prefix=mx.cumsum(hist,axis=1)-hist
  total=mx.sum(hist,axis=1);starts=mx.cumsum(total)-total
  mx.eval(ranks,prefix,starts,params)
  args=[keys,indices,ranks,prefix,starts,params]
  for round_id in range(a.rounds):
   outputs={};timings={}
   for tg in rng.permutation([128,256,512]).tolist():
    start=time.perf_counter()
    outputs[tg]=m._radix_scatter_kernel(inputs=args,
      output_shapes=[(n,),(n,)],output_dtypes=[mx.uint32,mx.int32],
      grid=(n,1,1),threadgroup=(tg,1,1))
    mx.eval(*outputs[tg])
    timings[str(tg)]=(time.perf_counter()-start)*1000
   for tg in (128,512):
    for j in (0,1):
     if not bool(mx.all(outputs[tg][j]==outputs[256][j]).item()):
      raise AssertionError(('scatter parity',shift,round_id,tg,j))
   records.append({'shift':shift,'round':round_id,'ms':timings})
   print('PASS',n,shift,round_id,*[round(timings[str(k)],3) for k in (128,256,512)],flush=True)
 summary=[]
 for shift in (0,4,8):
  group=[r for r in records if r['shift']==shift and r['round']>0]
  med={str(k):float(np.median([r['ms'][str(k)] for r in group])) for k in (128,256,512)}
  summary.append({'shift':shift,'median_ms':med,'versus_256_pct':{
    str(k):100*(med[str(k)]/med['256']-1) for k in (128,512)}})
 result={'phase':101,'capacity':n,'rounds':a.rounds,'summary':summary,'records':records,
         'parity':'exact keys and indices PASS for each group size',
         'peak_mlx_memory_bytes':int(mx.get_peak_memory()),
         'limitations':['Microbenchmark, not end-to-end rendering.',
                        'All launches use native scatter kernel and direct two-output layout.',
                        'Grid semantics and group-size support must be checked on the Mac.',
                        'No production code or defaults modified.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase101 PASS',json.dumps(summary),flush=True)
if __name__=='__main__':main()
