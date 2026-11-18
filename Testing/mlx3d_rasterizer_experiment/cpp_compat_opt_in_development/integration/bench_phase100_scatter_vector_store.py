#!/usr/bin/env python3
"""Phase 100: fused intermediate two-output scatter, vector-layout experiment.

Compare native scatter to a candidate that treats the two 32-bit output
streams as one ulong2-like write via packed ulong payloads. This is a
microbenchmark only; the packed output is INTERLEAVED and needs a conversion
for subsequent radix passes. We measure that conversion in the candidate's
total and compare recovered keys and indices bitwise.
"""
import argparse,json,sys,time
from pathlib import Path
import mlx.core as mx
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class

CAPS=(33154187,42477817,53283577)
PACK=mx.fast.metal_kernel(
 name="phase100_scatter_interleaved64",
 input_names=["keys","indices","ranks","prefix","starts","params"],
 output_names=["packed"],
 source=r"""
  uint i=thread_position_in_grid.x;
  if(i >= (uint)params[0])return;
  uint blocks=(uint)params[1],block=i>>8;
  uint digit=(keys[i]>>params[2])&15u;
  uint dst=(uint)(starts[digit]+prefix[digit*blocks+block]+ranks[i]);
  packed[dst]=((ulong)(uint)indices[i]<<32)|(ulong)keys[i];
 """)
UNPACK=mx.fast.metal_kernel(
 name="phase100_unpack_interleaved64",
 input_names=["packed","params"],
 output_names=["out_keys","out_indices"],
 source=r"""
  uint i=thread_position_in_grid.x;
  if(i >= (uint)params[0])return;
  ulong pair=packed[i];
  out_keys[i]=(uint)pair;
  out_indices[i]=(int)(pair>>32);
 """)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--capacity',type=int,choices=CAPS,required=True)
 p.add_argument('--rounds',type=int,default=8)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists() or a.rounds<4:p.error('output must not exist, rounds >=4')
 m=sys.modules[_renderer_class().__module__]
 n=a.capacity;blocks=(n+255)//256;tiles=8160
 x=mx.arange(n,dtype=mx.uint32)
 keys=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
               (x*1664525+1013904223)%tiles)
 ids=mx.arange(n,dtype=mx.int32);mx.eval(keys,ids)
 rng=np.random.default_rng(100)
 rows=[]
 for shift in (0,4,8):
  params=mx.array([n,blocks,shift],dtype=mx.int32)
  ranks,hist=m._radix_rank_hist_kernel(inputs=[keys,params],
   output_shapes=[(n,),(16,blocks)],output_dtypes=[mx.int32,mx.int32],
   grid=(blocks*256,1,1),threadgroup=(256,1,1))
  prefix=mx.cumsum(hist,axis=1)-hist
  totals=mx.sum(hist,axis=1);starts=mx.cumsum(totals)-totals
  mx.eval(ranks,prefix,starts,params)
  inp=[keys,ids,ranks,prefix,starts,params]
  for rep in range(a.rounds):
   times={};out={}
   for name in rng.permutation(['native','packed_unpack']).tolist():
    start=time.perf_counter()
    if name=='native':
     result=m._radix_scatter_kernel(inputs=inp,output_shapes=[(n,),(n,)],
      output_dtypes=[mx.uint32,mx.int32],grid=(n,1,1),threadgroup=(256,1,1))
     mx.eval(*result)
    else:
     packed=PACK(inputs=inp,output_shapes=[(n,)],
      output_dtypes=[mx.uint64],grid=(n,1,1),threadgroup=(256,1,1))[0]
     result=UNPACK(inputs=[packed,params],output_shapes=[(n,),(n,)],
      output_dtypes=[mx.uint32,mx.int32],grid=(n,1,1),threadgroup=(256,1,1))
     mx.eval(*result)
    times[name]=(time.perf_counter()-start)*1000
    out[name]=result
   for j in (0,1):
    if not bool(mx.all(out['native'][j]==out['packed_unpack'][j]).item()):
     raise AssertionError(('parity',shift,rep,j))
   rows.append({'shift':shift,'repeat':rep,'times_ms':times})
   print('PASS',n,shift,rep,round(times['native'],3),
         round(times['packed_unpack'],3),flush=True)
 summary=[]
 for shift in (0,4,8):
  group=[r for r in rows if r['shift']==shift and r['repeat']>0]
  median={k:float(np.median([r['times_ms'][k] for r in group]))
          for k in ('native','packed_unpack')}
  summary.append({'shift':shift,'median_ms':median,
   'relative_change_percent':100*(median['packed_unpack']/median['native']-1)})
 report={'phase':100,'capacity':n,'summary':summary,'rows':rows,
  'parity':'exact recovered keys and indices PASS',
  'peak_memory_bytes':int(mx.get_peak_memory()),
  'notes':['Packed layout is not directly usable by existing rank kernel.',
           'Unpack counted inside candidate timing; no omitted format-conversion cost.',
           'Extra packed buffer required; reject if net slower or memory worse.',
           'Isolated intermediate pass only; no production changes.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase100',json.dumps(summary),flush=True)
if __name__=='__main__':main()
