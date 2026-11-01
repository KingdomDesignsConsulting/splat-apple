#!/usr/bin/env python3
"""Phase102: full radix sort test with 512-thread scatter launches.

Experimental sorter duplicates production's stable LSD control flow but
changes *only* intermediate scatter launch size 256 -> 512. Final3 final
indices-only scatter remains unchanged. Exact permutation parity, paired
randomized timing, three large capacities; no production source mutations.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
CAPS=(33154187,42477817,53283577)

def candidate(m,keys,tiles,group):
 n=int(keys.size);blocks=(n+255)//256
 indices=mx.arange(n,dtype=mx.int32)
 plan=m.final_pass_plan(tiles)
 for shift in range(0,plan['shift'],4):
  params=mx.array([n,blocks,shift],dtype=mx.int32)
  ranks,hist=m._radix_rank_hist_kernel(
   inputs=[keys,params],output_shapes=[(n,),(16,blocks)],
   output_dtypes=[mx.int32,mx.int32],
   grid=(blocks*256,1,1),threadgroup=(256,1,1))
  prefix=mx.cumsum(hist,axis=1)-hist
  totals=mx.sum(hist,axis=1);starts=mx.cumsum(totals)-totals
  keys,indices=m._radix_scatter_kernel(
   inputs=[keys,indices,ranks,prefix,starts,params],
   output_shapes=[(n,),(n,)],output_dtypes=[mx.uint32,mx.int32],
   grid=(n,1,1),threadgroup=(group,1,1))
 params=mx.array([n,blocks,plan['shift']],dtype=mx.int32)
 ranks,hist=m._FINAL_RANK(
  inputs=[keys,params],output_shapes=[(n,),(3,blocks)],
  output_dtypes=[mx.int32,mx.int32],
  grid=(blocks*256,1,1),threadgroup=(256,1,1))
 prefix=mx.cumsum(hist,axis=1)-hist
 totals=mx.sum(hist,axis=1);starts=mx.cumsum(totals)-totals
 return m._FINAL_INDICES(
  inputs=[keys,indices,ranks,prefix,starts,params],
  output_shapes=[(n,)],output_dtypes=[mx.int32],
  grid=(n,1,1),threadgroup=(256,1,1))[0]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--capacity',type=int,choices=CAPS,required=True)
 p.add_argument('--rounds',type=int,default=8)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.rounds < 4:
  p.error('--rounds must be at least 4')
 if a.out.exists():
  p.error(f'output already exists: {a.out}; choose a new --out filename')
 m=sys.modules[_renderer_class().__module__];tiles=8160;n=a.capacity
 x=mx.arange(n,dtype=mx.uint32)
 keys=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
               (x*1664525+1013904223)%tiles)
 mx.eval(keys)
 random=np.random.default_rng(102)
 rows=[]
 for rep in range(a.rounds):
  times={};perms={}
  for group in random.permutation([256,512]).tolist():
   t=time.perf_counter()
   perms[group]=candidate(m,keys,tiles,int(group))
   mx.eval(perms[group])
   times[str(group)]=(time.perf_counter()-t)*1000
  if not bool(mx.all(perms[256]==perms[512]).item()):
   raise AssertionError(('full stable permutation mismatch',rep))
  rows.append({'round':rep,'ms':times})
  print('PASS',n,rep,round(times['256'],3),round(times['512'],3),flush=True)
 med={key:float(np.median([r['ms'][key] for r in rows[1:]])) for key in ('256','512')}
 result={'phase':102,'capacity':n,'rounds':a.rounds,'steady_median_ms':med,
  'change_percent':100*(med['512']/med['256']-1),
  'parity':'exact full stable permutation PASS','rows':rows,
  'peak_memory_bytes':int(mx.get_peak_memory()),
  'limitations':['Both paths run full 3 preceding passes plus the same specialized final3.',
                 'No renderer integration, no production code/default change.',
                 'GPU completion inside timing; bitwise parity outside timing.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase102 PASS',n,med,result['change_percent'],flush=True)
if __name__=='__main__':main()
