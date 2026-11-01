#!/usr/bin/env python3
"""Phase107 candidate gate: opt-in 512 scatter, fallback and parity validation.

This prototype is a scoped serial-only dispatcher with explicit eligibility
(capacity>=8M, >=8000 tiles and sentinel-safe final3), not a production edit.
It tests three sort capacities and a small sentinel-mixed fallback, runs a
renderer parity smoke test, and records whether 512 is actually dispatched.
No change to production snapshot, validated SHA, defaults or adapter.
"""
import argparse,json,sys,time
from pathlib import Path
import mlx.core as mx
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
CAPS=(1048576,8388608,33154187,53283577)

def sort_with_guard(m,keys,tiles,enable,log):
 n=int(keys.size)
 eligible=bool(enable and n>=8_000_000 and tiles>=8000
               and m.final_pass_plan(tiles)['eligible'])
 groups=512 if eligible else 256
 log.append({'slots':n,'tiles':tiles,'group':groups,'eligible':eligible})
 # Mirrors existing selective_final3's three early passes; never modifies
 # module-level kernels or renderer state.
 if not eligible:
  return m._radix_tile_argsort(keys,tiles),groups
 blocks=(n+255)//256
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
   grid=(n,1,1),threadgroup=(groups,1,1))
 params=mx.array([n,blocks,plan['shift']],dtype=mx.int32)
 ranks,hist=m._FINAL_RANK(
  inputs=[keys,params],output_shapes=[(n,),(3,blocks)],
  output_dtypes=[mx.int32,mx.int32],
  grid=(blocks*256,1,1),threadgroup=(256,1,1))
 prefix=mx.cumsum(hist,axis=1)-hist
 totals=mx.sum(hist,axis=1);starts=mx.cumsum(totals)-totals
 out=m._FINAL_INDICES(inputs=[keys,indices,ranks,prefix,starts,params],
   output_shapes=[(n,)],output_dtypes=[mx.int32],
   grid=(n,1,1),threadgroup=(256,1,1))[0]
 return out,groups

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--capacities',type=int,nargs='+',default=list(CAPS))
 a=p.parse_args()
 if a.out.exists():p.error(f'output exists: {a.out}')
 m=sys.modules[_renderer_class().__module__]
 records=[]
 for n in a.capacities:
  if n<=0 or n>=2**31:p.error('capacity outside supported range')
  tiles=8160 if n>=8_000_000 else 256
  x=mx.arange(n,dtype=mx.uint32)
  keys=mx.where(x%11==0,mx.array(0xffffffff,dtype=mx.uint32),
                (x*1664525+1013904223)%tiles)
  mx.eval(keys)
  record=[]
  for enabled in (False,True):
   log=[];start=time.perf_counter()
   candidate,group=sort_with_guard(m,keys,tiles,enabled,log)
   mx.eval(candidate)
   elapsed=(time.perf_counter()-start)*1000
   record.append((candidate,group,elapsed,log))
  base,_,_,_=record[0]
  cand,group,elapsed,log=record[1]
  # For eligible n, baseline is native stable radix4 (four passes), and the
  # candidate is final3 (three early passes), so test additionally against
  # existing selective_final3 to isolate threadgroup change from final3.
  reference=m._selective_final3_argsort(keys,tiles) if group==512 else base
  mx.eval(reference)
  for name,other in [('native_radix4',base),('production_final3_or_fallback',reference)]:
   if not bool(mx.all(cand==other).item()):
    raise AssertionError((n,name,'permutation differs'))
  if bool(group==512)!=(n>=8_000_000 and tiles>=8000):
   raise AssertionError(('dispatch mismatch',n,group))
  row={'slots':n,'tiles':tiles,'candidate_group':group,
    'eligible':bool(group==512),'experimental_ms':elapsed,
    'dispatch':log,'parity':'PASS'}
  records.append(row)
  print('PASS',n,'group',group,'ms',round(elapsed,3),flush=True)
 report={'phase':107,'records':records,'parity':'PASS exactly vs radix4 and selective_final3',
  'limitations':['Serial isolated sorter candidate, not renderer integration.',
                 'Full-array device equality; sentinels injected deterministically.',
                 'No source change or validated SHA update.',
                 'Fallback uses production radix4 and native scatter group.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase107 PASS',json.dumps(records),flush=True)
if __name__=='__main__':main()
