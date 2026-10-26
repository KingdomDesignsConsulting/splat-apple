#!/usr/bin/env python3
"""Phase 97: isolated radix scatter write-order candidate (experimental only).

Compare the frozen native scatter to a kernel writing permutation first, keys
second. No source monkeypatch, no production API or default changes. Rank/
prefix inputs are evaluated before timing and both output arrays are fully
evaluated. Exact parity of outputs is asserted for every trial. Paired order
randomized and cold trials excluded from steady medians.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class

CAPS=(33154187,42477817,53283577)
ALT=mx.fast.metal_kernel(
 name="phase97_scatter_indices_first",
 input_names=["keys","indices","ranks","prefix","starts","params"],
 output_names=["out_keys","out_indices"],
 source=r"""
   uint i = thread_position_in_grid.x;
   if (i >= (uint)params[0]) return;
   uint block = i >> 8;
   uint blocks = (uint)params[1];
   uint digit = (keys[i] >> params[2]) & 15u;
   uint dst = (uint)(starts[digit] + prefix[digit * blocks + block] + ranks[i]);
   out_indices[dst] = indices[i];
   out_keys[dst] = keys[i];
 """,
)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--capacity',type=int,choices=CAPS,required=True)
 p.add_argument('--repeats',type=int,default=10)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.repeats<4:p.error('repeats >= 4')
 if a.out.exists():p.error('output already exists')
 n=a.capacity;blocks=(n+255)//256;tiles=8160
 module=sys.modules[_renderer_class().__module__]
 idx=mx.arange(n,dtype=mx.uint32)
 keys=mx.where(idx%7==0,mx.array(0xffffffff,dtype=mx.uint32),
               (idx*1664525+1013904223)%tiles)
 ids=mx.arange(n,dtype=mx.int32)
 mx.eval(keys,ids)
 rng=np.random.default_rng(97)
 rows=[]
 for shift in (0,4,8,12):
  params=mx.array([n,blocks,shift],dtype=mx.int32)
  ranks,hist=module._radix_rank_hist_kernel(
   inputs=[keys,params],output_shapes=[(n,),(16,blocks)],
   output_dtypes=[mx.int32,mx.int32],
   grid=(blocks*256,1,1),threadgroup=(256,1,1))
  prefix=mx.cumsum(hist,axis=1)-hist
  totals=mx.sum(hist,axis=1)
  starts=mx.cumsum(totals)-totals
  mx.eval(params,ranks,prefix,starts)
  args=[keys,ids,ranks,prefix,starts,params]
  for repeat in range(a.repeats):
   times={};outputs={}
   for mode in rng.permutation(['native','indices_first']).tolist():
    fn=module._radix_scatter_kernel if mode=='native' else ALT
    start=time.perf_counter()
    outs=fn(inputs=args,output_shapes=[(n,),(n,)],
      output_dtypes=[mx.uint32,mx.int32],
      grid=(n,1,1),threadgroup=(256,1,1))
    mx.eval(*outs)
    times[mode]=(time.perf_counter()-start)*1000
    outputs[mode]=outs
   for i,label in enumerate(('keys','indices')):
    left=np.asarray(outputs['native'][i])
    right=np.asarray(outputs['indices_first'][i])
    if not np.array_equal(left,right):
     raise AssertionError((shift,repeat,label,
      np.flatnonzero(left!=right)[:5].tolist()))
   rows.append({'shift':shift,'repeat':repeat,'times_ms':times})
   print('PASS',n,shift,repeat,
       'native',round(times['native'],3),
       'indices_first',round(times['indices_first'],3),flush=True)
 summary=[]
 for shift in (0,4,8,12):
  group=[row for row in rows if row['shift']==shift and row['repeat']>0]
  med={mode:float(np.median([row['times_ms'][mode] for row in group]))
       for mode in ('native','indices_first')}
  summary.append({'shift':shift,'median_ms':med,
                  'relative_change_percent':(med['indices_first']/med['native']-1)*100})
 out={'phase':97,'capacity':n,'repeats':a.repeats,
      'summary':summary,'rows':rows,
      'peak_mlx_memory_bytes':int(mx.get_peak_memory()),
      'validation':'bitwise keys and permutation equality every pair',
      'limitations':['Experimental write-order kernel only.',
                     'Full-array NumPy parity comparisons outside timings.',
                     'Single isolated scatter operation, not end-to-end renderer.',
                     'Source/defaults not modified.']}
 a.out.write_text(json.dumps(out,indent=2)+'\n')
 print('RESULT phase97 PASS',json.dumps(summary),flush=True)
if __name__=='__main__':main()
