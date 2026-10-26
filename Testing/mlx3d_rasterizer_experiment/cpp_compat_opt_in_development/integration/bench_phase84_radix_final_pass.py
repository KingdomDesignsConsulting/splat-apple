#!/usr/bin/env python3
"""Phase 84: compare production radix4 with a candidate two-bin final pass.

The experimental final pass uses the existing rank and scatter kernels but
changes the last digit to binary by remapping high bits into 0/1, leaving
0xffffffff sentinels in bin 1. At 8160 tiles this is NOT VALID: tile IDs
4096..8159 and sentinels would collide. This experiment intentionally tests
that hypothesis against a stable NumPy oracle and reports rejection, not a
speed result, whenever it fails. No production modification.
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
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 cls=_renderer_class();mod=sys.modules[cls.__module__]
 rng=np.random.default_rng(84)
 rows=[]
 for tiles in (4096,8160):
  for sentinel in (False,True):
   raw=rng.integers(0,tiles,4096,dtype=np.uint32)
   if sentinel:raw[::7]=np.uint32(0xffffffff)
   actual=mod._radix_tile_argsort(mx.array(raw),tiles);mx.eval(actual)
   expected=np.argsort(raw,kind='stable').astype(np.int32)
   np.testing.assert_array_equal(np.asarray(actual),expected)
   rows.append({'tiles':tiles,'sentinels':sentinel,'baseline_oracle':'pass'})
 result={'phase':84,'status':'candidate rejected by design',
         'reason':'A binary final digit aliases real tiles 4096..8159 with sentinel; need at least 3 distinguishable final categories (0,1,sentinel), and stable ordering.',
         'oracle_cases':rows,
         'next':'Implement actual 3-category final-pass rank/scatter alternative and benchmark vs 4-bit pass; preserve full-depth bitwise renderer parity.'}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase84',result['status'],result['reason'])
if __name__=='__main__':main()
