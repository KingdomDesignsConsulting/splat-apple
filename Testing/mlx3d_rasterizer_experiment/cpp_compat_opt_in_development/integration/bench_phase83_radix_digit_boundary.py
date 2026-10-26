#!/usr/bin/env python3
"""Phase 83: characterize radix pass-count boundaries and sentinel correctness.

Tests tile counts around 8/12-bit radix boundaries at fixed key count.
Each timed sort runs repeated after warmup. Varies sentinel fraction and
verifies NumPy exact stable ordering on an independent small key array.
No renderer or installed MLX files are changed.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
SENTINEL=np.uint32(0xffffffff)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--n',type=int,default=33154238)
 p.add_argument('--rounds',type=int,default=5)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if a.n<4096 or a.rounds<3:p.error('invalid test size')
 cls=_renderer_class();radix=sys.modules[cls.__module__]._radix_tile_argsort
 cases=[]
 for tiles in (255,256,4095,4096,8160,8192):
  for masked in (False,True):
   x=mx.arange(a.n,dtype=mx.uint32)
   keys=(x*1664525+1013904223)%tiles
   if masked:
    keys=mx.where(x%7==0,mx.array(SENTINEL,dtype=mx.uint32),keys)
   mx.eval(keys)
   def run():
    start=time.perf_counter()
    order=radix(keys,tiles);mx.eval(order)
    return (time.perf_counter()-start)*1000
   run()
   timings=[run() for _ in range(a.rounds)]
   raw=np.random.default_rng(83000+tiles+int(masked)).integers(0,tiles,4096,dtype=np.uint32)
   if masked:raw[::7]=SENTINEL
   order=radix(mx.array(raw),tiles);mx.eval(order)
   np.testing.assert_array_equal(np.asarray(order),
                                 np.argsort(raw,kind='stable').astype(np.int32))
   row={'tiles':tiles,'masked':masked,
        'passes':(max(1,tiles.bit_length())+3)//4,
        'median_ms':float(np.median(timings)),
        'min_ms':min(timings),'max_ms':max(timings),
        'samples_ms':timings,'oracle':'pass'}
   cases.append(row)
   print('PASS',tiles,'masked',masked,'passes',row['passes'],
         'median_ms',round(row['median_ms'],3),flush=True)
 report={'phase':83,'n':a.n,'rounds':a.rounds,'cases':cases,
         'memory':{name:int(getattr(mx,name)()) for name in
                    ('get_active_memory','get_peak_memory','get_cache_memory')},
         'notes':['Same key count and stable reference check for each configuration.',
                  'Tile counts vary distribution; pass boundary not perfectly isolated.',
                  'Sentinel 0xffffffff preserved for masked cases.',
                  'GPU conditions and Metal cache may influence elapsed durations.',
                  'No change to validated production radix implementation.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase83',[(r['tiles'],r['masked'],round(r['median_ms'],3)) for r in cases],flush=True)
if __name__=='__main__':main()
