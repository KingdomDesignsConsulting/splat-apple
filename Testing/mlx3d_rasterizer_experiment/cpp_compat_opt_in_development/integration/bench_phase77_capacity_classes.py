#!/usr/bin/env python3
"""Phase 77: isolated radix capacity-class reuse benchmark.

Each --mode runs in its own fresh Python process. Warm baseline small sort,
then optionally prepare a 48M capacity class or exact 47.36M shape. Measure
target and next-neighbor sort (47.0M, 47.36M, 48.0M). All prewarm costs included.
This is diagnostic; exact stable-order oracle on an independent small case.
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
 p.add_argument('--mode',choices=['cold','class48','exact'],required=True)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 cls=_renderer_class();radix=sys.modules[cls.__module__]._radix_tile_argsort
 tiles=8192
 def keys(n):
  x=mx.arange(n,dtype=mx.uint32)
  return (x*1664525+1013904223)%tiles
 def call(n):
  start=time.perf_counter();o=radix(keys(n),tiles);mx.eval(o)
  if tuple(o.shape)!=(n,):raise AssertionError('sort shape')
  return (time.perf_counter()-start)*1000
 for _ in range(3):call(33154238)
 preparation=0.
 if a.mode!='cold':
  preparation=call(48000000 if a.mode=='class48' else 47363197)
  call(33154238)
 samples=[]
 for n in (47000000,47363197,48000000):
  times=[call(n) for _ in range(3)]
  samples.append({'length':n,'first_ms':times[0],
                  'repeated_median_ms':float(np.median(times[1:])),
                  'times_ms':times})
  print('PASS',a.mode,n,times,flush=True)
 rng=np.random.default_rng(77);raw=rng.integers(0,tiles,4096,dtype=np.uint32)
 t=radix(mx.array(raw),tiles);mx.eval(t)
 np.testing.assert_array_equal(np.asarray(t),np.argsort(raw,kind='stable').astype(np.int32))
 report={'phase':77,'mode':a.mode,'prewarm_ms':preparation,'samples':samples,
         'total_including_prewarm_ms':float(preparation+sum(sum(r['times_ms']) for r in samples)),
         'memory':{k:int(getattr(mx,k)()) for k in ('get_active_memory','get_peak_memory','get_cache_memory')},
         'notes':['Distinct processes required per mode.','Class48 is a capacity class prewarm, not actual class-based production sorting.','Execution order and shared system caches can affect measurements.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT',a.mode,'total',report['total_including_prewarm_ms'],flush=True)
if __name__=='__main__':main()
