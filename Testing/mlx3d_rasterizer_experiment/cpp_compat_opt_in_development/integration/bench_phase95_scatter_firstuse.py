#!/usr/bin/env python3
"""Phase 95: evaluate whether scatter output-shape preparation amortizes its cost.

Fresh subprocess for each mode/condition. Controls: cold and one exact-size
scratch-scatter rehearsal (using full real production rank/hist/scatter).
Preparation time and first+repeat sort times are separately reported. Does NOT
modify renderer or production defaults; reports total including preparation to
avoid misreading shifted latency as an optimization.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
CAPS=(33154187,42477817,53283577)
def child(a):
 import mlx.core as mx
 from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
 m=sys.modules[_renderer_class().__module__]
 n=a.capacity;tiles=8160
 x=mx.arange(n,dtype=mx.uint32)
 keys=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
               (x*1664525+1013904223)%tiles)
 mx.eval(keys)
 prep_start=time.perf_counter()
 if a.prep=='scatter':
  blocks=(n+255)//256
  params=mx.array([n,blocks,0],dtype=mx.int32)
  ranks,hist=m._radix_rank_hist_kernel(
   inputs=[keys,params],output_shapes=[(n,),(16,blocks)],
   output_dtypes=[mx.int32,mx.int32],
   grid=(blocks*256,1,1),threadgroup=(256,1,1))
  prefix=mx.cumsum(hist,axis=1)-hist
  totals=mx.sum(hist,axis=1)
  starts=mx.cumsum(totals)-totals
  order=mx.arange(n,dtype=mx.int32)
  next_keys,next_order=m._radix_scatter_kernel(
   inputs=[keys,order,ranks,prefix,starts,params],
   output_shapes=[(n,),(n,)],output_dtypes=[mx.uint32,mx.int32],
   grid=(n,1,1),threadgroup=(256,1,1))
  mx.eval(next_keys,next_order)
 prep_ms=(time.perf_counter()-prep_start)*1000
 rows=[]
 fn=m._radix_tile_argsort if a.mode=='radix4' else m._selective_final3_argsort
 for j in range(3):
  start=time.perf_counter()
  sorted_idx=fn(keys,tiles);mx.eval(sorted_idx)
  ms=(time.perf_counter()-start)*1000
  rows.append(ms)
  print('PASS',a.mode,a.prep,n,j,round(ms,3),flush=True)
 result={'mode':a.mode,'prep':a.prep,'capacity':n,'prep_ms':prep_ms,
         'first_ms':rows[0],'repeat_ms':rows[1:],
         'total_including_prep_ms':prep_ms+sum(rows),
         'peak_mlx_memory_bytes':int(mx.get_peak_memory()),
         'cache_mlx_memory_bytes':int(mx.get_cache_memory())}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
def parent(a):
 if a.out_dir.exists():raise SystemExit('Output directory already exists')
 a.out_dir.mkdir(parents=True)
 rows=[]
 try:
  for block in range(a.blocks):
   for cap in CAPS:
    conditions=[('radix4','cold'),('radix4','scatter'),
                ('final3','cold'),('final3','scatter')]
    if block%2:conditions.reverse()
    for mode,prep in conditions:
     out=a.out_dir/f'b{block}_n{cap}_{mode}_{prep}.json'
     cmd=[sys.executable,__file__,'--child','--mode',mode,'--prep',prep,
          '--capacity',str(cap),'--out',str(out)]
     proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
     (a.out_dir/(out.stem+'.log')).write_text(proc.stdout)
     if proc.returncode:raise RuntimeError(out.name+': '+proc.stdout[-1600:])
     obj=json.loads(out.read_text());rows.append({'block':block,**obj})
     print('PASS block',block,mode,prep,cap,
           'prep',round(obj['prep_ms'],2),'first',round(obj['first_ms'],2),
           'total',round(obj['total_including_prep_ms'],2),flush=True)
 finally:
  (a.out_dir/'partial_or_complete.json').write_text(json.dumps(rows,indent=2)+'\n')
 summaries=[]
 for cap in CAPS:
  for mode in ('radix4','final3'):
   for prep in ('cold','scatter'):
    v=[r for r in rows if r['capacity']==cap and r['mode']==mode and r['prep']==prep]
    summaries.append({'capacity':cap,'mode':mode,'prep':prep,**{
      k:float(np.median([r[k] for r in v]))
      for k in ('prep_ms','first_ms','total_including_prep_ms',
                'peak_mlx_memory_bytes','cache_mlx_memory_bytes')}})
 (a.out_dir/'summary.json').write_text(json.dumps({'phase':95,'summaries':summaries,
   'trials':rows,
   'caveats':['Scatter rehearsal performs real radix rank/scatter; it is not cost-free.',
              'Cold and rehearsal run separately in fresh processes.',
              'Evaluate total time including preparation and memory, not only first-frame.',
              'No automatic prewarm or production changes.']},indent=2)+'\n')
 print('RESULT phase95',json.dumps(summaries),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--child',action='store_true')
 p.add_argument('--mode',choices=('radix4','final3'))
 p.add_argument('--prep',choices=('cold','scatter'))
 p.add_argument('--capacity',type=int)
 p.add_argument('--out',type=Path)
 p.add_argument('--out-dir',type=Path,default=Path('phase95_scatter_firstuse'))
 p.add_argument('--blocks',type=int,default=3)
 a=p.parse_args()
 if a.blocks<1:p.error('blocks >=1')
 if a.child:
  if not a.mode or not a.prep or a.capacity not in CAPS or not a.out or a.out.exists():
   p.error('child mode/prep/capacity and unused out required')
  child(a)
 else:parent(a)
