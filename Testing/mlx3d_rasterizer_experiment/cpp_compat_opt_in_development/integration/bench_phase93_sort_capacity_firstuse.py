#!/usr/bin/env python3
"""Phase 93: separate first-use sorter shape work from full renderer work.

Fresh process per sort implementation, counterbalanced blocks. Pre-evaluate
the *same* keys for all target sizes BEFORE timed sorting; run each length
twice, then revisit. This excludes lazy key materialization from sort timers.
Control includes a key-only mx.eval() timing. Results cannot distinguish
Metal compilation from allocator initialization without GPU tracing.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
CAPS=(33154187,42477817,53283577)
STEPS=(0,0,1,1,2,2,1,0,2)
def child(a):
 import mlx.core as mx
 from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
 module=sys.modules[_renderer_class().__module__]
 tiles=8160
 arr=[]
 for n in CAPS:
  t=time.perf_counter()
  x=mx.arange(n,dtype=mx.uint32)
  keys=mx.where(x%7==0,mx.array(0xffffffff,dtype=mx.uint32),
                (x*1664525+1013904223)%tiles)
  mx.eval(keys)
  arr.append(keys)
  print('PASS key preparation',n,round((time.perf_counter()-t)*1000,3),flush=True)
 original=module._radix_tile_argsort
 def choose(x):
  if a.mode=='radix4':return original(x,tiles)
  return module._selective_final3_argsort(x,tiles)
 rows=[]
 for step,index in enumerate(STEPS):
  t=time.perf_counter()
  order=choose(arr[index]);mx.eval(order)
  elapsed=(time.perf_counter()-t)*1000
  if tuple(order.shape)!=(CAPS[index],):raise AssertionError('sort output shape')
  row={'step':step,'capacity':CAPS[index],'ms':elapsed}
  rows.append(row)
  print('PASS sort',a.mode,step,CAPS[index],round(elapsed,3),flush=True)
 # Moderate-size exact stable-order oracle for BOTH sorter modes, including sentinel.
 rng=np.random.default_rng(93)
 raw=rng.integers(0,tiles,4096,dtype=np.uint32);raw[::7]=np.uint32(0xffffffff)
 out=choose(mx.array(raw));mx.eval(out)
 np.testing.assert_array_equal(np.asarray(out),np.argsort(raw,kind='stable').astype(np.int32))
 report={'mode':a.mode,'rows':rows,
         'peak_memory_bytes':int(mx.get_peak_memory()),
         'cache_memory_bytes':int(mx.get_cache_memory()),
         'oracle':'4096 stable-order NumPy PASS'}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
def parent(a):
 if a.out_dir.exists():raise SystemExit('Output directory exists')
 a.out_dir.mkdir(parents=True)
 trials=[]
 try:
  for block in range(a.blocks):
   names=('radix4','final3') if block%2==0 else ('final3','radix4')
   for pos,mode in enumerate(names):
    output=a.out_dir/f'block{block:02d}_pos{pos}_{mode}.json'
    cmd=[sys.executable,__file__,'--child','--mode',mode,'--out',str(output)]
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (a.out_dir/f'block{block:02d}_pos{pos}_{mode}.log').write_text(proc.stdout)
    if proc.returncode:raise RuntimeError(f'{mode} failed; see log: {proc.stdout[-1500:]}')
    result=json.loads(output.read_text())
    trials.append({'block':block,'position':pos,**result})
    print('PASS block',block,mode,[round(x['ms'],2) for x in result['rows']],flush=True)
 finally:
  (a.out_dir/'partial_or_complete.json').write_text(json.dumps(trials,indent=2)+'\n')
 summaries={}
 for mode in ('radix4','final3'):
  cases=[t for t in trials if t['mode']==mode]
  summaries[mode]={'median_ms_by_step':[
    float(np.median([r['rows'][i]['ms'] for r in cases])) for i in range(len(STEPS))],
    'median_peak_bytes':float(np.median([r['peak_memory_bytes'] for r in cases])),
    'median_cache_bytes':float(np.median([r['cache_memory_bytes'] for r in cases]))}
 report={'phase':93,'capacities':CAPS,'sequence':STEPS,'blocks':a.blocks,
         'summary':summaries,'trials':trials,
         'notes':['All key input buffers fully evaluated outside timing.',
                  'Independent processes per mode; counterbalanced block ordering.',
                  'No full rendering or compositing in timed code.',
                  'Cross-process system GPU caches remain a possible confounder.',
                  'No production defaults changed.']}
 (a.out_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase93',json.dumps(summaries),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--child',action='store_true')
 p.add_argument('--mode',choices=('radix4','final3'))
 p.add_argument('--out',type=Path)
 p.add_argument('--out-dir',type=Path,default=Path('phase93_sort_capacity'))
 p.add_argument('--blocks',type=int,default=4)
 a=p.parse_args()
 if a.blocks<1:p.error('blocks >=1')
 if a.child:
  if not a.mode or not a.out or a.out.exists():p.error('child requires mode and unused out')
  child(a)
 else:parent(a)
