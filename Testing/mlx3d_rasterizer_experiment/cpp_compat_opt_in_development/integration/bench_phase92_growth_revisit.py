#!/usr/bin/env python3
"""Phase 92: isolate cold capacity transition and warm revisit across sorters.

Fresh processes per mode and counterbalanced block ordering. Within each
process, render the same fixed HD camera, force known capacities representing
Phase 91 steps, revisit them in order, and record first/repeated durations.
This is a diagnostic forced-capacity benchmark, not the normal guarded growth
policy. Bitwise RGB/alpha parity against radix4 at every capacity is checked
in a separate paired stage in Phase91; this script measures modes separately.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
CAPS=(33154187,42477817,53283577)
SEQUENCE=(0,0,1,1,2,2,1,0,2)
def child(a):
 import mlx.core as mx
 from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
 from bench_phase49_breakdown import scene,camera
 from bench_phase85_final3 import fast_camera
 gs=scene(a.n)
 renderer=PersistentCppCompatRenderer(gs,duplicate_sort_mode='capacity',
              tile_sort_impl=a.mode,tile_range_mode='auto')
 cam=fast_camera(camera(1920,1080,3))
 rows=[]
 # Initialize full graph once at original known-valid capacity.
 for step,idx in enumerate(SEQUENCE):
  renderer._renderer._capacity=CAPS[idx]
  t=time.perf_counter()
  frame=renderer._renderer.render(cam)
  mx.eval(frame['image'],frame['alpha'])
  elapsed=(time.perf_counter()-t)*1000
  counts=dict(renderer._renderer.last_frame_counts)
  if counts['sorted']!=CAPS[idx]:
   raise AssertionError(('capacity mismatch',step,counts,CAPS[idx]))
  row={'step':step,'capacity':CAPS[idx],'ms':elapsed,'counts':counts}
  rows.append(row)
  print('PASS',a.mode,step,CAPS[idx],round(elapsed,3),flush=True)
 report={'mode':a.mode,'frames':rows,'peak_memory':int(mx.get_peak_memory()),
         'cache_memory':int(mx.get_cache_memory())}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
def parent(a):
 if a.out_dir.exists():raise SystemExit('Output directory already exists')
 a.out_dir.mkdir(parents=True)
 results=[]
 try:
  for block in range(a.blocks):
   order=('radix4','selective_final3') if block%2==0 else ('selective_final3','radix4')
   for pos,mode in enumerate(order):
    output=a.out_dir/f'block{block:02d}_pos{pos}_{mode}.json'
    cmd=[sys.executable,__file__,'--child','--mode',mode,
         '--out',str(output),'--n',str(a.n)]
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (a.out_dir/f'block{block:02d}_pos{pos}_{mode}.log').write_text(proc.stdout)
    if proc.returncode:raise RuntimeError('See log for '+str(output)+': '+proc.stdout[-1500:])
    data=json.loads(output.read_text())
    results.append({'block':block,'position':pos,**data})
    print('PASS block',block,mode,
          [(x['capacity'],round(x['ms'],2)) for x in data['frames']],flush=True)
 finally:
  (a.out_dir/'partial_or_complete.json').write_text(json.dumps(results,indent=2)+'\n')
 summary={}
 for mode in ('radix4','selective_final3'):
  cases=[r for r in results if r['mode']==mode]
  summary[mode]={'median_ms_by_sequence_step':[
    float(np.median([r['frames'][step]['ms'] for r in cases]))
    for step in range(len(SEQUENCE))],
    'median_peak_memory':float(np.median([r['peak_memory'] for r in cases])),
    'median_cache_memory':float(np.median([r['cache_memory'] for r in cases]))}
 (a.out_dir/'summary.json').write_text(json.dumps({
   'phase':92,'blocks':a.blocks,'capacities':CAPS,'sequence':SEQUENCE,
   'summary':summary,'trials':results,
   'caveats':['Forced-capacity, fixed-camera diagnostic; not production guarded growth.',
              'Capacity must remain >= active duplicate count to preserve correct output.',
              'Separate-process runs avoid simultaneous renderer allocator interference.',
              'Phase91 handles per-frame bitwise parity; this script measures timing and shape.'],
   },indent=2)+'\n')
 print('RESULT phase92',json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--child',action='store_true')
 p.add_argument('--mode',choices=('radix4','selective_final3'))
 p.add_argument('--out',type=Path)
 p.add_argument('--out-dir',type=Path,default=Path('phase92_growth_revisit'))
 p.add_argument('--blocks',type=int,default=4)
 p.add_argument('--n',type=int,default=700000)
 a=p.parse_args()
 if a.blocks<1:p.error('blocks >=1')
 if a.child:
  if not a.mode or not a.out or a.out.exists():p.error('child requires mode and unused out')
  child(a)
 else:parent(a)
