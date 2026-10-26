#!/usr/bin/env python3
"""Phase 90: isolate first-use latency of native instance-local final3.

Fresh Python processes per renderer, alternating order across blocks. Each
process builds only one renderer for a 700K FullHD fixed camera, then executes
one initial frame and four repeats. Compare init, first, subsequent median and
total. No prewarm, no module patch. Does not assert machine-wide cache isolation.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np

def child(a):
 import mlx.core as mx
 from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
 from bench_phase49_breakdown import scene,camera
 from bench_phase85_final3 import fast_camera
 start=time.perf_counter()
 gs=scene(a.n)
 e=PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded',
        tile_sort_impl=a.mode,tile_range_mode='auto')
 setup=(time.perf_counter()-start)*1000
 cam=fast_camera(camera(a.width,a.height,3))
 rows=[]
 for index in range(a.frames):
  t=time.perf_counter()
  frame=e._renderer.render(cam)
  mx.eval(frame['image'],frame['alpha'])
  ms=(time.perf_counter()-t)*1000
  rows.append(ms)
  print('PASS',a.mode,index,round(ms,3),flush=True)
 report={'mode':a.mode,'setup_ms':setup,'frame_times_ms':rows,
         'first_frame_ms':rows[0],
         'repeat_median_ms':float(np.median(rows[1:])),
         'total_including_setup_ms':float(setup+sum(rows)),
         'peak_memory':int(mx.get_peak_memory()),
         'counts':dict(e._renderer.last_frame_counts)}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT child',a.mode,report['first_frame_ms'],
       report['repeat_median_ms'],flush=True)

def parent(a):
 if a.out_dir.exists():raise SystemExit('output directory already exists')
 a.out_dir.mkdir(parents=True)
 rows=[]
 try:
  for block in range(a.blocks):
   modes=('radix4','selective_final3') if block%2==0 else ('selective_final3','radix4')
   for pos,mode in enumerate(modes):
    out=a.out_dir/f'block{block}_position{pos}_{mode}.json'
    cmd=[sys.executable,__file__,'--child','--mode',mode,'--out',str(out),
         '--n',str(a.n),'--width',str(a.width),'--height',str(a.height),
         '--frames',str(a.frames)]
    result=subprocess.run(cmd,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True)
    (a.out_dir/f'block{block}_position{pos}_{mode}.log').write_text(result.stdout)
    if result.returncode:raise RuntimeError(f'{mode} process failed: {result.stdout[-1000:]}')
    value=json.loads(out.read_text())
    rows.append({'block':block,'position':pos,**value})
    print('PASS block',block,mode,'first',round(value['first_frame_ms'],3),
          'repeat',round(value['repeat_median_ms'],3),flush=True)
 finally:
  (a.out_dir/'partial_or_complete.json').write_text(json.dumps(rows,indent=2)+'\n')
 summary={}
 for mode in ('radix4','selective_final3'):
  entries=[r for r in rows if r['mode']==mode]
  summary[mode]={field:float(np.median([r[field] for r in entries]))
                 for field in ('setup_ms','first_frame_ms','repeat_median_ms',
                               'total_including_setup_ms','peak_memory')}
 (a.out_dir/'summary.json').write_text(json.dumps({'phase':90,
     'blocks':a.blocks,'frames_per_process':a.frames,
     'summary_median':summary,'trials':rows,
     'notes':['Each mode fresh Python process, alternating block order.',
              'Cross-process GPU driver caches and clock behavior not isolated.',
              'First-use and warm measurements do not imply automatic prewarming.',
              'No production defaults modified.']},indent=2)+'\n')
 print('RESULT phase90',json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--child',action='store_true')
 p.add_argument('--mode',choices=('radix4','selective_final3'))
 p.add_argument('--out',type=Path)
 p.add_argument('--out-dir',type=Path,default=Path('phase90_cold_start'))
 p.add_argument('--blocks',type=int,default=4)
 p.add_argument('--frames',type=int,default=5)
 p.add_argument('--n',type=int,default=700000)
 p.add_argument('--width',type=int,default=1920)
 p.add_argument('--height',type=int,default=1080)
 a=p.parse_args()
 if a.frames<2 or a.blocks<1:p.error('frames >=2; blocks >=1')
 if a.child:
  if not a.mode or not a.out or a.out.exists():p.error('child mode and unused out required')
  child(a)
 else:parent(a)
