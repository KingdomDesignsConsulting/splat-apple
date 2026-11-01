#!/usr/bin/env python3
"""Phase 88: exercise BOTH branches of selective final3 on real render workloads.

Use separate matched renderers and randomized execution order, compare bitwise
RGB/alpha, trace actual per-frame branch selection, and test changing poses.
Scoped experimental patch only; no change to production renderer or defaults.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera
from bench_phase57_active_sort import pose
from bench_phase85_final3 import fast_camera,assert_render_equal
import selective_final3_experiment as policy
import radix_final3_experiment as experiment

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--rounds',type=int,default=8)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if a.rounds<4:p.error('rounds must be >=4')
 rng=np.random.default_rng(88)
 module, original=experiment.backend()
 rows=[]
 # Different resolutions ensure both sides are selected on actual renderer calls.
 workloads=[(4096,1920,1080),(100000,1920,1080),
            (700000,1280,720),(700000,1920,1080)]
 for n,w,h in workloads:
  gs=scene(n)
  engines={name:PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded',
                     tile_sort_impl='radix4',tile_range_mode='auto')
           for name in ('baseline','selective')}
  for step in range(a.rounds):
   cam=fast_camera(camera(w,h,3) if step<2 else pose(w,h,[0,1,14,15,16,29][(step-2)%6]))
   frames={};times={};choices=[]
   def traced(keys,tiles):
    slots=int(keys.size);tile_count=int(tiles)
    choice='final3' if policy.select_final3(slots,tile_count) else 'radix4'
    choices.append({'choice':choice,'slots':slots,'tiles':tile_count})
    if choice=='final3':
     return experiment.stable_final3_argsort(keys,tiles,variant='indices_only')
    return original(keys,tiles)
   for name in rng.permutation(['baseline','selective']).tolist():
    if module._radix_tile_argsort is not original:raise AssertionError('patch leaked')
    start=time.perf_counter()
    if name=='selective':module._radix_tile_argsort=traced
    try:
     frames[name]=engines[name]._renderer.render(cam)
     mx.eval(frames[name]['image'],frames[name]['alpha'])
    finally:
     module._radix_tile_argsort=original
    times[name]=(time.perf_counter()-start)*1000
   assert_render_equal({'radix4':frames['baseline'],'final3':frames['selective']},
                       ('phase88',n,w,h,step))
   if len(choices)!=1:raise AssertionError(('unexpected dispatch count',n,w,h,step,choices))
   record={'gaussians':n,'resolution':[w,h],'step':step,
           'selection':choices[0], 'times_ms':times,
           'counts':dict(engines['selective']._renderer.last_frame_counts)}
   rows.append(record)
   print('PASS',n,w,h,step,choices[0]['choice'],
         'capacity',choices[0]['slots'],
         'baseline',round(times['baseline'],3),
         'selective',round(times['selective'],3),flush=True)
 if module._radix_tile_argsort is not original:raise AssertionError('patch leaked')
 actual={name:sum(r['selection']['choice']==name for r in rows)
         for name in ('radix4','final3')}
 if min(actual.values())==0:raise AssertionError(('both branches required',actual))
 summaries=[]
 for n,w,h in workloads:
  r=[x for x in rows if x['gaussians']==n and x['resolution']==[w,h]]
  r=r[2:] # omit startup and repeated fixed-camera warmup from median
  summaries.append({'gaussians':n,'resolution':[w,h],
    'branches':list(sorted({x['selection']['choice'] for x in r})),
    'baseline_median_ms':float(np.median([x['times_ms']['baseline'] for x in r])),
    'selective_median_ms':float(np.median([x['times_ms']['selective'] for x in r]))})
 report={'phase':88,'rounds':a.rounds,'actual_branch_counts':actual,
         'moving_pose_summaries':summaries,'rows':rows,
         'bitwise_rgb_alpha_parity':'PASS',
         'notes':['Paired order randomized; startup frames excluded from moving medians.',
                  'Dispatch selection traced once per frame; both branches required.',
                  'Guarded capacity can grow on pose transitions.',
                  'Scoped process-global experiment hook is serial-only, not production safe.',
                  'No change to production code, opt-in settings or PR merge status.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase88 PASS',actual,summaries,flush=True)
if __name__=='__main__':main()
