#!/usr/bin/env python3
"""Phase 89: test native per-instance final3 dispatch, without global monkeypatch.

Uses same workloads and moving poses as Phase88. Validates bitwise RGB/alpha
between radix4 and per-instance selective_final3, reports actual decisions
from capacity/tile policy and paired randomized timing. Must run on Mac gs-mps.
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

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--rounds',type=int,default=8)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if a.rounds<4:p.error('rounds must >=4')
 rng=np.random.default_rng(89)
 rows=[]
 for n,w,h in [(4096,1920,1080),(100000,1920,1080),
               (700000,1280,720),(700000,1920,1080)]:
  gs=scene(n)
  engines={name:PersistentCppCompatRenderer(
    gs,duplicate_sort_mode='guarded',
    tile_sort_impl=('radix4' if name=='radix4' else 'selective_final3'),
    tile_range_mode='auto') for name in ('radix4','selective')}
  for i in range(a.rounds):
   cam=fast_camera(camera(w,h,3) if i<2 else pose(w,h,[0,1,14,15,16,29][(i-2)%6]))
   frames={};times={}
   for name in rng.permutation(['radix4','selective']).tolist():
    start=time.perf_counter()
    frames[name]=engines[name]._renderer.render(cam)
    mx.eval(frames[name]['image'],frames[name]['alpha'])
    times[name]=(time.perf_counter()-start)*1000
   assert_render_equal({'radix4':frames['radix4'],'final3':frames['selective']},
                       ('phase89',n,w,h,i))
   counts=engines['selective']._renderer.last_frame_counts
   tiles=((w+15)//16)*((h+15)//16)
   slots=int(counts['sorted'])
   use=slots>=8_000_000 and tiles>=8_000 and ((tiles-1)>>(((tiles.bit_length()-1)//4)*4))<2
   row={'gaussians':n,'resolution':[w,h],'step':i,
        'selected':'final3' if use else 'radix4','slots':slots,
        'times_ms':times,'counts':dict(counts)}
   rows.append(row)
   print('PASS',n,w,h,i,row['selected'],slots,
         round(times['radix4'],3),round(times['selective'],3),flush=True)
 from collections import Counter
 choices=dict(Counter(x['selected'] for x in rows))
 if not choices.get('final3') or not choices.get('radix4'):
  raise AssertionError(('one branch untested',choices))
 summaries=[]
 for n,w,h in [(4096,1920,1080),(100000,1920,1080),
               (700000,1280,720),(700000,1920,1080)]:
  arows=[r for r in rows if r['gaussians']==n and r['resolution']==[w,h]][2:]
  summaries.append({'gaussians':n,'resolution':[w,h],
   'baseline_median_ms':float(np.median([r['times_ms']['radix4'] for r in arows])),
   'selective_median_ms':float(np.median([r['times_ms']['selective'] for r in arows]))})
 result={'phase':89,'rounds':a.rounds,'branches':choices,'summaries':summaries,
         'rows':rows,'parity':'bitwise RGB and alpha PASS',
         'implementation':'renderer instance tile_sort_impl=selective_final3; no monkeypatch',
         'notes':['Opt-in new sort setting only; default auto unchanged.',
                  'Actual production snapshot parity/compile requires Mac validation.',
                  'All matched rendering timed with randomized paired order.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase89 PASS',choices,summaries,flush=True)
if __name__=='__main__':main()
