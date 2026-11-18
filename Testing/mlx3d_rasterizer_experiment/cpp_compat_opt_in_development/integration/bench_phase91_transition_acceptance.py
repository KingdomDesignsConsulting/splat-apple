#!/usr/bin/env python3
"""Phase 91: native final3 vs radix4 camera-transition acceptance.

Compare independent renderer instances using paired alternating execution
orders, record stage-neutral frame wall time and MLX memory, and assert
bitwise image/alpha identity at each pose. No monkeypatch, no defaults change.
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
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--n',type=int,default=700000)
 p.add_argument('--rounds',type=int,default=2)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if a.rounds<2:p.error('rounds >=2')
 gs=scene(a.n)
 names=['radix4','selective_final3']
 engines={name:PersistentCppCompatRenderer(gs,duplicate_sort_mode='guarded',
           tile_sort_impl=name,tile_range_mode='auto') for name in names}
 rng=np.random.default_rng(91)
 positions=[3,3,0,1,14,15,16,29,30,31,34,37,40,44,3]
 rows=[]
 for round_i in range(a.rounds):
  for idx,k in enumerate(positions):
   cam=fast_camera(camera(1920,1080,3) if idx<2 else pose(1920,1080,k))
   images={};ms={};mem={}
   for name in rng.permutation(names).tolist():
    t=time.perf_counter()
    image=engines[name]._renderer.render(cam)
    mx.eval(image['image'],image['alpha'])
    images[name]=image
    ms[name]=(time.perf_counter()-t)*1000
    mem[name]=int(mx.get_active_memory())
   assert_render_equal({'radix4':images['radix4'],'final3':images['selective_final3']},
                       (round_i,idx,k))
   left=engines['radix4']._renderer.last_frame_counts
   right=engines['selective_final3']._renderer.last_frame_counts
   if left!=right:raise AssertionError(('workload mismatch',round_i,idx,k,left,right))
   rows.append({'round':round_i,'step':idx,'pose':k,'ms':ms,
                'active_memory_bytes_after':mem,'counts':dict(left)})
   print('PASS',round_i,idx,k,'capacity',left.get('sorted'),
         'baseline',round(ms['radix4'],3),
         'selective',round(ms['selective_final3'],3),flush=True)
 paired=[r for r in rows if r['step']>=2]
 def pct(name,q):return float(np.percentile([r['ms'][name] for r in paired],q))
 result={'phase':91,'n':a.n,'rounds':a.rounds,'parity':'bitwise RGB and alpha PASS',
         'steady_and_transitions':{name:{'median_ms':pct(name,50),'p95_ms':pct(name,95),
               'max_ms':max(r['ms'][name] for r in paired),
               'total_ms':sum(r['ms'][name] for r in paired)} for name in names},
         'peak_mlx_bytes':int(mx.get_peak_memory()),'rows':rows,
         'notes':['Workload parity checked after each frame; randomized paired order.',
                  'Both instances share process/allocator: memory is NOT per-renderer peak.',
                  'Not a GPU hardware timestamp benchmark.',
                  'No change to defaults; experimental opt-in only.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase91',json.dumps(result['steady_and_transitions']),flush=True)
if __name__=='__main__':main()
