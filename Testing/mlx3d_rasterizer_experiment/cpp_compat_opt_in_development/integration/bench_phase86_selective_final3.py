#!/usr/bin/env python3
"""Phase 86 conservative auto-dispatch vs radix4 with full Phase85 render matrix.

Uses Phase85's parity/moving-camera/scene-replacement/full-cull validation.
Additionally checks routing decisions at boundaries. Workload choice remains
opt-in via a serial scoped hook, NEVER a production default.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import bench_phase85_final3 as phase85
import selective_final3_experiment as policy
import radix_final3_experiment as original

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--rounds',type=int,default=12)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error('output exists')
 if a.rounds<6:p.error('rounds must >= 6')
 cases=[(4_000_000,8160,False),(8_000_000,8160,True),
        (33_154_238,8160,True),(33_154_238,3600,False),
        (33_154_238,8192,True),(33_154_238,65536,False)]
 for slots,tiles,want in cases:
  got=policy.select_final3(slots,tiles)
  if got!=want:raise AssertionError(('dispatch boundary',slots,tiles,got,want))
 # Reuse exact Phase85 bitwise renderer assertions and varied-camera matrix.
 saved=original.renderer_sort_override
 original.renderer_sort_override=lambda variant:policy.selective_final3_override()
 try:
  data=phase85.render_benchmark([4096,100000,700000],
                               [(1280,720),(1920,1080)],a.rounds,
                               "indices_only")
 finally:
  original.renderer_sort_override=saved
 result={'phase':86,'policy':{'min_slots':policy.MIN_SLOTS,
                             'min_tiles':policy.MIN_TILES},
         'routing_boundary_cases':cases,'renderer':data,
         'notes':['Scoped, serial opt-in hook. Production code/defaults unchanged.',
                  'Underlying Phase85 benchmark uses the final3 label for selective policy.',
                  'All renderer image and alpha comparisons are bitwise.',
                  'Includes fixed frames, moving poses, changed scene and full cull.']}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('RESULT phase86 PASS',[(r['gaussians'],r['resolution'],
      {k:{name:round(value,3) for name,value in v['median_ms'].items()}
       for k,v in r['steady'].items()}) for r in data['cases']],flush=True)
if __name__=='__main__':main()
