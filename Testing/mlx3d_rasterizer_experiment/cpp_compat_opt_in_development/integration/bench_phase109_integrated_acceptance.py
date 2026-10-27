#!/usr/bin/env python3
"""Phase109: fresh-process repeat of actual integrated scatter512 opt-in.

Each subprocess runs Phase108 (real independent renderer instances, source
integrity and bitwise RGB/alpha). Adds isolated sorter fallback correctness
for unsupported tile domains to distinguish fallback from renderer acceptance.
No global monkeypatch or default change.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class
PHASE108=Path(__file__).with_name('bench_phase108_integrated_scatter512.py')

def fallback_checks():
 cls=_renderer_class()
 samples=[(1048576,256),(1048576,8160),(8388608,256)]
 report=[]
 for n,tiles in samples:
  x=mx.arange(n,dtype=mx.uint32)
  keys=mx.where(x%11==0,mx.array(0xffffffff,dtype=mx.uint32),
                (x*1664525+1013904223)%tiles)
  mx.eval(keys)
  # Explicitly test the production final3 eligibility, and direct sorter
  # parity. Native renderer mode delegates to radix4 for all these cases.
  from inspect import signature
  module=sys.modules[cls.__module__]
  eligible=(n>=8000000 and tiles>=8000 and module.final_pass_plan(tiles)['eligible'])
  if eligible:raise AssertionError(('test intended as fallback',n,tiles))
  baseline=module._radix_tile_argsort(keys,tiles)
  candidate=module._selective_final3_argsort(keys,tiles,scatter_group=512)
  mx.eval(baseline,candidate)
  if not bool(mx.all(baseline==candidate).item()):
   raise AssertionError(('fallback permutation differs',n,tiles))
  report.append({'slots':n,'tiles':tiles,'expected_radix4_fallback':True,'parity':'PASS'})
  print('PASS fallback',n,tiles,flush=True)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--blocks',type=int,default=3)
 p.add_argument('--out-dir',type=Path,required=True)
 a=p.parse_args()
 if a.blocks<2:p.error('blocks must be >=2')
 if a.out_dir.exists():p.error(f'output directory exists: {a.out_dir}')
 a.out_dir.mkdir(parents=True)
 fallback=fallback_checks()
 stats=[]
 for i in range(a.blocks):
  filename=a.out_dir/f'phase108_block_{i}.json'
  print('START integrated renderer process',i,flush=True)
  run=subprocess.run([sys.executable,str(PHASE108),'--rounds','3','--out',str(filename)],
   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  print(run.stdout,flush=True)
  if run.returncode:raise SystemExit(f'block {i} failed with {run.returncode}')
  data=json.loads(filename.read_text())
  if data['parity']!='PASS exact RGB and alpha':raise AssertionError(('parity',i))
  stats.append({'block':i,'warmed_median_ms':data['warmed_median_ms'],
    'warmed_relative_percent':data['warmed_relative_percent'],
    'warmed_win_rate_512':data['warmed_win_rate_512'],
    'source_guard':data['source_guard']})
  print('BLOCK_RESULT',json.dumps(stats[-1]),flush=True)
 report={'phase':109,'blocks':a.blocks,'fallback':fallback,'integrated':stats,
  'all_parity':'PASS','integrated_block_win_count':sum(x['warmed_relative_percent']<0 for x in stats),
  'median_of_integrated_block_delta_pct':float(np.median([x['warmed_relative_percent'] for x in stats])),
  'limitations':['Fallback checks are direct source sorter parity, not complete renderer fallback scenes.',
                 'Phase108 comparison order is deterministic alternating by pose.',
                 'No default changes; source integrity checked by each subprocess.']}
 (a.out_dir/'phase109_summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase109 PASS',json.dumps({k:v for k,v in report.items() if k!='integrated'}),flush=True)
if __name__=='__main__':main()
