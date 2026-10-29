#!/usr/bin/env python3
"""Phase105: counterbalanced fresh-process renderer growth-order test.

Runs Phase103 in fresh processes, alternating which threadgroup executes
first at the two initial capacity increases (33M->42M and 42M->53M).
Random non-growth frame orders use varying seeds. Analyze growth transitions
separately from steady-state. Does not modify validated source or defaults.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
PHASE103=Path(__file__).with_name('bench_phase103_renderer_scatter512.py')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--pairs',type=int,default=4)
 p.add_argument('--out-dir',type=Path,required=True)
 a=p.parse_args()
 if a.pairs<2:p.error('--pairs must be >=2')
 if a.out_dir.exists():p.error(f'output directory exists: {a.out_dir}')
 a.out_dir.mkdir(parents=True)
 blocks=[];allrows=[]
 for pair in range(a.pairs):
  for order in ('256-first','512-first'):
   block=2*pair+(order=='512-first')
   path=a.out_dir/f'block_{block}_{order}.json'
   cmd=[sys.executable,str(PHASE103),'--rounds','2',
     '--seed',str(105+block),'--growth-order',order,'--out',str(path)]
   print('START',block,order,flush=True)
   run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   print(run.stdout,flush=True)
   if run.returncode:raise SystemExit(f'block {block} failed with exit {run.returncode}')
   data=json.loads(path.read_text())
   rows=data['rows']
   for r in rows:r['block']=block;r['growth_order']=order
   allrows.extend(rows)
   selected=[r for r in rows if r['step']>=2]
   med={k:float(np.median([r['ms'][k] for r in selected]))
        for k in ('256','512')}
   growth=[r for r in rows if r['round']==0 and r['step'] in (9,10)]
   blocks.append({'block':block,'growth_order':order,'median_ms':med,
    'median_pct_512':100*(med['512']/med['256']-1),
    'growth':[{'step':r['step'],'slots':r['counts'].get('sorted'),
     'ms':r['ms'],'difference_ms_512_minus_256':r['ms']['512']-r['ms']['256']} for r in growth]})
   print('BLOCK',json.dumps(blocks[-1]),flush=True)
 non_growth=[r for r in allrows if r['step']>=2 and r['step'] not in (9,10)]
 pooled={}
 for order in ('256-first','512-first'):
  chosen=[r for r in allrows if r['growth_order']==order]
  first_growth=[r for r in chosen if r['round']==0 and r['step'] in (9,10)]
  steady=[r for r in non_growth if r['growth_order']==order]
  pooled[order]={
   'initial_growth_by_step':{str(step):{
     'median_delta_ms_512_minus_256':float(np.median([
      r['ms']['512']-r['ms']['256'] for r in first_growth if r['step']==step])),
     'per_block_delta_ms_512_minus_256':[r['ms']['512']-r['ms']['256'] for r in first_growth if r['step']==step]
    } for step in (9,10)},
   'steady_pair_median_delta_ms_512_minus_256':float(np.median([
    r['ms']['512']-r['ms']['256'] for r in steady])),
   'steady_512_win_rate':float(np.mean([r['ms']['512']<r['ms']['256'] for r in steady]))}
 report={'phase':105,'processes':len(blocks),'all_parity':'PASS per Phase103 block',
  'counterbalanced_growth_results':pooled,'blocks':blocks,
  'limitations':['Two run orders across independent processes, not isolated GPU kernels.',
   'Growth may combine compilation, allocation, and compositor execution.',
   'Only selected cameras and fixed scene; no production integration.']}
 (a.out_dir/'phase105_summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase105 PASS',json.dumps(pooled),flush=True)
if __name__=='__main__':main()
