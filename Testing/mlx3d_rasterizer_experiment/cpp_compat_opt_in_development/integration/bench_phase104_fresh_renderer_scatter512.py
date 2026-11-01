#!/usr/bin/env python3
"""Phase104 fresh-process replication of Phase103 renderer scatter A/B.

Runs the unchanged Phase103 parity workload in independent Python processes,
then summarizes paired per-frame differences and capacity-transition outliers.
No production mutations. Results are retained in distinct per-block JSONs.
"""
import argparse,json,statistics,subprocess,sys
from pathlib import Path
import numpy as np
SCRIPT=Path(__file__).with_name('bench_phase103_renderer_scatter512.py')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--blocks',type=int,default=4)
 p.add_argument('--rounds',type=int,default=2)
 p.add_argument('--out-dir',type=Path,required=True)
 a=p.parse_args()
 if a.blocks<2 or a.rounds<2:p.error('--blocks and --rounds must be >=2')
 if a.out_dir.exists():p.error(f'directory exists: {a.out_dir}; use a new --out-dir')
 a.out_dir.mkdir(parents=True)
 allrows=[];blocks=[]
 for b in range(a.blocks):
  path=a.out_dir/f'block_{b}.json'
  command=[sys.executable,str(SCRIPT),'--rounds',str(a.rounds),'--out',str(path)]
  print('START fresh process',b,flush=True)
  completed=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  print(completed.stdout,flush=True)
  if completed.returncode:
   raise SystemExit(f'block {b} failed with exit {completed.returncode}; prior results preserved')
  result=json.loads(path.read_text())
  rows=result['rows']
  for row in rows:row['block']=b
  allrows.extend(rows)
  selected=[r for r in rows if r['step']>=2]
  med={k:float(np.median([r['ms'][k] for r in selected])) for k in ('256','512')}
  blocks.append({'block':b,'median_ms':med,
   'relative_delta_percent':100*(med['512']/med['256']-1),
   'total_ms':{k:sum(r['ms'][k] for r in selected) for k in ('256','512')},
   'large_frames':[{ 'round':r['round'],'step':r['step'],'pose':r['pose'],
     'times_ms':r['ms']} for r in selected if max(r['ms'].values())>250]})
  print('BLOCK_RESULT',b,json.dumps(blocks[-1]),flush=True)
 stable=[r for r in allrows if r['step']>=2]
 deltas=np.array([r['ms']['512']-r['ms']['256'] for r in stable])
 non_growth=[r for r in stable if r['step'] not in (9,10)]
 stats={
  'blocks':blocks,'phase':104,'all_parity':'PASS per Phase103 block',
  'frame_pairs_including_growth':len(stable),
  'pooled_paired_median_delta_ms':float(np.median(deltas)),
  'pooled_pair_win_rate_512':float(np.mean(deltas<0)),
  'per_block_median_delta_pct_median':float(np.median([b['relative_delta_percent'] for b in blocks])),
  'non_growth_pairs':len(non_growth),
  'non_growth_paired_median_delta_ms':float(np.median([r['ms']['512']-r['ms']['256'] for r in non_growth])),
  'non_growth_pair_win_rate_512':float(np.mean([r['ms']['512']<r['ms']['256'] for r in non_growth])),
  'limitations':['Independent processes but same deterministic scene/camera sequence.',
    'Large growth/cold-start tails reported separately, not assumed causal.',
    'Phase103 serial-only scatter override; subprocesses run sequentially.',
    'No production code/default changes.']}
 (a.out_dir/'phase104_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
 print('RESULT phase104 PASS',json.dumps({k:v for k,v in stats.items() if k!='blocks'}),flush=True)

if __name__=='__main__':main()
