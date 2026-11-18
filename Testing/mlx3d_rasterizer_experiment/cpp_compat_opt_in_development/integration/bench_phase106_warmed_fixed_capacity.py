#!/usr/bin/env python3
"""Phase106: warmed fixed-capacity renderer A/B in independent processes.

Reuses the Phase103 exact RGB+alpha parity benchmark without changing renderer
sources. Each fresh process runs three identical camera-sequence rounds: round0
grows sorter capacity; round1 and round2 retain the maximum 53M slot workspace.
Report ONLY rounds1/2 and step>=2 for steady-state decisions. Seed is varied
between processes to randomize which variant executes first on ordinary frames.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
SCRIPT=Path(__file__).with_name('bench_phase103_renderer_scatter512.py')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--blocks',type=int,default=4)
 parser.add_argument('--out-dir',type=Path,required=True)
 args=parser.parse_args()
 if args.blocks<3:parser.error('--blocks must be >=3')
 if args.out_dir.exists():parser.error(f'output directory exists: {args.out_dir}')
 args.out_dir.mkdir(parents=True)
 rows=[];per_block=[]
 for block in range(args.blocks):
  output=args.out_dir/f'block_{block}.json'
  cmd=[sys.executable,str(SCRIPT),'--rounds','3','--seed',str(10600+block),
       '--out',str(output)]
  print('START fresh process',block,flush=True)
  done=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  print(done.stdout,flush=True)
  if done.returncode:raise SystemExit(f'process {block} failed; previous JSON retained')
  info=json.loads(output.read_text())
  chosen=[r for r in info['rows'] if r['round']>=1 and r['step']>=2]
  if len(chosen)!=26:raise AssertionError(('unexpected frames',block,len(chosen)))
  capacities=[int(r['counts']['sorted']) for r in chosen]
  if len(set(capacities))!=1:
   raise AssertionError(('capacity changed in warmed frames',block,set(capacities)))
  if block and capacities[0]!=per_block[0]['capacity']:
   raise AssertionError(('inconsistent capacity across blocks',block,capacities[0]))
  diffs=[r['ms']['512']-r['ms']['256'] for r in chosen]
  median256=float(np.median([r['ms']['256'] for r in chosen]))
  median512=float(np.median([r['ms']['512'] for r in chosen]))
  out={'block':block,'capacity':capacities[0],
       'median_ms_256':median256,'median_ms_512':median512,
       'median_pct_512':100*(median512/median256-1),
       'median_paired_delta_ms':float(np.median(diffs)),
       'win_rate_512':float(np.mean(np.array(diffs)<0)),
       'paired_deltas_ms':diffs}
  per_block.append(out)
  rows.extend([{'block':block,'round':r['round'],'step':r['step'],
                'pose':r['pose'],'ms':r['ms']} for r in chosen])
  print('BLOCK_RESULT',json.dumps({k:v for k,v in out.items() if k!='paired_deltas_ms'}),flush=True)
 deltas=[r['ms']['512']-r['ms']['256'] for r in rows]
 summary={'phase':106,'all_parity':'PASS from Phase103 per block',
  'processes':args.blocks,'paired_warmed_frames':len(rows),
  'capacity':per_block[0]['capacity'],
  'median_of_block_median_percent':float(np.median([r['median_pct_512'] for r in per_block])),
  'block_win_count_512':sum(r['median_pct_512']<0 for r in per_block),
  'pooled_paired_median_delta_ms':float(np.median(deltas)),
  'pooled_win_rate_512':float(np.mean(np.array(deltas)<0)),
  'per_block':per_block,
  'method':['No capacity changes in measured rounds 1 and 2.',
            'Rounds 0 and first two frames of each subsequent round excluded.',
            'Identical deterministic camera sequence, varying random execution-order seed.',
            'Serial-only experimental scatter hook inherited from Phase103.',
            'No production renderer edits or global default changes.']}
 (args.out_dir/'phase106_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('RESULT phase106 PASS',json.dumps({k:v for k,v in summary.items() if k!='per_block'}),flush=True)
if __name__=='__main__':main()
