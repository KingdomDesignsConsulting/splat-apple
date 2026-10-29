#!/usr/bin/env python3
"""Phase 79: balanced independent-process Phase 78 replication.

Run every mode once per block in cyclic counterbalanced order; each trial is
a fresh Python process. Never overwrite earlier outputs. Collect first-sort,
preparation, total, and allocator peak/cache into a summary. This controls
process-local MLX caches; machine-wide load and thermals are still confounders.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
SCRIPT=Path(__file__).with_name("bench_phase78_prewarm_controls.py")
MODES=("cold","generic","near","exact")
ORDERS=[MODES,MODES[1:]+MODES[:1],MODES[2:]+MODES[:2],MODES[3:]+MODES[:3]]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--blocks',type=int,default=4)
 p.add_argument('--out-dir',type=Path,default=Path('phase79_balanced_trials'))
 a=p.parse_args()
 if a.blocks<1 or a.blocks>24:p.error('blocks must be 1..24')
 if a.out_dir.exists():p.error('output directory already exists')
 a.out_dir.mkdir(parents=True)
 records=[]
 try:
  for block in range(a.blocks):
   for position,mode in enumerate(ORDERS[block%4]):
    target=a.out_dir/f'block{block:02d}_pos{position}_{mode}.json'
    log=a.out_dir/f'block{block:02d}_pos{position}_{mode}.log'
    command=[sys.executable,str(SCRIPT),'--mode',mode,'--out',str(target)]
    t=time.perf_counter()
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
    log.write_text(result.stdout)
    if result.returncode:
     raise RuntimeError(f'{mode} block={block} exited {result.returncode}; see {log}')
    d=json.loads(target.read_text())
    row={'block':block,'position':position,'mode':mode,'first_ms':d['first_ms'],
         'prep_ms':d['prep_ms'],'total_ms':d['total_including_prep_ms'],
         'peak_memory':d['memory']['get_peak_memory'],
         'cache_memory':d['memory']['get_cache_memory'],
         'elapsed_wall_s':time.perf_counter()-t}
    records.append(row)
    print('PASS',row,flush=True)
 finally:
  (a.out_dir/'partial_or_complete_results.json').write_text(json.dumps(records,indent=2)+'\n')
 if len(records)!=4*a.blocks:raise AssertionError('missing trial')
 stats={}
 for mode in MODES:
  rows=[x for x in records if x['mode']==mode]
  stats[mode]={field:{'median':float(np.median([r[field] for r in rows])),
                       'min':float(min(r[field] for r in rows)),
                       'max':float(max(r[field] for r in rows))}
               for field in ('first_ms','prep_ms','total_ms','peak_memory','cache_memory')}
 report={'phase':79,'blocks':a.blocks,'orders':[list(ORDERS[i%4]) for i in range(a.blocks)],
         'summary':stats,'trials':records,
         'notes':['New independent Python process per trial.',
                  'Cyclic ordering balances position after four blocks.',
                  'Single shared Mac GPU and system shader cache still persist across processes.',
                  'Compare medians and block-matched results; avoid universal cause attribution.',
                  'Run with no other GPU-heavy workload; preserves individual logs and JSON.']}
 (a.out_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase79',json.dumps({m:{k:v['median'] for k,v in stats[m].items()} for m in MODES}),flush=True)
if __name__=='__main__':main()
