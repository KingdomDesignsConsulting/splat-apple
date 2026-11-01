#!/usr/bin/env python3
"""Phase 80: quantify cross-process cold radix timing vs idle interval.

Run ONLY the Phase 78 cold mode in fresh Python processes. Alternate no delay
and configured idle delay in counterbalanced order. Avoid clearing caches or
changing system settings. Record per-process first-sort and total durations.
A delay-dependent change would be suggestive, not proof, of system GPU cache
aging. A persistent fast state would favor a long-lived cross-process cache.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
SCRIPT=Path(__file__).with_name("bench_phase78_prewarm_controls.py")
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--rounds',type=int,default=4)
 p.add_argument('--idle-seconds',type=float,default=30.)
 p.add_argument('--out-dir',type=Path,default=Path("phase80_cache_persistence"))
 a=p.parse_args()
 if not 1<=a.rounds<=12 or not 0<=a.idle_seconds<=300:p.error('invalid rounds or delay')
 if a.out_dir.exists():p.error('directory exists; preserve old data')
 a.out_dir.mkdir(parents=True)
 rows=[]
 try:
  for block in range(a.rounds):
   delays=[0.,a.idle_seconds] if block%2==0 else [a.idle_seconds,0.]
   for pos,delay in enumerate(delays):
    if delay:time.sleep(delay)
    out=a.out_dir/f'block{block:02d}_pos{pos}_cold.json'
    log=out.with_suffix('.log')
    cmd=[sys.executable,str(SCRIPT),'--mode','cold','--out',str(out)]
    start=time.perf_counter()
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log.write_text(proc.stdout)
    if proc.returncode:raise RuntimeError(f'cold run failed ({proc.returncode}): {log}')
    d=json.loads(out.read_text())
    row={'block':block,'position':pos,'idle_before_s':delay,
         'first_ms':d['first_ms'],'total_ms':d['total_including_prep_ms'],
         'peak_memory':d['memory']['get_peak_memory'],
         'wall_s':time.perf_counter()-start}
    rows.append(row)
    print('PASS',row,flush=True)
 finally:
  (a.out_dir/'partial_or_complete.json').write_text(json.dumps(rows,indent=2)+'\n')
 stats={}
 for group in ('immediate','delayed'):
  arr=[r for r in rows if (r['idle_before_s']==0.)==(group=='immediate')]
  stats[group]={'n':len(arr),
      'first_median_ms':float(np.median([r['first_ms'] for r in arr])),
      'first_min_ms':min(r['first_ms'] for r in arr),
      'first_max_ms':max(r['first_ms'] for r in arr),
      'total_median_ms':float(np.median([r['total_ms'] for r in arr]))}
 report={'phase':80,'rounds':a.rounds,'idle_seconds':a.idle_seconds,
         'groups':stats,'rows':rows,
         'notes':['Only cold radix runs, each in a fresh Python process.',
                  'The first run can inherit machine state from earlier experiments.',
                  'Waiting does not clear GPU or shader caches; no causal proof.',
                  'No system cache deletion, no production renderer changes.']}
 (a.out_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase80',json.dumps(stats),flush=True)
if __name__=='__main__':main()
