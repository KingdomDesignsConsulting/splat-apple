#!/usr/bin/env python3
"""Phase111: consolidated opt-in scatter512 acceptance gate.

Runs actual renderer fallback (Phase110), then multiple independent integrated
renderer parity/timing processes (Phase109), in one output directory. Failure
at any step fails gate. All intermediate evidence retained; no default edits.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def run(script,args,logfile):
 cmd=[sys.executable,str(HERE/script),*args]
 print('RUN', ' '.join(cmd),flush=True)
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 logfile.write_text(p.stdout)
 print(p.stdout,flush=True)
 if p.returncode:raise RuntimeError(f'{script} exited {p.returncode}; evidence at {logfile}')
 return p

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out-dir',type=Path,required=True)
 p.add_argument('--blocks',type=int,default=3)
 a=p.parse_args()
 if a.out_dir.exists():p.error(f'output directory exists: {a.out_dir}')
 if a.blocks<2:p.error('--blocks must be >= 2')
 a.out_dir.mkdir(parents=True)
 report={'phase':111,'status':'INCOMPLETE','blocks_requested':a.blocks,
         'acceptance_thresholds':['Phase110 real fallback must PASS',
          'Phase109 exact renderer RGB/alpha parity must PASS in all processes',
          'Every warmed block median must favor 512 over 256',
          'Median of warmed block relative changes must be below zero']}
 summary=a.out_dir/'phase111_summary.json'
 try:
  fallback=a.out_dir/'phase110_fallback.json'
  run('bench_phase110_renderer_fallback.py',['--out',str(fallback)],a.out_dir/'phase110_stdout.txt')
  fb=json.loads(fallback.read_text())
  if fb['parity']!='PASS' or not all(row['calls']['radix4']==1 and
         row['calls']['final3']==0 and
         all(tuple(g)==(256,1,1) for g in row['scatter_groups'])
         for row in fb['cases']):
   raise AssertionError('renderer-level fallback gate failed')
  report['fallback']='PASS'
  subdir=a.out_dir/'phase109'
  run('bench_phase109_integrated_acceptance.py',
   ['--blocks',str(a.blocks),'--out-dir',str(subdir)],
   a.out_dir/'phase109_stdout.txt')
  data=json.loads((subdir/'phase109_summary.json').read_text())
  if data['all_parity']!='PASS' or data['integrated_block_win_count']!=a.blocks:
   raise AssertionError('integrated renderer parity or process win gate failed')
  for x in data['integrated']:
   if x['source_guard']!='PASS _renderer_class instantiated':
    raise AssertionError('source integrity guard failed')
  report['integrated']='PASS'
  report['per_block_warmed_percent_512']=[x['warmed_relative_percent'] for x in data['integrated']]
  report['median_block_percent_512']=float(np.median(report['per_block_warmed_percent_512']))
  if report['median_block_percent_512']>=0:raise AssertionError('median improvement gate failed')
  report['status']='PASS'
 except Exception as exc:
  report['status']='FAIL'
  report['error']=str(exc)
  summary.write_text(json.dumps(report,indent=2)+'\n')
  raise
 summary.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase111 PASS',json.dumps(report),flush=True)

if __name__=='__main__':main()
