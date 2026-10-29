#!/usr/bin/env python3
"""Phase 42: bounded staged-CPP-compat regression matrix (no production edits).

Run from splat-apple repository root after conda activate gs-mps.
Each case launches the existing Phase 41 tested harness in a fresh process, with a
115 second deadline. Results/logs are written to a fresh timestamped directory.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('Testing/mlx3d_rasterizer_experiment')
SCRIPT = ROOT/'cpp_compat_opt_in_development'/'bench_phase41_staged_regression.py'
BASE = ROOT/'cpp_compat_opt_in_development'
PLY = Path('results/bonsai_mlx_cpp_20261008_200230/ply/bonsai_final.ply')
# Highest precision (t_min=0.0001): 4 independent camera/density acceptance targets
HIGH_ACCURACY = {(72,400000): 0.00007187, (72,700000): 0.00009528,
                 (0,400000): 0.00006807, (0,700000): 0.00006873}
# Established threshold 1/255: 14 scenario-specific validation targets.
STANDARD = {
 ('displaced',72,400000):.001964,('displaced',72,700000):.002094,
 ('displaced',0,400000):.002010,('displaced',0,700000):.002089,
 ('trained',72,206613):.001845,('trained',0,206613):.001937,
 ('normal',72,400000):.001972,('normal',72,700000):.002171,
 ('normal',0,400000):.002025,('normal',0,700000):.002140,
 ('displaced',72,1600000):.002470,('displaced',0,1600000):.002407,
 ('normal',72,1600000):.002736,('normal',0,1600000):.002558,
}

def pick_cases(group):
    if group == 'camera0':
        return [('displaced',0,n,0.0001,HIGH_ACCURACY[(0,n)]) for n in (400000,700000)]
    if group == 'high-accuracy':
        return [('displaced',cam,n,0.0001,expected) for (cam,n),expected in HIGH_ACCURACY.items()]
    if group == 'standard':
        return [(kind,cam,n,1/255,expected) for (kind,cam,n),expected in STANDARD.items()]
    raise ValueError(group)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group',choices=['camera0','high-accuracy','standard'],default='camera0')
    ap.add_argument('--per-case-timeout',type=int,default=115)
    ap.add_argument('--runs',type=int,default=3)
    ap.add_argument('--tolerance',type=float,default=.000025,
                    help='Absolute RMSE allowance above historical target for platform variance')
    ap.add_argument('--out-root',type=Path)
    ap.add_argument('--ply',type=Path,default=PLY)
    a=ap.parse_args()
    if not SCRIPT.is_file():ap.error(f'missing Phase41 harness: {SCRIPT}')
    if not a.ply.is_file():ap.error(f'missing trained PLY: {a.ply}')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    outroot=a.out_root or BASE/f'phase42_{a.group.replace("-","_")}_{stamp}'
    if outroot.exists():ap.error(f'output directory exists: {outroot}')
    outroot.mkdir(parents=True)
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join((str(ROOT.resolve()),str(Path.cwd().resolve()),env.get('PYTHONPATH','')))
    summary={'phase':42,'group':a.group,'timeout_sec':a.per_case_timeout,'cases':[]}
    failures=[]
    for kind,cam,n,tmin,expected in pick_cases(a.group):
        name=f'{kind}_camera{cam}_n{n}'
        out=outroot/name
        args=[sys.executable,'-u',str(SCRIPT),'--ply',str(a.ply),
              '--camera',str(cam),'--counts',str(n),'--population-max',str(max(800000,n)),
              '--normalized-scene','--displacement',str(2 if kind=='displaced' else 0),
              '--seed','12345','--offset','0.5','--variants','staged_cpp_compat',
              '--t-min',str(tmin),'--runs',str(a.runs),'--color','--output-dir',str(out)]
        if kind=='trained':args+=['--trained']
        print('RUN',name,'target',expected,flush=True)
        start=time.monotonic()
        try:
            done=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,timeout=a.per_case_timeout,env=env)
            code=done.returncode;log=done.stdout
        except subprocess.TimeoutExpired as ex:
            code=124;log=(ex.stdout or b'').decode(errors='replace') if isinstance(ex.stdout,bytes) else (ex.stdout or '')
            log+='\nTIMEOUT: process terminated after per-case deadline\n'
        elapsed=time.monotonic()-start
        (outroot/f'{name}.log').write_text(log)
        item={'name':name,'scene':kind,'camera':cam,'n':n,'t_min':tmin,
              'expected_rmse':expected,'elapsed_s':round(elapsed,3),'returncode':code,
              'command':args,'log':f'{name}.log'}
        resultpath=out/'results.json'
        if code==0 and resultpath.is_file():
            raw=json.loads(resultpath.read_text())
            actual=raw['cases'][0]['results'][0]['global_rmse']
            item['actual_rmse']=actual
            item['pass']=bool(actual<=expected+a.tolerance)
        else:item['pass']=False
        summary['cases'].append(item)
        (outroot/'phase42_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if not item['pass']:failures.append(name)
        print('PASS' if item['pass'] else 'FAIL',name,
              'rmse=',item.get('actual_rmse'),'seconds=',round(elapsed,2),flush=True)
        if not item['pass']:
            print('Log:',outroot/f'{name}.log',flush=True)
            # Stop early to avoid using later tests on a broken staged integration.
            break
    print('Summary:',outroot/'phase42_summary.json',flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
