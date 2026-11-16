#!/usr/bin/env python3
"""CPU-only, fail-closed aggregation of retained qualification evidence.

Exit 1 means NOT READY, including incomplete/missing/stale evidence. This is
not a GPU test and cannot manufacture readiness from committed test scripts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from qualification_common import HERE, ROOT, ADAPTER, invariants


def aggregate(evidence, performance):
    rows=[]
    def check(name, action):
        try:
            detail=action()
            rows.append({'name':name,'status':'PASS','detail':detail})
        except Exception as exc:
            rows.append({'name':name,'status':'FAIL','error':f'{type(exc).__name__}: {exc}'})
    def load(name):
        p=evidence/name
        d=json.loads(p.read_text())
        assert d['status']=='PASS', f'{name}: status={d.get("status")}'
        return d
    def all_pass(rows,count):
        assert len(rows)==count, f'expected {count} checks; got {len(rows)}'
        assert all((r.get('status')=='PASS') if 'status' in r else
                   (r.get('returncode')==0 and 'PASS' in r.get('stdout','')) for r in rows)
    current=invariants()
    def installation():
        d=load('phase121/phase121_summary.json')
        assert len(d['steps'])==7 and all(s['returncode']==0 for s in d['steps'])
        assert d['wheel_sha256']=='bdfdffe97d2a933afb6bdac2c01ae609c119bf1e22fe129469f4ab03a59439bf'
        assert d['wheel_fast_sha256']=='4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc'
        for s in d['steps']:assert (evidence/'phase121'/s['log']).stat().st_size>0
        gate=load('phase121/api_gate.json');all_pass(gate['checks'],5)
        inventory=(evidence/'phase121/package_inventory.log').read_text()
        assert 'mlx==0.32.4' in inventory and 'mlx-metal==0.32.4' in inventory
        return 'seven stages + five real API checks; actual final MLX/Metal 0.32.4'
    def legacy():
        d=load('phase122/phase122_legacy_isolation_3.json');all_pass(d['checks'],3)
        assert d['native_build']['returncode']==0
        assert {r['backend'] for r in d['checks']}=={'python','cpp','missing_opt_in'}
        assert all(r['mlx3d_absent'] and r['returncode']==0 for r in d['checks'])
        assert d['caller_environment_unchanged']
        return 'actual Python and built CPP renders without MLX3D; missing opt-in isolated'
    def lifecycle():
        d=load('phase123/phase123_source_lifecycle_2.json');all_pass(d['checks'],10)
        latest=load('phase124/phase124_loader_regression_1.json');all_pass(latest['checks'],10)
        assert latest['invariants']==current, 'loader evidence stale for current adapter'
        return '10 lifecycle cases; repeated on current adapter'
    def api():
        d=load('phase124/phase124_api_contract_2.json')
        all_pass(d['contracts']['checks'],65);all_pass(d['api_gate']['checks'],5)
        assert d['contracts_returncode']==d['api_gate_returncode']==0
        assert d['invariants']==current and d['caller_environment_unchanged']
        return '65 contracts and five-check real gate; adapter digest matches current'
    def perf():
        d=load(performance)
        assert d['invariants']==current and d['caller_environment_unchanged']
        assert len(d['boundaries']['cases'])==6 and d['boundaries']['status']=='PASS'
        assert {w['name'] for w in d['workloads']}=={'small','fullhd','highres'}
        for w in d['workloads']:
            assert len(w['blocks'])==4
            assert {b['block'] for b in w['blocks']}==set(range(4))
            for b in w['blocks']:
                assert b['status']=='PASS' and len(b['warm_pairs'])==16
        assert len(d['gpu_gates'])==13 and all(g['utilization_pct']<=30 for g in d['gpu_gates'])
        return 'six boundaries and twelve complete fresh-process benchmark blocks'
    def review():
        s=json.loads((evidence/'phase126/github_review.json').read_text())
        assert s['draft'] and not s['merged']
        assert not s['inline_review_threads'], 'unresolved/review threads require explicit disposition'
        assert s['head_sha']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() or subprocess.run(['git','merge-base','--is-ancestor',s['head_sha'],'HEAD'],cwd=ROOT).returncode==0
        assert not subprocess.check_output(['git','diff','cb459ac','HEAD','--','mlx_gs/renderer/renderer.py',str(HERE.relative_to(ROOT)/'mlx3d/splatting/fast.py')],cwd=ROOT,text=True)
        return {'reviewed_head':s['head_sha'],'ci':'NO CI PASS ESTABLISHED','scope':s['review_scope']}
    check('phase121_installation',installation)
    check('phase122_real_legacy_isolation',legacy)
    check('phase123_loader_lifecycle',lifecycle)
    check('phase124_api_contracts',api)
    check('phase125_complete_performance',perf)
    check('source_defaults_and_review',review)
    failed=[r['name'] for r in rows if r['status']!='PASS']
    return {'phase':126,'status':'FAIL' if failed else 'PASS',
        'decision':'NOT READY' if failed else 'CONDITIONALLY READY',
        'utc':datetime.now(timezone.utc).isoformat(),'checks':rows,'blockers':failed,
        'invariants':current,'performance_evidence':performance,
        'limits':['Mac mini M4 / macOS 26.7.1 / Python 3.13 only; other devices unqualified',
                  'No independent external approval or CI pass established',
                  'Forward-only; immutable deployment/load-time source guard; no concurrent sharing',
                  'Process RSS is not per-mode allocation; no broad speedup or default promotion'],
        'evidence_manifest':{str(p.relative_to(evidence)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(evidence.rglob('*')) if p.is_file()},
        'policy':'Draft/unmerged irrespective of decision. Missing/incomplete required evidence fails closed.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evidence',type=Path,default=HERE/'qualification_evidence')
    p.add_argument('--performance',default='phase125/phase125_integration_performance_2.json')
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            r=aggregate(Path(temp),a.performance)
            assert r['decision']=='NOT READY' and len(r['blockers'])==6
        print('PASS missing evidence fails closed')
        return
    r=aggregate(a.evidence,a.performance)
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print('RESULT PHASE126',r['decision'],a.out)
    for c in r['checks']:print(c['name'],c['status'],c.get('error',c.get('detail')))
    raise SystemExit(0 if r['status']=='PASS' else 1)


if __name__=='__main__':main()
