"""CPU-only smoke and adversarial tests; NO MAC METAL EXECUTION IMPLIED."""
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# CI/container portability: phase138 helpers need only these shared CPU hooks.
common = types.ModuleType('qualification_common')
common.HERE = HERE
common.ROOT = HERE
common.command = lambda *a, **k: None
common.digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
common.invariants = lambda: {'fast_sha256':'f'*64,'adapter_sha256':'a'*64,
                             'defaults':{'native_fast':'mlx','one_shot_adapter':'auto',
                                         'persistent_adapter':'auto'}}
common.inventory = lambda: {'mlx':'0.32.4'}
common.metadata = lambda: {'python':'Python3.13-mock','platform':'macOS-26.7.1-arm64',
                           'utc':'fixed-utc'}
common.output_paths = lambda p: (Path(p), make_logs(p))
common.write_report = lambda p, r: Path(p).write_text(json.dumps(r))
common.gpu_gate = lambda: {'utilization_pct':0,'utc':'utc0'}
common.memory_pressure_gate = lambda:98
common.isolated_runtime = lambda *a, **k: None
sys.modules['qualification_common'] = common

import phase138_checkpoint as cp
import run_phase138_mode_isolated_memory as runner


def make_logs(out):
    out=Path(out)
    if out.exists():
        raise FileExistsError(out)
    d=out.parent/(out.stem+'_artifacts')
    d.mkdir(exist_ok=False)
    return d


class TestGPU(unittest.TestCase):
    def test_two_stable_low_samples_required(self):
        readings=[96,2,97,18,22]
        calls=[]; sleeps=[]
        def next_gate():
            value=readings.pop(0)
            calls.append(value)
            if value>30:
                raise RuntimeError(f'GPU test blocked: utilization {[value]}; limit 30%')
            return {'utilization_pct':value,'utc':f'sample-{len(calls)}'}
        report={'gpu_gate_attempts':[]}
        gate=cp.stable_gpu_gate(report,reader=next_gate,sleeper=sleeps.append,
                                attempts=7,cooldown_seconds=10,confirm_seconds=2,
                                settle_seconds=4)
        self.assertEqual(calls,[96,2,97,18,22])
        self.assertEqual(gate['utilization_pct'],22)
        self.assertEqual([18,22],[r['utilization_pct'] for r in gate['confirmations']])
        self.assertEqual(sleeps,[4,10,2,10,2])
        self.assertEqual(len(report['gpu_gate_attempts']),1)

    def test_fail_closed_if_no_stable_window(self):
        states=[96,2,98,1]
        def read():
            v=states.pop(0)
            if v>30:
                raise RuntimeError('GPU test blocked: utilization [96]; limit 30%')
            return {'utilization_pct':v, 'utc':'t'}
        r={'gpu_gate_attempts':[]}
        with self.assertRaisesRegex(RuntimeError,'remained unstable'):
            cp.stable_gpu_gate(r,reader=read,sleeper=lambda n:None,attempts=4)
        self.assertEqual(len(r['gpu_gate_attempts'][0]),4)

    def test_reader_over_threshold_fails(self):
        with self.assertRaisesRegex(RuntimeError,'did not establish'):
            cp.stable_gpu_gate({'gpu_gate_attempts':[]},reader=lambda:{'utilization_pct':96,'utc':'t'},
                               sleeper=lambda n:None)


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_integrity_and_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);label='small_0_auto'; inv=common.invariants()
            (p/(label+'.json')).write_text(json.dumps({'phase':138,'status':'PASS',
                'mode':'auto','workload':'small','block':0,'invariants':inv,
                'frames':[{'label':'first_frame','sha256_rgb':'1'*64}],
                'timeline':[{'label':'first_frame','mlx_peak_bytes':1}]}))
            (p/(label+'.log')).write_text('worker passed')
            gate={'utilization_pct':12,'utc':'t','attempts':[],
                  'confirmations':[{'status':'PASS','utilization_pct':8},
                                   {'status':'PASS','utilization_pct':12}]}
            entry={'workload':'small','block':0,'mode':'auto',
                   'free_memory_pct':90,'gpu_gate':gate}
            cp.save_checkpoint(p,label,entry,inv)
            self.assertIsNotNone(cp.completed_checkpoint(p,label,mode='auto',
                workload='small',block=0,invariants=inv))
            (p/(label+'.json')).write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'integrity mismatch'):
                cp.completed_checkpoint(p,label,mode='auto',workload='small',
                                        block=0,invariants=inv)

    def test_legacy_artifacts_cannot_resume(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(RuntimeError,'predates Phase138 checkpoints'):
                cp.validate_manifest(t,{'phase':138})

    def test_locked_source_cannot_resume(self):
        import fcntl
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);marker={'phase':138,'checkpoint_schema':2}
            cp.write_once(d/'phase138_manifest.json',marker)
            lock=(d/'phase138_active.lock').open('x')
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaisesRegex(RuntimeError,'still active'):
                    cp.validate_manifest(t,marker)
            finally:
                lock.close()


class TestOrchestrator(unittest.TestCase):
    def test_pass_and_resume_only_verified_workers(self):
        from types import SimpleNamespace
        @contextlib.contextmanager
        def isolated(*a,**k):
            yield '/mock/python',None,{}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'experimental_adaptive_sort.py').write_text('# source\n')
            (root/'phase138_checkpoint.py').write_bytes((HERE/'phase138_checkpoint.py').read_bytes())
            count={'workers':0,'gates':0}
            def gate(report,**kwargs):
                count['gates']+=1
                g={'utilization_pct':14,'utc':'gate-t',
                   'attempts':[{'status':'PASS','utilization_pct':10},
                               {'status':'PASS','utilization_pct':14}],
                   'confirmations':[{'status':'PASS','utilization_pct':10},
                                    {'status':'PASS','utilization_pct':14}]}
                report['gpu_gate_attempts'].append(g['attempts'])
                return g
            def command(argv, log=None,**kwargs):
                if '-m' in argv and 'check' in argv:
                    Path(log).write_text('pip check PASS')
                    return 0
                count['workers']+=1
                get=lambda arg:argv[argv.index(arg)+1]
                workload=get('--workload');block=int(get('--block'));mode=get('--mode')
                out=Path(get('--out'))
                data={'phase':138,'status':'PASS','mode':mode,
                      'workload':workload,'block':block,
                      'invariants':common.invariants(),
                      'frames':[{'label':label,'sha256_rgb':'1'*64,
                                 'elapsed_ms':3.0} for label in
                                ['first_frame','warm_0','warm_1','warm_2','warm_3']],
                      'timeline':[{'label':k,'mlx_peak_bytes':100,
                                   'mlx_cache_bytes':50,'process_peak_rss_bytes':1000}
                                  for k in ['first_frame','after_release']]}
                out.write_text(json.dumps(data))
                Path(log).write_text('completed PASS')
                return 0
            with patch.object(runner,'HERE',root),\
                 patch.object(runner,'isolated_runtime',isolated),\
                 patch.object(runner,'command',command),\
                 patch.object(runner,'stable_gpu_gate',gate),\
                 patch.object(runner,'memory_pressure_gate',lambda:98):
                args=SimpleNamespace(out=root/'phase138_1.json',workload=['small'],
                                     blocks=1,attest_m4=True,resume_from=None)
                self.assertEqual(runner.run_parent(args),0)
                first=json.loads(args.out.read_text())
                self.assertEqual(first['completed_workers'],5)
                self.assertEqual(count['workers'],5)
                self.assertEqual(len(list((root/'phase138_1_artifacts').glob('*.checkpoint.json'))),5)
                args=SimpleNamespace(out=root/'phase138_2.json',workload=['small'],
                                     blocks=1,attest_m4=True,
                                     resume_from=root/'phase138_1_artifacts')
                self.assertEqual(runner.run_parent(args),0)
                result=json.loads(args.out.read_text())
                self.assertEqual(result['completed_workers'],5)
                self.assertEqual(count['workers'],5) # no GPU child rerun
                self.assertEqual(count['gates'],5)
                self.assertTrue(all(x.get('reused_from') for x in result['workers']))
                self.assertEqual(len(list((root/'phase138_2_artifacts').glob('*.checkpoint.json'))),5)


    def test_partial_failure_resumes_without_rerunning_good_workers(self):
        from types import SimpleNamespace
        @contextlib.contextmanager
        def isolated(*a, **k):
            yield '/mock/python', None, {}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'experimental_adaptive_sort.py').write_text('# source\n')
            (root/'phase138_checkpoint.py').write_bytes((HERE/'phase138_checkpoint.py').read_bytes())
            seen=[]
            fail_once={'enabled':True}
            def gate(report, **kwargs):
                g={'utilization_pct':14,'utc':'t',
                   'attempts':[{'status':'PASS','utilization_pct':9},
                               {'status':'PASS','utilization_pct':14}],
                   'confirmations':[{'status':'PASS','utilization_pct':9},
                                    {'status':'PASS','utilization_pct':14}]}
                report['gpu_gate_attempts'].append(g['attempts'])
                return g
            def command(argv, log=None, **kwargs):
                if '-m' in argv and 'check' in argv:
                    Path(log).write_text('pip check ok')
                    return 0
                get=lambda a:argv[argv.index(a)+1]
                w,b,m=get('--workload'),int(get('--block')),get('--mode')
                seen.append(m)
                if m=='scatter512' and fail_once['enabled']:
                    fail_once['enabled']=False
                    Path(log).write_text('simulated failure')
                    return 124
                frames=[{'label':x,'sha256_rgb':'2'*64,'elapsed_ms':1.0}
                        for x in ['first_frame','warm_0','warm_1','warm_2','warm_3']]
                timeline=[{'label':x,'mlx_peak_bytes':1,'mlx_cache_bytes':1,
                           'process_peak_rss_bytes':1} for x in
                          ('first_frame','after_release')]
                Path(get('--out')).write_text(json.dumps({
                    'phase':138,'status':'PASS','mode':m,'workload':w,
                    'block':b,'invariants':common.invariants(),
                    'frames':frames,'timeline':timeline}))
                Path(log).write_text('simulated pass')
                return 0
            with patch.object(runner,'HERE',root),\
                 patch.object(runner,'isolated_runtime',isolated),\
                 patch.object(runner,'command',command),\
                 patch.object(runner,'stable_gpu_gate',gate),\
                 patch.object(runner,'memory_pressure_gate',lambda:98):
                args=SimpleNamespace(out=root/'phase138_partial.json',
                                     workload=['small'],blocks=1,attest_m4=True,
                                     resume_from=None)
                self.assertEqual(runner.run_parent(args),1)
                self.assertEqual(seen, ['auto','selective_final3','scatter512'])
                source=root/'phase138_partial_artifacts'
                self.assertEqual(len(list(source.glob('*.checkpoint.json'))),2)
                args=SimpleNamespace(out=root/'phase138_resumed.json',
                                     workload=['small'],blocks=1,attest_m4=True,
                                     resume_from=source)
                self.assertEqual(runner.run_parent(args),0)
                self.assertEqual(seen, ['auto','selective_final3','scatter512',
                                        'scatter512','active','adaptive'])
                report=json.loads(args.out.read_text())
                self.assertEqual(report['completed_workers'],5)
                self.assertEqual(sum(bool(x.get('reused_from')) for x in report['workers']),2)


if __name__=='__main__':
    unittest.main()
