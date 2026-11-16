#!/usr/bin/env python3
"""Phase122: actual python/cpp rendering in a disposable no-MLX3D runtime."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from qualification_common import (ROOT,HERE,command,clean_env,invariants,inventory,
    isolated_runtime,output_paths,write_report,gpu_gate,metadata)


def worker(backend,repo,out):
    sys.path=[str(repo)]+[p for p in sys.path if Path(p).resolve() not in (HERE.resolve(),ROOT.resolve())]
    import mlx.core as mx
    import numpy as np
    from types import SimpleNamespace
    import mlx_gs.renderer.renderer as module
    assert Path(module.__file__).resolve().is_relative_to(repo.resolve())
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert importlib.util.find_spec('mlx3d') is None
    n=8
    params={'means':mx.array([[((i%4)-1.5)*0.08,((i//4)-0.5)*0.1,3+i*0.1] for i in range(n)],dtype=mx.float32),
            'scales':mx.full((n,3),-3.0),'quaternions':mx.array([[1.,0.,0.,0.]]*n),
            'opacities':mx.zeros((n,1)),'sh_coeffs':mx.zeros((n,1,3))}
    cam=SimpleNamespace(W=48,H=32,fx=40.,fy=40.,cx=24.,cy=16.,W2C=mx.eye(4))
    result={'backend':backend,'status':'FAIL','mlx3d_absent':True,'renderer_module':str(module.__file__),'venv_prefix':sys.prefix,'base_python_prefix':sys.base_prefix,'mlx_runtime':str(mx.__file__)}
    try:
        def render(which=None):
            image=(module.render(params,cam) if which is None else module.render(params,cam,rasterizer_type=which))
            mx.eval(image)
            a=np.asarray(image)
            assert a.shape==(32,48,3) and np.isfinite(a).all() and np.max(a)>0
            return a.copy()
        if backend=='missing_opt_in':
            before=render('python')
            try: render('mlx3d_cpp_compat')
            except ModuleNotFoundError as exc:
                assert exc.name=='mlx3d' or exc.name.startswith('mlx3d.')
                result['dependency_error']={'type':type(exc).__name__,'name':exc.name,'message':str(exc)}
            else: raise AssertionError('Opt-in accepted absent mlx3d')
            assert np.array_equal(before.view(np.uint32),render('python').view(np.uint32))
        else:
            if backend=='cpp':
                assert module.rasterizer_metal is not None and module.rasterizer_metal.rasterizer_metal is not None
                native=module.rasterizer_metal.rasterizer_metal
                assert Path(native.__file__).resolve().is_relative_to(repo.resolve())
                result['native_extension']=str(native.__file__)
            first=render(backend); second=render(backend)
            assert np.array_equal(first.view(np.uint32),second.view(np.uint32))
            if backend=='python': assert np.array_equal(first.view(np.uint32),render().view(np.uint32))
            result['shape']=list(first.shape); result['finite']=True; result['repeat_bitwise']=True
        assert importlib.util.find_spec('mlx3d') is None
        assert not any(name=='mlx3d' or name.startswith('mlx3d.') for name in sys.modules)
        result['status']='PASS'
    except Exception as exc:
        result['error']={'type':type(exc).__name__,'message':str(exc)}
    write_report(out,result)
    print('RESULT worker',backend,result['status'],flush=True)
    return result['status']=='PASS'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--worker',choices=['python','cpp','missing_opt_in'])
    p.add_argument('--repo',type=Path)
    a=p.parse_args()
    if a.worker:
        if not a.repo or not a.out: p.error('worker requires repo and out')
        return 0 if worker(a.worker,a.repo,a.out) else 1
    if not a.out: p.error('out required')
    out,logs=output_paths(a.out)
    report={'phase':122,'status':'FAIL','environment':metadata(),'checks':[], 'gpu_gates':[]}
    original=invariants(); packages=inventory()
    try:
        with isolated_runtime(logs,native_build=True) as (py,tmp,env):
            repo=tmp/'repo'; repo.mkdir()
            # Archive bytes must not pass through text decoding.
            import subprocess,shlex
            args=['git','archive','HEAD','--','mlx_gs','setup_mlx.py']
            print('+ '+shlex.join(args)+' [binary source archive]',flush=True)
            data=subprocess.check_output(args,cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(data)) as archive: archive.extractall(repo,filter='data')
            build_rc=command([py,'setup_mlx.py','build_ext','--inplace'],logs/'build_cpp.log',cwd=repo,env=env)
            report['native_build']={'returncode':build_rc,'log':'build_cpp.log'}
            for name in ('python','cpp','missing_opt_in'):
                if name=='cpp' and build_rc:
                    report['checks'].append({'backend':'cpp','status':'UNAVAILABLE',
                        'reason':'isolated native build failed; this is not a CPP test PASS'})
                    continue
                report['gpu_gates'].append(gpu_gate())
                target=logs/(name+'.json')
                rc=command([py,Path(__file__).resolve(),'--worker',name,'--repo',repo,'--out',target],
                           logs/(name+'.log'),cwd=repo,env=env)
                item=json.loads(target.read_text()) if target.exists() else {'backend':name,'status':'FAIL','reason':'worker produced no report'}
                item['returncode']=rc
                report['checks'].append(item)
                print(item['status'],name,flush=True)
            command([py,'-m','pip','freeze'],logs/'package_inventory.log',env=env)
        if len(report['checks'])==3 and all(x['status']=='PASS' and x.get('returncode')==0 for x in report['checks']):
            report['status']='PASS'
    except Exception as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc)}
    report['invariants']=invariants()
    assert original==report['invariants'] and packages==inventory()
    report['caller_environment_unchanged']=True
    report['logs']=str(logs)
    write_report(out,report)
    print('RESULT PHASE122',report['status'],out,flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
