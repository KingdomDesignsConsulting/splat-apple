#!/usr/bin/env python3
"""Phase124: real supported contracts, safe negative boundaries, update recovery."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from qualification_common import (ROOT,HERE,ADAPTER,command,invariants,inventory,isolated_runtime,
    output_paths,write_report,gpu_gate,metadata,digest)


def worker(out,reference):
    sys.path.insert(0,str(ROOT))
    import mlx.core as mx
    import numpy as np
    from types import SimpleNamespace
    from unittest.mock import patch
    import mlx_gs.renderer.mlx3d_cpp_compat as api
    installed=api.importlib.import_module('mlx3d.splatting.fast')
    assert Path(installed.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert digest(installed.__file__)==api._ORIGINAL_INSTALLED_SHA
    spec=importlib.util.spec_from_file_location('_phase124_reference_adapter',reference)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.__file__=str(ADAPTER)
    old._MODULE_NAME='mlx3d.splatting._phase124_reference_fast'
    n=16
    params={'means':mx.array([[((i%4)-1.5)*0.1,((i//4)-1.5)*0.1,3+i*.03] for i in range(n)],dtype=mx.float32),
            'scales':mx.full((n,3),-3.3),'quaternions':mx.array([[1.,0.,0.,0.]]*n),
            'opacities':mx.zeros((n,1)),'sh_coeffs':mx.zeros((n,1,3))}
    cam=SimpleNamespace(W=65,H=49,fx=55.,fy=55.,cx=32.5,cy=24.5,W2C=mx.eye(4))
    rows=[]
    def add(name,fn,kind):
        try:details=fn();row={'name':name,'kind':kind,'status':'PASS','details':details}
        except Exception as exc:row={'name':name,'kind':kind,'status':'FAIL','error':{'type':type(exc).__name__,'message':str(exc)}}
        rows.append(row);print(row['status'],name,flush=True)
    def bits(image):
        mx.eval(image);a=np.asarray(image)
        assert a.shape==(cam.H,cam.W,3) and np.isfinite(a).all()
        return a.copy().view(np.uint32)
    def supported(name,p,bg=None):
        def run():
            ref=bits(old.render_cpp_compat(p,cam,background=bg))
            assert np.array_equal(bits(api.render_cpp_compat(p,cam,background=bg)),ref)
            e=api.PersistentCppCompatRenderer(p)
            assert np.array_equal(bits(e.render(cam,background=bg)),ref)
            assert np.array_equal(bits(e.render(cam,background=bg)),ref)
            return {'one_shot_persistent_reference_bitwise':True}
        add(name,run,'supported')
    supported('float32_dict',params)
    supported('attribute_object',SimpleNamespace(**params))
    supported('float16_scene',{k:v.astype(mx.float16) for k,v in params.items()})
    supported('integer_means',{**params,'means':params['means'].astype(mx.int32)})
    supported('vector_opacity',{**params,'opacities':params['opacities'].reshape((-1,))})
    supported('higher_sh_ignored',{**params,'sh_coeffs':mx.concatenate([params['sh_coeffs'],mx.ones((n,3,3))],axis=1)})
    supported('zero_quaternions',{**params,'quaternions':mx.zeros((n,4))})
    supported('transparent_logits',{**params,'opacities':mx.full((n,1),-float('inf'))})
    supported('zero_activated_scales',{**params,'scales':mx.full((n,3),-float('inf'))})
    supported('empty_scene',{k:v[:0] for k,v in params.items()})
    supported('background_vector',params,mx.array([-.1,.2,1.2]))
    supported('background_row',params,mx.array([[-.1,.2,1.2]]))
    supported('background_column',params,mx.array([[-.1],[.2],[1.2]]))

    class UnsafeNativeReached(RuntimeError):pass
    def blocked(*args,**kwargs):raise UnsafeNativeReached('malformed input reached native boundary')
    class BlockedRenderer:
        def __init__(self,*args,**kwargs):blocked()
    allowed=(ValueError,TypeError,KeyError,AttributeError,IndexError)
    def expect_rejected(fn):
        try:fn()
        except UnsafeNativeReached:raise
        except allowed as exc:
            assert str(exc)
            return {'exception':type(exc).__name__,'message':str(exc)}
        raise AssertionError('Unsupported request was accepted')
    scenes={
        'missing_means':{k:v for k,v in params.items() if k!='means'},
        'none_field':{**params,'scales':None},
        'wrong_mean_columns':{**params,'means':mx.zeros((n,2))},
        'wrong_scales_rows':{**params,'scales':mx.zeros((n-1,3))},
        'wrong_quaternion_columns':{**params,'quaternions':mx.zeros((n,3))},
        'wrong_opacity_shape':{**params,'opacities':mx.zeros((n,2))},
        'missing_sh_dc':{**params,'sh_coeffs':mx.zeros((n,0,3))},
        'wrong_sh_channels':{**params,'sh_coeffs':mx.zeros((n,1,2))},
        'complex_means':{**params,'means':params['means'].astype(mx.complex64)},
        'nan_means':{**params,'means':params['means'].at[0,0].add(float('nan'))},
        'overflow_scales':{**params,'scales':mx.full((n,3),1000.)},
    }
    persistent=api.PersistentCppCompatRenderer(params)
    for name,bad in scenes.items():
        def one(bad=bad):
            with patch.object(api,'_renderer_class',return_value=BlockedRenderer):
                return expect_rejected(lambda:api.render_cpp_compat(bad,cam))
        def update(bad=bad):
            with patch.object(api,'_renderer_class',return_value=BlockedRenderer),patch.object(persistent._renderer,'update',side_effect=blocked):
                return expect_rejected(lambda:persistent.update(bad))
        add('one_shot_'+name,one,'negative_boundary')
        add('update_'+name,update,'negative_boundary')
    cameras={
        'zero_width':SimpleNamespace(**{**cam.__dict__,'W':0}),
        'negative_height':SimpleNamespace(**{**cam.__dict__,'H':-1}),
        'fractional_width':SimpleNamespace(**{**cam.__dict__,'W':65.5}),
        'wrong_matrix':SimpleNamespace(**{**cam.__dict__,'W2C':mx.eye(3)}),
        'nan_matrix':SimpleNamespace(**{**cam.__dict__,'W2C':mx.eye(4).at[0,0].add(float('nan'))}),
        'zero_focal':SimpleNamespace(**{**cam.__dict__,'fx':0.}),
        'nan_focal':SimpleNamespace(**{**cam.__dict__,'fy':float('nan')}),
        'missing_intrinsic':SimpleNamespace(**{k:v for k,v in cam.__dict__.items() if k!='cx'}),
    }
    for name,bad in cameras.items():
        def one(bad=bad):
            with patch.object(api,'_renderer_class',return_value=BlockedRenderer):return expect_rejected(lambda:api.render_cpp_compat(params,bad))
        def repeated(bad=bad):
            with patch.object(persistent._renderer,'render',side_effect=blocked):return expect_rejected(lambda:persistent.render(bad))
        add('one_shot_camera_'+name,one,'negative_boundary')
        add('persistent_camera_'+name,repeated,'negative_boundary')
    for name,bad in [('wrong_background',mx.zeros((2,))),('nan_background',mx.array([0.,float('nan'),0.])),
                     ('batch_background',mx.zeros((2,3)))]:
        def one(bad=bad):
            with patch.object(api,'_renderer_class',return_value=BlockedRenderer):return expect_rejected(lambda:api.render_cpp_compat(params,cam,background=bad))
        def repeated(bad=bad):
            with patch.object(persistent._renderer,'render',side_effect=blocked):return expect_rejected(lambda:persistent.render(cam,background=bad))
        add('one_shot_'+name,one,'negative_boundary')
        add('persistent_'+name,repeated,'negative_boundary')
    for key in ('tile_sort_impl','tile_range_mode','duplicate_sort_mode'):
        add('invalid_one_shot_'+key,lambda key=key:expect_rejected(lambda:api.render_cpp_compat(params,cam,**{key:'INVALID'})),'negative_mode')
        add('invalid_persistent_'+key,lambda key=key:expect_rejected(lambda:api.PersistentCppCompatRenderer(params,**{key:'INVALID'})),'negative_mode')

    def atomic_update():
        e=api.PersistentCppCompatRenderer(params);before=bits(e.render(cam));owned=e._renderer
        bad={k:v[:8] for k,v in params.items()};bad['quaternions']=mx.zeros((8,3))
        expect_rejected(lambda:e.update(bad))
        unchanged=(e._renderer is owned and e._renderer.n==n and
                   np.array_equal(np.asarray(e._renderer.means),np.asarray(params['means'])))
        # Recover before any further render if the old implementation mutated.
        e.update(params);after=bits(e.render(cam))
        assert np.array_equal(before,after),'Valid replacement failed to recover'
        assert unchanged,'Rejected update mutated persistent scene state'
        return {'rejected_scene_unchanged':True,'valid_update_recovered':True}
    add('rejected_update_atomicity_and_recovery',atomic_update,'state_recovery')
    def rejected_render_recovery():
        e=api.PersistentCppCompatRenderer(params);before=bits(e.render(cam))
        expect_rejected(lambda:e.render(cameras['wrong_matrix']))
        assert np.array_equal(before,bits(e.render(cam)))
        changed={**params,'means':params['means']+mx.array([.2,.1,0.])}
        e.update(changed)
        assert np.array_equal(bits(e.render(cam)),bits(old.render_cpp_compat(changed,cam)))
        return {'repeated_render_and_replacement_recovered':True}
    add('rejected_render_then_valid_replacement',rejected_render_recovery,'state_recovery')
    result={'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'checks':rows,'count':len(rows),'reference_adapter_sha256':digest(reference),
            'installed_fast_sha256':digest(installed.__file__),
            'scope':'Positive cases use actual kernels/reference adapter. Unsafe malformed cases must be rejected before a guarded native boundary.'}
    write_report(out,result);print('RESULT Phase124 worker',result['status'],len(rows),flush=True)
    return result['status']=='PASS'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--reference',type=Path);a=p.parse_args()
    if a.worker:return 0 if worker(a.out,a.reference) else 1
    out,logs=output_paths(a.out);before=invariants();packages=inventory()
    report={'phase':124,'status':'FAIL','environment':metadata(),'gpu_gates':[]}
    try:
        rc,text,err=command(['git','show','6487258:mlx_gs/renderer/mlx3d_cpp_compat.py'])
        if rc:raise RuntimeError('Cannot read pre-Phase124 reference adapter')
        reference=logs/'reference_adapter.py';reference.write_text(text)
        with isolated_runtime(logs,with_mlx3d=True) as (py,tmp,env):
            if command([py,'-m','pip','check'],logs/'dependency_check.log',env=env):raise RuntimeError('Dependency constraints failed')
            report['gpu_gates'].append(gpu_gate())
            contracts=logs/'contracts.json'
            rc=command([py,Path(__file__).resolve(),'--worker','--reference',reference,'--out',contracts],logs/'contracts.log',env=env)
            report['contracts']=json.loads(contracts.read_text()) if contracts.exists() else {'status':'FAIL','error':'no worker report'}
            report['contracts_returncode']=rc
            report['gpu_gates'].append(gpu_gate())
            gate=logs/'api_gate.json'
            rc2=command([py,HERE/'run_cpp_compat_api_integrity_gate.py','--out',gate],logs/'api_gate.log',env=env)
            report['api_gate']=json.loads(gate.read_text()) if gate.exists() else {'status':'FAIL','error':'no gate report'}
            report['api_gate_returncode']=rc2
            command([py,'-m','pip','freeze'],logs/'package_inventory.log',env=env)
            if rc==0 and rc2==0 and report['contracts']['status']=='PASS' and report['api_gate']['status']=='PASS':report['status']='PASS'
    except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)}
    report['invariants']=invariants();assert report['invariants']==before and inventory()==packages
    report['caller_environment_unchanged']=True;report['artifacts']=str(logs)
    write_report(out,report);print('RESULT PHASE124',report['status'],out,flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
