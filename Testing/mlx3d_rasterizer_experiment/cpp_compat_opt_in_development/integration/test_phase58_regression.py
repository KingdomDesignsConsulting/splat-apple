#!/usr/bin/env python3
"""Regression gates for duplicate sizing, tile ranges and source integrity."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from mlx3d.cameras import Camera
from mlx_gs.renderer import mlx3d_cpp_compat as adapter
from bench_phase49_breakdown import scene, camera


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    a=parser.parse_args()
    source=a.reference.read_bytes()
    assert hashlib.sha256(source).hexdigest() == 'b6aa0444116920d1ae5e526eaf5821e4970bf681c74ce04096fb7f57ae9c11d1'
    # Load the exact pre-change validated source independently of the adapter.
    spec=importlib.util.spec_from_file_location('mlx3d.splatting._phase58_reference',a.reference)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls=adapter._renderer_class()
    for n in (0,1,4096):
        gs=scene(max(n,1))
        means=gs.means[:n]
        args=dict(means=means,quats=gs.quaternions[:n],scales=mx.exp(gs.scales[:n]),
                  opacities=mx.sigmoid(gs.opacities[:n]).reshape((-1,)),
                  colors=mx.clip(gs.sh_coeffs[:n,0,:]*0.28209479177387814+0.5,0,1),
                  sh_degree=0,geometry_policy='cpp_compat',alpha_mode='exponential',t_min=0.0001)
        engines={'original':(cls(**args) if n == 1 else module.FastGaussianRenderer(**args))}
        for sort in ('capacity','active','bucketed','adaptive','auto'):
            for ranges in ('scan','binary','auto'):
                engines[sort+'_'+ranges]=cls(**args,duplicate_sort_mode=sort,tile_range_mode=ranges)
                engines[sort+'_'+ranges]._capacity=1
        for algorithm in ('radix4','auto'):
            engines['automatic_'+algorithm]=cls(
                **args,duplicate_sort_mode='auto',tile_range_mode='auto',tile_sort_impl=algorithm)
        for frame in range(3):
            t=mx.array([0.1*frame,-0.1*frame,-100.0 if frame==2 else 0.0])
            cam=Camera(R=mx.eye(3),t=t,fx=150.,fy=150.,cx=96.5,cy=64.5,width=193,height=129)
            bg=mx.array([0.1,0.2,0.3])
            values={key:engine.render(cam,background=bg) for key,engine in engines.items()}
            mx.eval(values)
            for key,value in values.items():
                for field in ('image','alpha'):
                    actual=np.asarray(value[field]); reference=np.asarray(values['original'][field])
                    assert np.isfinite(actual).all(),(n,frame,key,field)
                    assert np.array_equal(actual,reference),(n,frame,key,field)
        print('PASS', 'tiny-scene fixed-kernel parity' if n == 1 else 'frozen-reference parity', 'image/alpha n=',n,flush=True)
    # Scene updates must alter cached contents and preserve direct parity.
    gs=scene(4096)
    persistent=adapter.PersistentCppCompatRenderer(gs,duplicate_sort_mode='bucketed',tile_range_mode='binary')
    cam=camera(193,129,2)
    before=persistent.render(cam)
    mx.eval(before)
    gs.means=gs.means+mx.array([0.3,0.1,0.0])
    gs.sh_coeffs=gs.sh_coeffs+0.4
    persistent.update(gs)
    after=persistent.render(cam)
    expected=adapter.render_cpp_compat(gs,cam,duplicate_sort_mode='capacity',tile_range_mode='scan')
    mx.eval(after,expected)
    assert not np.array_equal(np.asarray(before),np.asarray(after))
    assert np.array_equal(np.asarray(after),np.asarray(expected))
    print('PASS changed scene update',flush=True)
    original_sha=adapter._VALIDATED_SOURCE_SHA
    try:
        adapter._renderer_class.cache_clear()
        adapter._VALIDATED_SOURCE_SHA='0'*64
        try:
            adapter._renderer_class()
        except RuntimeError as exc:
            assert 'hash mismatch' in str(exc)
        else:
            raise AssertionError('Integrity check accepted wrong hash')
    finally:
        adapter._VALIDATED_SOURCE_SHA=original_sha
        adapter._renderer_class.cache_clear()
    print('PASS source integrity rejection',flush=True)
    for keyword in ('duplicate_sort_mode','tile_range_mode','tile_sort_impl'):
        try: cls(**args,**{keyword:'invalid'})
        except ValueError: pass
        else: raise AssertionError('Accepted invalid mode')
    print('PASS invalid policy rejection',flush=True)


if __name__=='__main__':main()
