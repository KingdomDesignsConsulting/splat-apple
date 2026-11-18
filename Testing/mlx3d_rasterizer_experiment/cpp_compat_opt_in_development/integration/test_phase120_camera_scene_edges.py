#!/usr/bin/env python3
"""Phase120: camera transitions and replacement scene match one-shot rendering."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer, render_cpp_compat

def scene(n, opacity=0.0):
    return {"means": mx.array(np.column_stack((np.linspace(-0.6,0.6,n),
             np.zeros(n), np.full(n,3.0))).astype(np.float32)),
            "scales": mx.full((n,3), -2.7),
            "quaternions": mx.array(np.tile([1.,0.,0.,0.],(n,1)).astype(np.float32)),
            "opacities": mx.full((n,1), opacity),
            "sh_coeffs": mx.zeros((n,1,3))}

def camera(w,h,shift):
    transform=mx.eye(4)
    transform=mx.array(np.array(transform).astype(np.float32))
    translation=np.array(transform)
    translation[0,3]=shift
    return SimpleNamespace(W=w,H=h,fx=float(min(w,h)),fy=float(min(w,h)),
        cx=w/2.,cy=h/2.,W2C=mx.array(translation))

def main():
    first=scene(128)
    renderer=PersistentCppCompatRenderer(first)
    for label,gs,cam in [
        ("initial",first,camera(128,96,0.0)),
        ("shifted_camera",first,camera(128,96,0.1)),
        ("resized_camera",first,camera(144,112,-0.1)),
    ]:
        a,b=renderer.render(cam),render_cpp_compat(gs,cam)
        mx.eval(a,b)
        if not np.array_equal(np.asarray(a),np.asarray(b)):
            raise AssertionError(label)
        if not np.isfinite(np.asarray(a)).all() or np.asarray(a).shape!=(cam.H,cam.W,3):
            raise AssertionError((label,"invalid shape/values"))
        print("PASS",label,flush=True)
    second=scene(64,2.0)
    renderer.update(second)
    cam=camera(128,96,0.0)
    a,b=renderer.render(cam),render_cpp_compat(second,cam)
    mx.eval(a,b)
    if not np.array_equal(np.asarray(a),np.asarray(b)):
        raise AssertionError("replacement_scene")
    print("PASS replacement_scene",flush=True)

if __name__=="__main__":
    main()
