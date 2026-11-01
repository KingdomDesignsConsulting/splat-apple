#!/usr/bin/env python3
"""Persistent CPP-compat API regression: repeat frame, scene update, direct parity."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from types import SimpleNamespace
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer, render_cpp_compat

def main():
    n = 128
    means = mx.array(np.column_stack((np.linspace(-0.5, 0.5, n),
        np.zeros(n), np.full(n, 3.0))).astype(np.float32))
    gaussians = {
        "means": means,
        "scales": mx.full((n, 3), -2.5, dtype=mx.float32),
        "quaternions": mx.array(np.tile([1., 0., 0., 0.], (n, 1)).astype(np.float32)),
        "opacities": mx.zeros((n, 1), dtype=mx.float32),
        "sh_coeffs": mx.zeros((n, 1, 3), dtype=mx.float32),
    }
    cam = SimpleNamespace(W=128, H=96, fx=100., fy=100.,
                          cx=64., cy=48., W2C=mx.eye(4))
    renderer = PersistentCppCompatRenderer(gaussians, tile_sort_impl="auto")
    def match(label, params):
        persistent = renderer.render(cam)
        oneshot = render_cpp_compat(params, cam)
        mx.eval(persistent, oneshot)
        a, b = np.asarray(persistent), np.asarray(oneshot)
        if a.shape != (96, 128, 3) or not np.isfinite(a).all():
            raise AssertionError((label, "shape/finite", a.shape))
        if not np.array_equal(a, b):
            raise AssertionError((label, "persistent vs one-shot mismatch"))
        return a
    initial = match("initial", gaussians)
    repeat = match("repeat", gaussians)
    if not np.array_equal(initial, repeat):
        raise AssertionError("unchanged scene changed across renders")
    updated = dict(gaussians)
    updated["opacities"] = mx.full((n, 1), 3.0, dtype=mx.float32)
    renderer.update(updated)
    after = match("scene updated", updated)
    if np.array_equal(initial, after):
        raise AssertionError("scene update produced no visible change")
    print("PASS persistent initial, repeated, updated scene vs one-shot exact parity", flush=True)

if __name__ == "__main__":
    main()
