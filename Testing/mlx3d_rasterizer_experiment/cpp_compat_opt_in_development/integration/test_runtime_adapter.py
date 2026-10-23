#!/usr/bin/env python3
"""Self-contained opt-in runtime smoke test: no COLMAP, PyTorch, or native extensions.

Run from repository root in gs-mps:
    python Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/test_runtime_adapter.py
"""
import argparse
from types import SimpleNamespace
import numpy as np
import mlx.core as mx

from mlx_gs.core.gaussians import Gaussians
from mlx_gs.renderer.renderer import render
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=96)
    a = p.parse_args()
    if a.n < 1 or a.width < 1 or a.height < 1:
        p.error("--n, --width and --height must be positive")
    rng = np.random.default_rng(12345)
    means = mx.array(np.column_stack([
        rng.uniform(-1.0, 1.0, a.n),
        rng.uniform(-0.8, 0.8, a.n),
        rng.uniform(3.0, 6.0, a.n),
    ]).astype(np.float32))
    scales = mx.full((a.n, 3), -3.0, dtype=mx.float32)
    quats = mx.array(np.tile([1.0, 0.0, 0.0, 0.0], (a.n, 1)).astype(np.float32))
    opacities = mx.full((a.n, 1), 0.0, dtype=mx.float32)
    sh = mx.zeros((a.n, 1, 3), dtype=mx.float32)
    gs = Gaussians(means, scales, quats, opacities, sh)
    f = float(min(a.width, a.height))
    cam = SimpleNamespace(W=a.width, H=a.height, fx=f, fy=f,
                          cx=a.width / 2, cy=a.height / 2,
                          W2C=mx.eye(4, dtype=mx.float32))
    result = render(gs, cam, rasterizer_type="mlx3d_cpp_compat")

    from mlx3d.cameras import Camera as FastCamera
    fast_cam = FastCamera(R=cam.W2C[:3, :3], t=cam.W2C[:3, 3],
                          fx=cam.fx, fy=cam.fy, cx=cam.cx + .5, cy=cam.cy + .5,
                          width=cam.W, height=cam.H)
    direct = _renderer_class()(
        means=means, quats=quats, scales=mx.exp(scales),
        opacities=mx.sigmoid(opacities).reshape((-1,)),
        colors=mx.full((a.n, 3), 0.5, dtype=mx.float32),
        sh_degree=0, geometry_policy="cpp_compat",
        alpha_mode="exponential", t_min=0.0001,
    ).render(fast_cam)["image"]
    mx.eval(result, direct)
    actual, expected = np.asarray(result), np.asarray(direct)
    assert actual.shape == (a.height, a.width, 3), actual.shape
    assert np.isfinite(actual).all(), "Non-finite pixels"
    assert np.array_equal(actual, expected), "Public dispatch differs from direct renderer"
    assert np.any(actual > 0), "Unexpected empty render"
    print("PASS: public dispatch matches validated renderer bitwise")
    print("PASS: finite nonempty image", actual.shape)
    print("PASS: no COLMAP or torch_gs._C dependency required")


if __name__ == "__main__":
    main()
