#!/usr/bin/env python3
"""Runtime integration smoke test. Run from splat-apple root in gs-mps.

Does NOT modify installed packages. Exercises the public splat-apple render()
dispatch and compares it to the directly constructed validated Fast renderer.
"""
import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx

from mlx_gs.core.gaussians import Gaussians
from mlx_gs.io.colmap import load_colmap_dataset
from mlx_gs.renderer.renderer import render
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", type=Path, default=Path("results/bonsai_mlx_cpp_20261008_200230/ply/bonsai_final.ply"))
    p.add_argument("--data-dir", default="/Users/Shared/Main Hard Drive/My Projects/Projects In Progress/Applications/Gaussian Splatting/Scenes/360_v2/bonsai")
    p.add_argument("--img-folder", default="images_4")
    p.add_argument("--camera", type=int, default=72)
    p.add_argument("--n", type=int, default=4096)
    a = p.parse_args()
    if not a.ply.is_file(): p.error(f"Missing PLY: {a.ply}")
    from bench_bonsai_trained import load_trained_ply
    from mlx3d.cameras import Camera as FastCamera
    source = load_trained_ply(a.ply)
    n = min(a.n, len(source[0]))
    means, scales, quats, opacities = [
        mx.array(np.ascontiguousarray(x[:n])) for x in source[:4]
    ]
    sh = mx.full((n, 1, 3), (0.999999 - 0.5) / 0.28209479177387814, dtype=mx.float32)
    gs = Gaussians(means=means, scales=scales, quaternions=quats, opacities=opacities, sh_coeffs=sh)
    _, _, cameras, _ = load_colmap_dataset(a.data_dir, a.img_folder)
    cam = cameras[a.camera]
    adapted = render(gs, cam, rasterizer_type="mlx3d_cpp_compat")
    fast_cam = FastCamera(R=cam.W2C[:3, :3], t=cam.W2C[:3, 3],
                          fx=float(cam.fx), fy=float(cam.fy),
                          cx=float(cam.cx)+0.5, cy=float(cam.cy)+0.5,
                          width=int(cam.W), height=int(cam.H))
    direct = _renderer_class()(
        means=means, quats=quats, scales=mx.exp(scales),
        opacities=mx.sigmoid(opacities).reshape((-1,)),
        colors=mx.clip(sh[:, 0, :] * 0.28209479177387814 + 0.5, 0.0, 1.0),
        sh_degree=0, geometry_policy="cpp_compat", alpha_mode="exponential",
        t_min=0.0001,
    ).render(fast_cam)["image"]
    mx.eval(adapted, direct)
    actual, expected = np.asarray(adapted), np.asarray(direct)
    assert actual.shape == (int(cam.H), int(cam.W), 3)
    assert np.isfinite(actual).all()
    assert np.array_equal(actual, expected), "Runtime adapter differs from validated direct renderer"
    print("PASS: public opt-in dispatch matches validated Fast source bitwise")
    print("PASS: output is finite, with expected shape", actual.shape)
    print("NOTE: no backwards/gradient test; backend is intentionally forward-only")

if __name__ == "__main__":
    main()
