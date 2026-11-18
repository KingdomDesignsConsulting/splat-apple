#!/usr/bin/env python3
"""Execute original and staged Fast MLX3D Metal kernels without changing site-packages.
Run from splat-apple repository root in gs-mps. A quick smoke test, not the 14-case regression.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np
import mlx.core as mx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--staged', type=Path, default=Path('Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/staging/mlx3d/splatting/fast.py'))
    ap.add_argument('--ply', type=Path, default=Path('results/bonsai_mlx_cpp_20261008_200230/ply/bonsai_final.ply'))
    ap.add_argument('--n', type=int, default=4096)
    ap.add_argument('--camera', type=int, default=72)
    ap.add_argument('--data-dir', type=Path, default=Path('/Users/Shared/Main Hard Drive/My Projects/Projects In Progress/Applications/Gaussian Splatting/Scenes/360_v2/bonsai'))
    ap.add_argument('--img-folder', default='images_4')
    a = ap.parse_args()
    sys.path.insert(0, str(Path.cwd() / 'Testing/mlx3d_rasterizer_experiment'))
    sys.path.insert(0, str(Path.cwd()))
    from bench_bonsai_trained import load_trained_ply
    from bench_bonsai_alpha_transmittance import C0
    from mlx_gs.io.colmap import load_colmap_dataset
    from mlx3d.cameras import Camera
    from mlx3d.splatting import GaussianModel
    import mlx3d.splatting.fast as installed

    staged_path = a.staged.resolve()
    assert staged_path.is_file(), f'Missing staged fast.py: {staged_path}'
    assert a.ply.is_file(), f'Missing scene PLY: {a.ply}'
    spec = importlib.util.spec_from_file_location('mlx3d.splatting._fast_cpp_compat_smoketest', staged_path)
    staged = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = staged
    spec.loader.exec_module(staged)

    source = load_trained_ply(a.ply)
    n = min(a.n, len(source[0]))
    positions, scales, quats, opacity_logits = [mx.array(np.ascontiguousarray(v[:n])) for v in source[:4]]
    sh = mx.full((n,1,3), (0.999999-0.5)/C0, dtype=mx.float32)
    model = GaussianModel({'means': positions, 'scales': scales, 'quats': quats,
      'opacities': mx.reshape(opacity_logits, (n,)), 'sh_dc': sh,
      'sh_rest': mx.zeros((n,0,3),dtype=mx.float32)},sh_degree=0)
    model.active_sh_degree = 0
    _, _, cameras, _ = load_colmap_dataset(str(a.data_dir), a.img_folder)
    cam = cameras[a.camera]
    raw = np.asarray(cam.W2C, np.float32)
    camera = Camera(R=mx.array(raw[:3,:3]), t=mx.array(raw[:3,3]),
      fx=float(cam.fx), fy=float(cam.fy), cx=float(cam.cx)+0.5,
      cy=float(cam.cy)+0.5, width=int(cam.W), height=int(cam.H))

    def execute(title, cls, **kwargs):
        print('Executing:', title, flush=True)
        ren = cls(model, sh_degree=0, antialias=False, t_min=1/255, **kwargs)
        rendered = ren.render(camera)
        mx.eval(rendered['alpha'], rendered['image'])
        alpha = np.asarray(rendered['alpha'])
        image = np.asarray(rendered['image'])
        assert alpha.shape[:2] == (int(cam.H), int(cam.W)), (title, alpha.shape)
        assert np.isfinite(alpha).all() and np.isfinite(image).all(), title+' nonfinite output'
        print(f'  PASS: alpha shape {alpha.shape}; alpha sum {alpha.sum():.6f}', flush=True)
        return alpha, image

    old_a, old_i = execute('installed default', installed.FastGaussianRenderer)
    new_a, new_i = execute('staged default', staged.FastGaussianRenderer)
    assert np.array_equal(old_a,new_a), 'STAGED DEFAULT CHANGED ALPHA!'
    assert np.array_equal(old_i,new_i), 'STAGED DEFAULT CHANGED RGB!'
    print('PASS: staged default bitwise-identical to installed default',flush=True)
    exp_a, exp_i = execute('staged default geometry + exponential alpha', staged.FastGaussianRenderer,
                           geometry_policy='default', alpha_mode='exponential')
    compat_a, compat_i = execute('staged CPP geometry + exponential alpha', staged.FastGaussianRenderer,
                                 geometry_policy='cpp_compat', alpha_mode='exponential')
    print('PASS: CPP compatibility geometry and raster Metal kernels executed',flush=True)
    print(f'CPP geometry effect: max alpha difference {np.max(np.abs(compat_a-exp_a)):.9g}')
    high = staged.FastGaussianRenderer(model,sh_degree=0,antialias=False,geometry_policy='cpp_compat',
                                       alpha_mode='exponential',t_min=0.0001).render(camera)
    mx.eval(high['alpha'],high['image'])
    assert np.isfinite(np.asarray(high['alpha'])).all()
    print('PASS: high-accuracy t_min=0.0001 compiled and ran')
    print('SMOKE TEST PASSED. Full 14-case regression remains required.')

if __name__ == '__main__':
    main()
