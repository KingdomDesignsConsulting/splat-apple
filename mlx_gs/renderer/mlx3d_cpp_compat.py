"""Opt-in, forward-only CPP-compatible MLX3D Fast rendering adapter.

Uses the validated repository-local MLX3D source snapshot. No site-packages
writes, monkeypatching, or changes to the existing cpp/python rasterizers.
"""
from functools import lru_cache
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

import mlx.core as mx

_VALIDATED_SOURCE_SHA = "804753d13a4ec05b73935c56fbc8bab92fc8a298c85efc743d291d6acbc99780"
_ORIGINAL_INSTALLED_SHA = "4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc"
_MODULE_NAME = "mlx3d.splatting._splat_apple_cpp_compat_fast"


@lru_cache(maxsize=1)
def _renderer_class():
    # Delayed import: normal splat-apple users do not need mlx3d installed.
    installed = importlib.import_module("mlx3d.splatting.fast")
    installed_path = Path(installed.__file__).resolve()
    if hashlib.sha256(installed_path.read_bytes()).hexdigest() != _ORIGINAL_INSTALLED_SHA:
        raise RuntimeError("Installed MLX3D Fast source differs from validated baseline; refusing compatibility adapter")
    root = Path(__file__).resolve().parents[2]
    source = root / "Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py"
    if not source.is_file():
        raise FileNotFoundError(f"Validated CPP-compat renderer missing: {source}")
    if hashlib.sha256(source.read_bytes()).hexdigest() != _VALIDATED_SOURCE_SHA:
        raise RuntimeError("CPP-compat renderer hash mismatch; refusing unvalidated source")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CPP-compat renderer: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module.FastGaussianRenderer


def render_cpp_compat(
    gaussians, camera, background=None, *, t_min=0.0001,
    duplicate_sort_mode="auto", tile_range_mode="auto", tile_sort_impl="auto",
):
    """Render activated 3DGS values with validated opt-in MLX3D Fast geometry.

    Input uses splat-apple's log-scales, opacity logits and SH-DC coefficients.
    Returns an (H,W,3) MLX array like mlx_gs.renderer.renderer.render().
    Forward-only: do not select this backend for differentiable training.
    """
    from mlx3d.cameras import Camera as FastCamera

    params = gaussians if isinstance(gaussians, dict) else {
        "means": gaussians.means,
        "scales": gaussians.scales,
        "quaternions": gaussians.quaternions,
        "opacities": gaussians.opacities,
        "sh_coeffs": gaussians.sh_coeffs,
    }
    w2c = camera.W2C
    fast_camera = FastCamera(
        R=w2c[:3, :3], t=w2c[:3, 3],
        fx=float(camera.fx), fy=float(camera.fy),
        cx=float(camera.cx) + 0.5, cy=float(camera.cy) + 0.5,
        width=int(camera.W), height=int(camera.H),
    )
    colors = mx.clip(
        params["sh_coeffs"][:, 0, :] * 0.28209479177387814 + 0.5,
        0.0, 1.0,
    )
    renderer = _renderer_class()(
        means=params["means"],
        quats=params["quaternions"],
        scales=mx.exp(params["scales"]),
        opacities=mx.sigmoid(params["opacities"]).reshape((-1,)),
        colors=colors,
        sh_degree=0,
        geometry_policy="cpp_compat",
        alpha_mode="exponential",
        t_min=t_min,
        duplicate_sort_mode=duplicate_sort_mode,
        tile_range_mode=tile_range_mode,
        tile_sort_impl=tile_sort_impl,
    )
    return renderer.render(fast_camera, background=background)["image"]


class PersistentCppCompatRenderer:
    """Forward-only renderer for successive camera frames of a fixed Gaussian scene.

    Keeps MLX3D's activated arrays, covariance, static colors and tile capacity
    resident across frames. No implicit scene-change detection: call update()
    whenever Gaussian values change. Do not share across concurrent threads.
    The existing render(..., rasterizer_type="mlx3d_cpp_compat") remains one-shot.
    """

    def __init__(
        self, gaussians, *, t_min=0.0001, duplicate_sort_mode="auto",
        tile_range_mode="auto", tile_sort_impl="auto",
    ):
        self.t_min = float(t_min)
        self.duplicate_sort_mode = duplicate_sort_mode
        self.tile_range_mode = tile_range_mode
        self.tile_sort_impl = tile_sort_impl
        self._renderer = None
        self.update(gaussians)

    def update(self, gaussians):
        """Replace scene contents and invalidate cached geometry/colors."""
        params = gaussians if isinstance(gaussians, dict) else {
            "means": gaussians.means,
            "scales": gaussians.scales,
            "quaternions": gaussians.quaternions,
            "opacities": gaussians.opacities,
            "sh_coeffs": gaussians.sh_coeffs,
        }
        colors = mx.clip(
            params["sh_coeffs"][:, 0, :] * 0.28209479177387814 + 0.5,
            0.0, 1.0,
        )
        args = dict(
            means=params["means"],
            quats=params["quaternions"],
            scales=mx.exp(params["scales"]),
            opacities=mx.sigmoid(params["opacities"]).reshape((-1,)),
            colors=colors,
            sh_degree=0,
        )
        if self._renderer is None:
            self._renderer = _renderer_class()(
                **args, geometry_policy="cpp_compat",
                alpha_mode="exponential", t_min=self.t_min,
                duplicate_sort_mode=self.duplicate_sort_mode,
                tile_range_mode=self.tile_range_mode,
                tile_sort_impl=self.tile_sort_impl,
            )
        else:
            self._renderer.update(**args)

    def render(self, camera, background=None):
        """Return an HxWx3 MLX image from the unchanged scene at this camera."""
        from mlx3d.cameras import Camera as FastCamera
        w2c = camera.W2C
        fast_camera = FastCamera(
            R=w2c[:3, :3], t=w2c[:3, 3],
            fx=float(camera.fx), fy=float(camera.fy),
            cx=float(camera.cx) + 0.5, cy=float(camera.cy) + 0.5,
            width=int(camera.W), height=int(camera.H),
        )
        return self._renderer.render(fast_camera, background=background)["image"]
