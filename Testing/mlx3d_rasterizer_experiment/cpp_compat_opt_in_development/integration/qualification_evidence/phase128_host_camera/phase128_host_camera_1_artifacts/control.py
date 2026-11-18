"""Opt-in, forward-only CPP-compatible MLX3D Fast rendering adapter.

Uses the validated repository-local MLX3D source snapshot. No site-packages
writes, monkeypatching, or changes to the existing cpp/python rasterizers.
"""
from functools import lru_cache
from collections.abc import Mapping
import math
import operator
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

import mlx.core as mx

_VALIDATED_SOURCE_SHA = "56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710"
_ORIGINAL_INSTALLED_SHA = "4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc"
_MODULE_NAME = "mlx3d.splatting._splat_apple_cpp_compat_fast"


@lru_cache(maxsize=1)
def _renderer_class():
    # Only uncached calls enter here. Never expose a previous owned module
    # after a rejected revalidation; existing external class references are
    # unaffected, as documented by the load-time integrity contract.
    sys.modules.pop(_MODULE_NAME, None)
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
        return module.FastGaussianRenderer
    except BaseException:
        # Cleanup also covers a missing export and an interrupted import.
        # Re-raise unchanged so interrupts and dependency errors retain meaning.
        sys.modules.pop(_MODULE_NAME, None)
        raise



_FIELDS = ("means", "scales", "quaternions", "opacities", "sh_coeffs")
_REAL_DTYPES = tuple(getattr(mx, name) for name in (
    "float16", "bfloat16", "float32", "float64", "int8", "uint8", "int16",
    "uint16", "int32", "uint32", "int64", "uint64") if hasattr(mx, name))


def _real_array(value, name):
    if not isinstance(value, mx.array) or value.dtype not in _REAL_DTYPES:
        raise TypeError(f"{name} must be a real numeric MLX array")
    return value


def _scene_args(gaussians):
    if isinstance(gaussians, Mapping):
        params = {name: gaussians[name] for name in _FIELDS}
    else:
        params = {name: getattr(gaussians, name) for name in _FIELDS}
    for name, value in params.items():
        _real_array(value, f"Gaussian {name}")
    means = params["means"]
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("Gaussian means must have shape (N,3)")
    n = means.shape[0]
    for name, shape in (("scales", (n,3)), ("quaternions", (n,4))):
        if tuple(params[name].shape) != shape:
            raise ValueError(f"Gaussian {name} must have shape {shape}")
    if tuple(params["opacities"].shape) not in ((n,), (n,1)):
        raise ValueError("Gaussian opacities must have shape (N,) or (N,1)")
    sh = params["sh_coeffs"]
    if sh.ndim != 3 or sh.shape[0] != n or sh.shape[1] < 1 or sh.shape[2] != 3:
        raise ValueError("Gaussian sh_coeffs must have shape (N,K,3), K>=1")
    # Preserve original activation dtype/rounding, including float16 inputs.
    scales = mx.exp(params["scales"])
    opacities = mx.sigmoid(params["opacities"]).reshape((-1,))
    colors = mx.clip(sh[:,0,:] * 0.28209479177387814 + 0.5, 0.0, 1.0)
    checks = [
        ("means", mx.all(mx.isfinite(means))),
        ("quaternions", mx.all(mx.isfinite(params["quaternions"]))),
        ("SH DC coefficients", mx.all(mx.isfinite(sh[:,0,:]))),
        ("activated scales", mx.all(mx.isfinite(scales))),
        ("opacity logits", mx.all(~mx.isnan(params["opacities"]))),
    ]
    mx.eval(*(test for _,test in checks))
    for name, test in checks:
        if not bool(test.item()):
            raise ValueError(f"Gaussian {name} contain unsupported non-finite values")
    return dict(means=means, quats=params["quaternions"], scales=scales,
                opacities=opacities, colors=colors, sh_degree=0)


def _camera(camera):
    from mlx3d.cameras import Camera as FastCamera
    dimensions = []
    for name in ("W", "H"):
        value = getattr(camera, name)
        if isinstance(value, bool):
            raise TypeError(f"Camera {name} must be a positive integer")
        try: dimension = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"Camera {name} must be a positive integer") from exc
        if dimension <= 0:
            raise ValueError(f"Camera {name} must be positive")
        dimensions.append(dimension)
    scalars = {name: float(getattr(camera, name)) for name in ("fx", "fy", "cx", "cy")}
    if not all(math.isfinite(value) for value in scalars.values()):
        raise ValueError("Camera intrinsics must be finite")
    if scalars["fx"] <= 0 or scalars["fy"] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    w2c = _real_array(camera.W2C, "Camera W2C")
    if tuple(w2c.shape) != (4,4):
        raise ValueError("Camera W2C must have shape (4,4)")
    if not bool(mx.all(mx.isfinite(w2c)).item()):
        raise ValueError("Camera W2C must be finite")
    return FastCamera(R=w2c[:3,:3], t=w2c[:3,3], fx=scalars["fx"], fy=scalars["fy"],
                      cx=scalars["cx"]+0.5, cy=scalars["cy"]+0.5,
                      width=dimensions[0], height=dimensions[1])


def _background(background):
    if background is None:
        return None
    _real_array(background, "Background")
    if tuple(background.shape) not in ((3,), (1,3), (3,1)):
        raise ValueError("Background must have shape (3,), (1,3), or (3,1)")
    if not bool(mx.all(mx.isfinite(background)).item()):
        raise ValueError("Background must be finite")
    return background.reshape((3,))


def render_cpp_compat(
    gaussians, camera, background=None, *, t_min=0.0001,
    duplicate_sort_mode="auto", tile_range_mode="auto", tile_sort_impl="auto",
):
    """Validated forward-only adapter; preserve original activation rounding."""
    fast_camera = _camera(camera)
    args = _scene_args(gaussians)
    background = _background(background)
    renderer = _renderer_class()(
        **args, geometry_policy="cpp_compat", alpha_mode="exponential", t_min=t_min,
        duplicate_sort_mode=duplicate_sort_mode, tile_range_mode=tile_range_mode,
        tile_sort_impl=tile_sort_impl,
    )
    return renderer.render(fast_camera, background=background)["image"]


class PersistentCppCompatRenderer:
    """Forward-only fixed-scene renderer; explicit, atomic scene replacement.

    Call update() after changing scene data. A rejected replacement leaves the
    previous scene usable. Do not share across concurrent render/update calls.
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
        """Validate and build before swapping; rejected updates are atomic."""
        args = _scene_args(gaussians)
        replacement = _renderer_class()(
            **args, geometry_policy="cpp_compat", alpha_mode="exponential", t_min=self.t_min,
            duplicate_sort_mode=self.duplicate_sort_mode, tile_range_mode=self.tile_range_mode,
            tile_sort_impl=self.tile_sort_impl,
        )
        if self._renderer is not None:
            # Preserve the existing reserve policy across valid scene changes.
            # The validated native renderer still owns safe growth/rerender.
            replacement._capacity = self._renderer._capacity
        self._renderer = replacement

    def render(self, camera, background=None):
        """Return HxWx3 RGB; reject bad camera/background before native work."""
        return self._renderer.render(_camera(camera), background=_background(background))["image"]
