"""Experimental explicit opt-in. Existing adapter auto remains the fallback."""
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer, _renderer_class, _scene_args

def select_sort(*, supported_device, width, height, tiles, capacity):
    # Start with a narrow extrapolation around the replicated 33.15M-slot
    # Full HD case. Capacities outside this band use the existing auto path.
    if (supported_device and (width, height, tiles) == (1920, 1080, 8160)
            and 32_000_000 <= capacity <= 34_000_000):
        return "selective_final3_scatter512"
    return "auto"

class ExperimentalAdaptiveRenderer(PersistentCppCompatRenderer):
    """Separate opt-in API. Device support must be explicitly established."""

    def __init__(self, gaussians, *, supported_device=False, **kwargs):
        self._supports_experiment = supported_device
        self.last_sort_decisions = []
        super().__init__(gaussians, tile_sort_impl="auto", **kwargs)

    def render(self, camera, background=None):
        self.last_sort_decisions.clear()
        return super().render(camera, background)

    def update(self, gaussians):
        if not hasattr(self, "_adaptive_class"):
            native = _renderer_class()
            owner = self

            class AdaptiveFast(native):
                def _composite(
                    frame, xy, conic_opac, rgbh, order, offsets, counts, bbox,
                    background, total, w, h, tiles_x, tiles_y, num_tiles,
                ):
                    active = frame._frame_duplicate_count
                    capacity = max(active, 1) if active is not None else frame._capacity
                    mode = select_sort(supported_device=owner._supports_experiment,
                                       width=w, height=h, tiles=num_tiles, capacity=capacity)
                    owner.last_sort_decisions.append({"mode": mode, "capacity": capacity,
                                                      "tiles": num_tiles})
                    frame.tile_sort_impl = mode
                    try:
                        return super()._composite(xy, conic_opac, rgbh, order, offsets,
                                                  counts, bbox, background, total, w, h,
                                                  tiles_x, tiles_y, num_tiles)
                    finally:
                        frame.tile_sort_impl = "auto"

            self._adaptive_class = AdaptiveFast
        replacement = self._adaptive_class(
            **_scene_args(gaussians), geometry_policy="cpp_compat", alpha_mode="exponential",
            t_min=self.t_min, duplicate_sort_mode=self.duplicate_sort_mode,
            tile_range_mode=self.tile_range_mode, tile_sort_impl="auto",
        )
        if self._renderer is not None:
            replacement._capacity = self._renderer._capacity
        self._renderer = replacement
