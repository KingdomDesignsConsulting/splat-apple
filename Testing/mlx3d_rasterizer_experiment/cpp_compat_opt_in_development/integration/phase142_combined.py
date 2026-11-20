"""Explicit experimental policy, built around the validated native renderer."""
from pathlib import Path
import json
import time
from weakref import ref

# Phase 141 only populates bands with repeatable timing and memory evidence.
ACTIVE_BANDS = ()

def choose_sizing(*, supported, width, height, n, previous, bands=ACTIVE_BANDS):
    if supported and previous is not None and previous[:2] == (width,height):
        active = previous[2]
        for w, h, count, low, high in bands:
            if (width,height,n) == (w,h,count) and low <= active <= high:
                return "active"
    return "auto"

def choose_sort(*, supported, width, height, tiles, slots):
    if supported and (width,height,tiles) == (1920,1080,8160) and 32_000_000 <= slots <= 34_000_000:
        return "selective_final3_scatter512"
    return "auto"

def renderer_api():
    from mlx_gs.renderer import mlx3d_cpp_compat as api
    return api

class CombinedRenderer:
    """Preserves the adapter contracts in a separate opt-in entry point."""
    def __init__(self, gaussians, *, supported_device=False, t_min=0.0001,
                 duplicate_sort_mode="auto", tile_range_mode="auto", tile_sort_impl="auto"):
        self.supported = supported_device is True
        self.options = dict(t_min=t_min, duplicate_sort_mode=duplicate_sort_mode,
                            tile_range_mode=tile_range_mode, tile_sort_impl=tile_sort_impl)
        self._renderer = None
        self._previous = None
        self.last_sort_decisions = []
        self.last_policy_ms = 0.0
        self.update(gaussians)

    def update(self, gaussians):
        api = renderer_api()
        args = api._scene_args(gaussians)
        native = api._renderer_class()
        owner = ref(self)
        class CombinedNative(native):
            def _composite(frame, xy, conic_opac, rgbh, order, offsets, counts, bbox,
                           background, total, w, h, tiles_x, tiles_y, num_tiles):
                selected = owner()
                tick = time.perf_counter()
                length = (max(frame._frame_duplicate_count, 1)
                          if frame._frame_duplicate_count is not None else frame._capacity)
                original = frame.tile_sort_impl
                chosen = original if original != "auto" else choose_sort(
                    supported=selected.supported, width=w, height=h, tiles=num_tiles, slots=length)
                selected.last_policy_ms += (time.perf_counter()-tick)*1000
                selected.last_sort_decisions.append(dict(mode=chosen, capacity=length, tiles=num_tiles,
                                                        sizing=frame.duplicate_sort_mode))
                frame.tile_sort_impl = chosen
                try:
                    return super()._composite(xy, conic_opac, rgbh, order, offsets, counts, bbox,
                                              background, total, w, h, tiles_x, tiles_y, num_tiles)
                finally:
                    frame.tile_sort_impl = original
        replacement = CombinedNative(**args, geometry_policy="cpp_compat",
                                     alpha_mode="exponential", **self.options)
        if self._renderer is not None:
            replacement._capacity = self._renderer._capacity
        self._renderer = replacement
        self._previous = None

    def _render_prepared(self, camera, background):
        self.last_sort_decisions.clear()
        tick = time.perf_counter()
        frame = self._renderer
        original = frame.duplicate_sort_mode
        if original == "auto":
            frame.duplicate_sort_mode = choose_sizing(supported=self.supported, width=camera.width,
                height=camera.height, n=frame.n, previous=self._previous)
        self.last_policy_ms = (time.perf_counter()-tick)*1000
        try:
            image = frame.render(camera, background)["image"]
        finally:
            frame.duplicate_sort_mode = original
        observed = frame.last_frame_counts.get("active")
        self._previous = (camera.width,camera.height,observed) if observed is not None else None
        return image

    def render(self, camera, background=None):
        api = renderer_api()
        return self._render_prepared(api._camera(camera), api._background(background))

def render_combined(gaussians, camera, background=None, **options):
    api = renderer_api()
    view = api._camera(camera)
    bg = api._background(background)
    return CombinedRenderer(gaussians, **options)._render_prepared(view, bg)

def run_contracts(out, reference, job):
    import hashlib
    import sys
    from functools import partial
    from unittest.mock import patch
    import mlx.core as mx
    import numpy as np
    from qualification_common import invariants, inventory, metadata, write_report
    from phase138_checkpoint import hash_inventory
    from run_phase124_api_contract import worker
    from bench_phase49_breakdown import camera, scene
    api = renderer_api()
    plain = api.PersistentCppCompatRenderer
    raw = out.with_name("combined_contract_detail.json")
    with patch.object(api, "PersistentCppCompatRenderer", partial(CombinedRenderer, supported_device=True)), \
         patch.object(api, "render_cpp_compat", partial(render_combined, supported_device=True)):
        passed = worker(raw, reference)
    data = json.loads(raw.read_text())
    passed = passed and data["count"] == 65
    edges = []
    checksums = {}
    if passed:
        gs = scene(96)
        candidate = CombinedRenderer(gs, supported_device=True)
        control = plain(gs)
        mod = sys.modules[api._renderer_class().__module__]
        sort = mod._selective_final3_argsort
        calls = []
        def trace(keys, tiles, **opts):
            calls.append((int(keys.size), tiles, opts.get("scatter_group")))
            return sort(keys, tiles, **opts)
        for width,height,cap,expected in (
            (1920,1080,31_999_999,False), (1920,1080,32_000_000,True),
            (1920,1080,34_000_000,True), (1920,1080,34_000_001,False),
            (2560,1440,32_000_000,False),
        ):
            candidate._renderer._capacity = control._renderer._capacity = cap
            view = camera(width,height,3)
            calls.clear()
            with patch.object(mod, "_selective_final3_argsort", trace):
                image = candidate.render(view)
            baseline = control.render(view)
            mx.eval(image, baseline)
            a,b = np.asarray(image),np.asarray(baseline)
            if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
                raise AssertionError("Boundary bitwise parity failed")
            if bool(calls) != expected or (expected and calls != [(cap,8160,512)]):
                raise AssertionError(f"Unexpected final-pass boundary dispatch: {calls}")
            key = f"{width}x{height}_{cap}"
            checksums[key] = hashlib.sha256(np.ascontiguousarray(a)).hexdigest()
            edges.append(dict(width=width,height=height,capacity=cap,actual_calls=list(calls),
                              selections=list(candidate.last_sort_decisions)))
    write_report(out,dict(status="PASS" if passed else "FAIL",job=job,invariants=invariants(),
                 runtime=metadata(), dependency_inventory_sha=hash_inventory(inventory()),
                 contracts=data,boundaries=edges,output_checksums=checksums))
    return passed
