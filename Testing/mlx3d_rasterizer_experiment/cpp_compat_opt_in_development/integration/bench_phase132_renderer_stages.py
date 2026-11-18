"""Synchronized host-wall stage decomposition through experimental instrumentation."""
from contextlib import ExitStack
import statistics
import sys
from unittest.mock import patch

def run(context):
    mx, np, api, gs, cam, timed, equal = context
    engine = api.PersistentCppCompatRenderer(gs)
    native = engine._renderer
    module = sys.modules[type(native).__module__]
    camera = api._camera(cam)
    mx.eval(native.cov3d)
    reference = native.render(camera)
    mx.eval(reference)
    stage_names = {
        "_cpp_compat_geometry_kernel": "projection_geometry",
        "_expand_kernel": "depth_order_prefix_and_expand",
        "_radix_tile_argsort": "mask_and_tile_sort",
        "_ranges_kernel": "tile_ranges",
        "_ranges_binary_kernel": "tile_ranges",
        "_cpp_compat_raster_kernel": "raster_composite",
    }
    rows = []
    for index in range(4):
        stages = {}
        def wrap(fn, stage):
            def instrument(*args, **kwargs):
                # The sort's keys may still carry the tail-mask dependency.
                if stage == "tile_ranges" and "mask_and_tile_sort" not in stages:
                    _, elapsed = timed(lambda: mx.eval(kwargs["inputs"][0], kwargs["inputs"][1]))
                    stages["mask_and_tile_sort"] = elapsed
                value, elapsed = timed(lambda: fn(*args, **kwargs), evaluate=True)
                stages[stage] = stages.get(stage, 0.0) + elapsed
                return value
            return instrument
        def frame(instrument):
            with ExitStack() as stack:
                if instrument:
                    for name, stage in stage_names.items():
                        stack.enter_context(patch.object(module, name, wrap(getattr(module,name), stage)))
                return timed(lambda: native.render(camera), evaluate=True)
        order = (False, True) if index % 2 == 0 else (True, False)
        times = {}
        for instrument in order:
            image, elapsed = frame(instrument)
            equal(reference["image"], image["image"])
            equal(reference["alpha"], image["alpha"])
            times["instrumented" if instrument else "baseline"] = elapsed
        rows.append({"order": list(order), "wall_ms": times, "stages_ms": stages})
    ranks = []
    for stage in sorted(rows[0]["stages_ms"]):
        ranks.append({"stage": stage, "median_ms": statistics.median(r["stages_ms"][stage] for r in rows)})
    ranks.sort(key=lambda r: -r["median_ms"])
    baseline = statistics.median(row["wall_ms"]["baseline"] for row in rows)
    for rank in ranks:
        rank["isolated_fraction_of_baseline"] = rank["median_ms"] / baseline
    return {"rows": rows, "ranked": ranks, "baseline_median_ms": baseline,
            "scope": "Stage walls include host eval and dependency work. Fractions are rough Amdahl indicators, not additive native GPU timings."}
