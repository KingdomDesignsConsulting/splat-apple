"""Compare the five sorting policies and verify real dispatch through instrumentation."""
from contextlib import ExitStack
import statistics
import sys
from unittest.mock import patch

def run(context, block):
    mx, np, api, gs, cam, timed, equal = context
    from bench_phase49_breakdown import camera, scene
    from experimental_adaptive_sort import ExperimentalAdaptiveRenderer, select_sort
    for capacity in [7_999_999, 8_000_000, 8_000_001, 31_999_999, 32_000_000,
                     33_154_187, 34_000_000, 34_000_001]:
        for tiles in [7999, 8000, 8001, 8160, 8192, 14400]:
            for supported in (False, True):
                observed = select_sort(supported_device=supported, capacity=capacity,
                                       width=1920, height=1080, tiles=tiles)
                expected = (supported and tiles == 8160 and 32_000_000 <= capacity <= 34_000_000)
                assert (observed == "selective_final3_scatter512") == expected
    names = ["auto", "radix4", "selective_final3", "selective_final3_scatter512", "adaptive"]
    constructors = {mode: lambda mode=mode: api.PersistentCppCompatRenderer(gs, tile_sort_impl=mode)
                    for mode in names[:-1]}
    constructors["adaptive"] = lambda: ExperimentalAdaptiveRenderer(gs, supported_device=True)
    cold_order = (names[block:] + names[:block])[::1 if block % 2 == 0 else -1]
    cold = {}
    engines = {}
    reference = None
    for mode in cold_order:
        engine, construct = timed(constructors[mode])
        image, first = timed(lambda: engine.render(cam), evaluate=True)
        if reference is None:
            reference = image
        equal(reference, image)
        engines[mode] = engine
        cold[mode] = {"constructor_ms": construct, "first_frame_ms": first}
    samples = []
    for index in range(6):
        shift = (index // 2 + block) % len(names)
        order = (names[shift:] + names[:shift])[::1 if index % 2 == 0 else -1]
        row = {}
        for mode in order:
            output, elapsed = timed(lambda: engines[mode].render(cam), evaluate=True)
            equal(reference, output)
            row[mode] = elapsed
        samples.append({"order": order, "ms": row})
    native = api._renderer_class()
    module = sys.modules[native.__module__]
    def trace(engine, view):
        actual = []
        def wrap(fn, route):
            def call(*args, **kwargs):
                actual.append((route, kwargs.get("scatter_group")))
                return fn(*args, **kwargs)
            return call
        with ExitStack() as stack:
            for name, route in [("_radix_tile_argsort", "radix4"),
                                ("_selective_final3_argsort", "final3")]:
                stack.enter_context(patch.object(module, name, wrap(getattr(module,name), route)))
            output, _ = timed(lambda: engine.render(view), evaluate=True)
        return output, actual or [("mlx", None)]
    routes = {}
    for mode in names:
        output, actual = trace(engines[mode], cam)
        equal(reference, output)
        routes[mode] = dict(actual=actual, counts=dict(engines[mode]._renderer.last_frame_counts))
    adaptive = engines["adaptive"]
    decisions = list(adaptive.last_sort_decisions)
    for decision, (route, width) in zip(decisions, routes["adaptive"]["actual"], strict=True):
        expected = (("final3", 512) if decision["mode"].endswith("512")
                    else ("radix4", None) if decision["capacity"] >= 2**22 else ("mlx", None))
        assert (route, width) == expected
    unknown = ExperimentalAdaptiveRenderer(scene(1024))
    stable = api.PersistentCppCompatRenderer(scene(1024))
    transitions = []
    for renderer in (adaptive, engines["auto"]):
        renderer.update(scene(4096))
    for tag, view in [("small", camera(640, 360, 2)), ("fullhd", camera(1920,1080,5)),
                      ("highres", camera(2560,1440,4))]:
        candidate, actual = trace(adaptive, view)
        equal(candidate, engines["auto"].render(view))
        equal(unknown.render(view), stable.render(view))
        assert all(row["mode"] == "auto" for row in unknown.last_sort_decisions)
        transitions.append({"camera": tag, "actual": actual,
                            "decisions": list(adaptive.last_sort_decisions),
                            "unknown_decisions": list(unknown.last_sort_decisions)})
    return {"cold_order": cold_order, "cold": cold, "samples": samples, "routing": routes,
            "adaptive_decisions": decisions, "transitions": transitions,
            "median_ms": {m:statistics.median(row["ms"][m] for row in samples) for m in names},
            "paired_adaptive_minus_auto_ms": [row["ms"]["adaptive"] - row["ms"]["auto"] for row in samples],
            "host_eligibility_matrix_cases": 96}
