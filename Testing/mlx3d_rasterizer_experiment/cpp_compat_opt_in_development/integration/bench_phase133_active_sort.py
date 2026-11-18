"""Test exact active-sized sorting against the default capacity policy."""
import statistics

def run(context, block):
    mx, np, api, gs, cam, timed, equal = context
    from bench_phase49_breakdown import camera
    import sys
    from unittest.mock import patch
    modes = ("auto", "active")
    native = api._renderer_class()
    module = sys.modules[native.__module__]
    engines = {}
    cold = {}
    cold_order = modes[::1 if block in (0, 3) else -1]
    images = {}
    for mode in cold_order:
        engine, constructor = timed(lambda: api.PersistentCppCompatRenderer(gs, duplicate_sort_mode=mode))
        image, first = timed(lambda: engine.render(cam), evaluate=True)
        engines[mode] = engine
        images[mode] = image
        cold[mode] = {"constructor_ms": constructor, "first_ms": first}
    equal(images["active"], images["auto"])
    rows = []
    import random
    rng = random.Random(13300 + block)
    for round_index in range(4):
        view = camera(cam.W, cam.H, round_index + block)
        # Check warmed behavior on the same camera following both orderings.
        for engine in engines.values():
            timed(lambda: engine.render(view), evaluate=True)
        orders = [modes, modes[::-1]]
        rng.shuffle(orders)
        for order in orders:
            samples = {}
            for mode in order:
                images[mode], elapsed = timed(lambda: engines[mode].render(view), evaluate=True)
                samples[mode] = {"ms": elapsed, "counts": dict(engines[mode]._renderer.last_frame_counts)}
            equal(images["active"], images["auto"])
            rows.append({"order": order, "samples": samples, "camera_index": round_index + block})
    route = {}
    for mode, engine in engines.items():
        calls = []
        radix = module._radix_tile_argsort
        def traced(keys, tiles):
            calls.append({"sort": "radix4", "length": keys.shape[0], "tiles": tiles})
            return radix(keys, tiles)
        with patch.object(module, "_radix_tile_argsort", traced):
            images[mode], _ = timed(lambda: engine.render(cam), evaluate=True)
        counts = engine._renderer.last_frame_counts
        assert (not calls and counts["sorted"] < 2**22) or calls[-1]["length"] == counts["sorted"]
        if mode == "active":
            assert counts["sorted"] == max(counts["active"], 1)
        route[mode] = calls or [{"sort": "mlx", "length": counts["sorted"]}]
    equal(images["active"], images["auto"])
    return {"samples": rows, "dispatch": route, "cold": cold, "cold_order": cold_order,
            "seed": 13300 + block,
            "median_ms": {mode: statistics.median(row["samples"][mode]["ms"] for row in rows)
                          for mode in modes},
            "paired_delta_ms": [row["samples"]["active"]["ms"] - row["samples"]["auto"]["ms"]
                                for row in rows],
            "hypothesis": "Fewer sorted slots and no sentinel tail mask may repay the preflight readback."}
