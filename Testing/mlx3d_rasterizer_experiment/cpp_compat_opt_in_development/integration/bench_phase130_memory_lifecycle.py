"""Memory timeline for repeated rendering, transitions and scene replacement."""
import gc
import resource

def run(context):
    mx,np,api,gs,cam,timed,equal = context
    from bench_phase49_breakdown import scene, camera
    rows = []
    def sample(label, engine=None):
        counters = dict(active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory(),
                        cache_bytes=mx.get_cache_memory(),
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if counters["active_bytes"] + counters["cache_bytes"] > 13 * 2**30:
            raise MemoryError("active plus allocator cache exceeded the 13 GiB experiment cap")
        rows.append(dict(label=label, **counters, counts=(
            dict(getattr(engine._renderer, "last_frame_counts", {}) or {}) if engine else {})))
    sample("before_constructor")
    engine = api.PersistentCppCompatRenderer(gs)
    sample("after_constructor", engine)
    reference, _ = timed(lambda: engine.render(cam), evaluate=True)
    sample("first_frame", engine)
    smaller_scene = scene(min(4096, gs.means.shape[0]))
    small_camera = camera(640, 360, 1)
    for block in range(6):
        engine.update(smaller_scene)
        timed(lambda: engine.render(small_camera), evaluate=True)
        sample(f"{block}_after_small_replacement", engine)
        engine.update(gs)
        image, _ = timed(lambda: engine.render(cam), evaluate=True)
        equal(reference, image)
        sample(f"{block}_after_original_replacement", engine)
    del image, reference, engine
    gc.collect()
    sample("after_release")
    return {"timeline": rows, "scope": "Process-wide counters; Darwin RSS is high water. No cache clear."}
