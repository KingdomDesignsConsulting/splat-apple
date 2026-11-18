"""Process-first timings, with the shared driver compilation cache left intact."""
def run(context):
    _, _, api, gs, cam, timed, equal = context
    _, verification = timed(api._renderer_class)
    engine, constructor = timed(lambda: api.PersistentCppCompatRenderer(gs))
    first, cold = timed(lambda: engine.render(cam), evaluate=True)
    capacity = dict(engine._renderer.last_frame_counts)
    warmed = []
    for _ in range(5):
        image, ms = timed(lambda: engine.render(cam), evaluate=True)
        equal(first, image)
        warmed.append(ms)
    one_shot, one_shot_ms = timed(lambda: api.render_cpp_compat(gs, cam), evaluate=True)
    equal(first, one_shot)
    return dict(source_verification_and_load_ms=verification, constructor_ms=constructor,
                first_render_ms=cold, cold_capacity=capacity, warmed_ms=warmed,
                post_warm_one_shot_ms=one_shot_ms,
                scope="Process-first with existing system Metal caches. Render includes first sizing and synchronized completion.")
