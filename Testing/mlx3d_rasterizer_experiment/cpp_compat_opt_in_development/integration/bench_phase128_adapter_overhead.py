"""Run by run_overnight_renderer.py in the pinned disposable runtime."""
import math
import statistics

def run(context):
    mx, np, api, gs, cam, timed, equal = context
    from mlx_gs.renderer.renderer import render
    engines = api.PersistentCppCompatRenderer
    results = {key: [] for key in ("scene_metadata_cpu", "activation_eval",
               "full_scene_args", "camera_metadata_cpu", "matrix_finite_gpu",
               "camera_current", "camera_host_candidate", "renderer_construction",
               "persistent_update", "public_one_shot", "direct_one_shot")}
    prior = engines(gs)
    fields = ("means", "scales", "quaternions", "opacities", "sh_coeffs")
    fields_map = {name: getattr(gs, name) for name in fields}
    def activated():
        return (mx.exp(gs.scales), mx.sigmoid(gs.opacities),
                mx.clip(gs.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5, 0, 1))
    def validate_metadata():
        for name in fields:
            api._real_array(fields_map[name], name)
        return [value.shape for value in fields_map.values()]
    def camera_candidate():
        # Isolate matrix checking. Keep dimensions/intrinsics handled by the
        # unchanged adapter in production. This is an observational candidate.
        if not all(math.isfinite(v) for row in cam.W2C.tolist() for v in row):
            raise ValueError("Camera W2C must be finite")
    candidate_checks = []
    for matrix in [cam.W2C, mx.zeros((4,4)), mx.full((4,4), float("nan")),
                   mx.full((4,4), float("inf"))]:
        current = bool(mx.all(mx.isfinite(matrix)).item())
        proposed = all(math.isfinite(v) for row in matrix.tolist() for v in row)
        assert current == proposed
        candidate_checks.append(current)
    def measure(key, fn, eval_output=False):
        value, ms = timed(fn, evaluate=eval_output)
        results[key].append(ms)
        return value
    for block in range(6):
        measure("scene_metadata_cpu", validate_metadata)
        measure("activation_eval", activated, True)
        args = measure("full_scene_args", lambda: api._scene_args(gs))
        measure("camera_metadata_cpu", lambda: (int(cam.W), int(cam.H), float(cam.fx), cam.W2C.shape))
        alternatives = [("matrix_finite_gpu", lambda: mx.all(mx.isfinite(cam.W2C)).item()),
                        ("camera_host_candidate", camera_candidate)]
        for key, fn in alternatives[::1 if block % 2 == 0 else -1]:
            measure(key, fn)
        measure("camera_current", lambda: api._camera(cam))
        measure("renderer_construction", lambda: api._renderer_class()(
            **args, geometry_policy="cpp_compat", alpha_mode="exponential"))
        measure("persistent_update", lambda: prior.update(gs))
        calls = [("direct_one_shot", lambda: api.render_cpp_compat(gs, cam)),
                 ("public_one_shot", lambda: render(gs, cam, rasterizer_type="mlx3d_cpp_compat"))]
        outputs = [measure(key, fn, True) for key, fn in calls[::1 if block % 2 == 0 else -1]]
        equal(*outputs)
    return {"samples_ms": results, "medians_ms": {k: statistics.median(v) for k,v in results.items()},
            "matrix_finite_parity": candidate_checks,
            "scope": "Metadata probe is a subset. Paired matrix finite checks only; no candidate adapter promotion."}
