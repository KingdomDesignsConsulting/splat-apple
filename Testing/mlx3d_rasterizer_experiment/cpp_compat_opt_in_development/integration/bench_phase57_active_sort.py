#!/usr/bin/env python3
"""Active-only duplicate sorting: paired parity, varied poses, timing and breakdown.

Synchronized phase timings include dispatch and forced synchronization. They
are diagnostic wall times, not hardware GPU counters or additive normal frame
costs. Paired end-to-end timings run without instrumentation or image readback.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene, memory


def pose(w, h, i):
    # 45 combinations: translation, rotation and focal length. Unlike the
    # historical camera helper, these poses change duplicate demand widely.
    angle = ((i % 5) - 2) * 0.12
    c, s = np.cos(angle), np.sin(angle)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    w2c[:3, 3] = [((i // 5) % 3 - 1) * 0.65, ((i % 3) - 1) * 0.2, 0.0]
    f = min(w, h) * [0.8, 1.2, 1.8][(i // 15) % 3]
    return SimpleNamespace(W=w, H=h, fx=f, fy=f, cx=w/2, cy=h/2, W2C=mx.array(w2c))


def timed(engine, cam):
    start = time.perf_counter()
    image = engine.render(cam)
    mx.eval(image)
    return (time.perf_counter()-start)*1000, image


@contextmanager
def breakdown(engine):
    """Restore every patched function even if a diagnostic render fails."""
    module = importlib.import_module(engine._renderer.__class__.__module__)
    samples = defaultdict(list)
    patched = []
    inside_radix = [False]
    def patch(owner, name, label):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            if inside_radix[0]:
                return original(*args, **kwargs)
            start = time.perf_counter()
            value = original(*args, **kwargs)
            mx.eval(value)
            samples[label].append((time.perf_counter()-start)*1000)
            return value
        patched.append((owner, name, original))
        setattr(owner, name, wrapped)
    try:
        for name, label in [('_cpp_compat_geometry_kernel', 'geometry_projection'),
                            ('_expand_kernel', 'duplicate_expansion'),
                            ('_ranges_kernel', 'tile_ranges'),
                            ('_ranges_binary_kernel', 'tile_ranges'),
                            ('_cpp_compat_raster_kernel', 'compositing')]:
            if hasattr(module, name):
                patch(module, name, label)
        original_sort = mx.argsort
        def sort(array, *args, **kwargs):
            label = 'tile_key_sort' if array.dtype == mx.uint32 else 'global_depth_sort'
            start = time.perf_counter()
            value = original_sort(array, *args, **kwargs)
            mx.eval(value)
            samples[label].append((time.perf_counter()-start)*1000)
            return value
        patched.append((mx, 'argsort', original_sort))
        mx.argsort = sort
        patch(mx, 'cumsum', 'prefix_sum_and_gather')
        patch(mx, 'sum', 'active_count_reduce')
        if hasattr(module, '_radix_tile_argsort'):
            original_radix = module._radix_tile_argsort
            def radix(*args, **kwargs):
                start = time.perf_counter()
                inside_radix[0] = True
                try:
                    value = original_radix(*args, **kwargs)
                    mx.eval(value)
                finally:
                    inside_radix[0] = False
                samples['tile_key_sort'].append((time.perf_counter()-start)*1000)
                return value
            patched.append((module, '_radix_tile_argsort', original_radix))
            module._radix_tile_argsort = radix
        yield samples
    finally:
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=700000)
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--survey', type=int, default=45)
    p.add_argument('--profile-rounds', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('Refusing to overwrite output')
    if not 1 <= a.n <= 800000 or min(a.width, a.height, a.rounds, a.survey) < 1:
        p.error('Invalid workload')
    gs = scene(a.n)
    engines = {}
    setup = {}
    for mode in ('capacity', 'active'):
        start = time.perf_counter()
        engines[mode] = PersistentCppCompatRenderer(gs, duplicate_sort_mode=mode)
        mx.eval(engines[mode]._renderer.cov3d)
        setup[mode] = (time.perf_counter()-start)*1000
    rows = []
    rng = np.random.default_rng(5757)
    for i in range(-3, a.rounds):
        cam = pose(a.width, a.height, (i+3) % 45)
        values, times, counts = {}, {}, {}
        for mode in rng.permutation(['capacity', 'active']):
            times[mode], values[mode] = timed(engines[mode], cam)
            counts[mode] = dict(engines[mode]._renderer.last_frame_counts)
        if not np.array_equal(np.asarray(values['capacity']), np.asarray(values['active'])):
            raise AssertionError(f'Image parity failed at frame {i}')
        if counts['capacity']['active'] != counts['active']['active']:
            raise AssertionError('Active counts differ')
        if i >= 0:
            rows.append({'pose': (i+3) % 45, 'ms': times, 'counts': counts})
        print('PASS paired', i, times, flush=True)
    # Force tiny initial capacity, then demand growth across 45 varied poses.
    for engine in engines.values(): engine._renderer._capacity = 1
    survey = []
    for i in range(a.survey):
        cam = pose(a.width, a.height, i)
        images = {mode: engine.render(cam) for mode, engine in engines.items()}
        mx.eval(images)
        if not np.array_equal(np.asarray(images['capacity']), np.asarray(images['active'])):
            raise AssertionError(f'Survey parity failed at pose {i}')
        row = {'pose': i, 'counts': {m: dict(e._renderer.last_frame_counts) for m,e in engines.items()}}
        if row['counts']['active']['sorted'] != max(1, row['counts']['active']['active']):
            raise AssertionError('Active path sorted slack')
        survey.append(row)
        print('PASS survey', i, row['counts']['active'], flush=True)
    # Fully culled image and subsequent scene replacement are regression gates.
    hidden = pose(a.width, a.height, 0)
    hidden.W2C = hidden.W2C.at[2,3].add(-100)
    images = [engines[m].render(hidden) for m in engines]
    mx.eval(images)
    if not all(np.array_equal(np.asarray(images[0]), np.asarray(v)) for v in images):
        raise AssertionError('Fully culled parity')
    for engine in engines.values(): engine.update(gs)
    images = [engines[m].render(pose(a.width, a.height, 2)) for m in engines]
    mx.eval(images)
    if not all(np.array_equal(np.asarray(images[0]), np.asarray(v)) for v in images):
        raise AssertionError('Scene update parity')
    phases = {}
    for mode, engine in engines.items():
        with breakdown(engine) as samples:
            for i in range(a.profile_rounds):
                engine.render(pose(a.width, a.height, 3+i))
        phases[mode] = {key: {'median_ms': float(np.median(v)), 'raw_ms': v} for key,v in samples.items()}
    report = {'phase': 57, 'n': a.n, 'resolution': [a.width, a.height],
              'rounds': a.rounds, 'setup_covariance_ms': setup,
              'median_ms': {m: float(np.median([r['ms'][m] for r in rows])) for m in engines},
              'paired_active_over_capacity': float(np.median([r['ms']['active']/r['ms']['capacity'] for r in rows])),
              'samples': rows, 'camera_survey': survey, 'synchronized_breakdown': phases,
              'memory_process_after': memory(),
              'correctness': 'bitwise image parity; tiny capacity growth; full cull; scene update',
              'limitations': ['Synthetic workload, parity against existing capacity path, not original C++.',
                             'Breakdown includes dispatch and synchronization; not pure GPU timestamps.',
                             'Process memory includes both renderers and MLX allocator cache.']}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print('RESULT', report['median_ms'], 'paired ratio', report['paired_active_over_capacity'], flush=True)


if __name__ == '__main__': main()
