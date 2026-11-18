#!/usr/bin/env python3
"""Paired Phase 85 sort/renderer benchmarks; no production file/default changes."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from mlx3d.cameras import Camera as FastCamera
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene, camera, memory
from bench_phase57_active_sort import pose
import radix_final3_experiment as experiment

BASE = Path(__file__).resolve().parent
PRODUCTION_FILES = (BASE/'mlx3d/splatting/fast.py', ROOT/'mlx_gs/renderer/mlx3d_cpp_compat.py')


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in PRODUCTION_FILES}


def summarize(rows, names):
    medians = {name: float(np.median([r[name] for r in rows])) for name in names}
    paired = {}
    for name in names[1:]:
        ratios = np.array([r[name]/r[names[0]] for r in rows])
        rng = np.random.default_rng(85085)
        bootstrap = np.median(ratios[rng.integers(0, len(rows), (5000, len(rows)))], axis=1)
        paired[name] = {'median_ratio': float(np.median(ratios)),
                        'within_run_bootstrap_95pct': np.percentile(bootstrap, [2.5,97.5]).tolist()}
    return {'median_ms': medians, 'paired': paired, 'raw_ms': rows}


def sort_benchmark(slots, tiles, rounds):
    _, baseline = experiment.backend()
    rng = np.random.default_rng(85085)
    names = ['radix4', *experiment.VARIANTS]
    cases = []
    for n in slots:
        for masked in (False, True):
            x = mx.arange(n, dtype=mx.uint32)
            keys = (x*1664525+1013904223) % tiles
            if masked:
                keys = mx.where(x%7 == 0, mx.array(experiment.SENTINEL, dtype=mx.uint32), keys)
            mx.eval(keys)
            def run(name):
                start = time.perf_counter()
                order = (baseline(keys, tiles) if name == 'radix4' else
                         experiment.stable_final3_argsort(keys, tiles, variant=name))
                mx.eval(order)
                return order, (time.perf_counter()-start)*1000
            for _ in range(3):
                for name in rng.permutation(names).tolist(): run(name)
            rows = []
            for i in range(rounds):
                values, times = {}, {}
                for name in rng.permutation(names).tolist():
                    values[name], times[name] = run(name)
                for name in names[1:]:
                    # Exact comparison of EVERY large-array permutation entry.
                    if not bool(mx.all(values[name] == values['radix4']).item()):
                        raise AssertionError(('large permutation mismatch', n, masked, name, i))
                rows.append(times)
            case = {'slots': n, 'tiles': tiles, 'masked': masked,
                    'plan': experiment.final_pass_plan(tiles), **summarize(rows, names)}
            cases.append(case)
            print('PASS sort', n, 'masked', masked, case['median_ms'], case['paired'], flush=True)
    return {'cases': cases,
            'oracle': 'all full-sized permutation entries equal unchanged production radix4',
            'timing': 'three warmups, randomized paired order, GPU completion in timer, comparisons outside'}


def fast_camera(cam):
    w2c = cam.W2C
    return FastCamera(R=w2c[:3,:3], t=w2c[:3,3], fx=cam.fx, fy=cam.fy,
                      cx=cam.cx+0.5, cy=cam.cy+0.5, width=cam.W, height=cam.H)


def assert_render_equal(values, label):
    for field in ('image', 'alpha'):
        left = np.asarray(values['radix4'][field])
        right = np.asarray(values['final3'][field])
        if left.dtype != np.float32 or right.dtype != np.float32:
            raise AssertionError(('unexpected output dtype', label, field))
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise AssertionError(('nonfinite output', label, field))
        if not np.array_equal(left.view(np.uint32), right.view(np.uint32)):
            raise AssertionError(('image/alpha bit mismatch', label, field))


def render_benchmark(scenes, resolutions, rounds, variant):
    rng = np.random.default_rng(85185)
    names = ['radix4', 'final3']
    cases = []
    for n in scenes:
        gs = scene(n)
        for w,h in resolutions:
            tiles = ((w+15)//16)*((h+15)//16)
            engines = {name: PersistentCppCompatRenderer(gs, duplicate_sort_mode='capacity',
                tile_sort_impl='radix4', tile_range_mode='auto') for name in names}
            bg = mx.array([0.03, 0.05, 0.07], dtype=mx.float32)
            def paired(cam, label):
                values, times = {}, {}
                fc = fast_camera(cam)
                for name in rng.permutation(names).tolist():
                    context = (experiment.renderer_sort_override(variant) if name == 'final3' else nullcontext())
                    # Exclude the Python experiment hook from pipeline timing.
                    with context:
                        start = time.perf_counter()
                        values[name] = engines[name]._renderer.render(fc, background=bg)
                        mx.eval(values[name]['image'], values[name]['alpha'])
                        times[name] = (time.perf_counter()-start)*1000
                assert_render_equal(values, label)
                counts = {name: dict(e._renderer.last_frame_counts) for name,e in engines.items()}
                if counts['radix4'] != counts['final3']:
                    raise AssertionError(('different sizing/workload', label, counts))
                return times, counts
            fixed = camera(w,h,3)
            _, counts = paired(fixed, 'initialize')
            active = counts['radix4']['active']
            steady = {}
            for layout,capacity in [('exact_unmasked', max(active,1)),
                                    ('headroom_masked', active*105//100+1024)]:
                for engine in engines.values(): engine._renderer._capacity = capacity
                for i in range(3): paired(fixed, (layout,'warm',i))
                rows = []
                for i in range(rounds):
                    times, counts = paired(fixed, (layout,i))
                    if counts['radix4']['sorted'] != capacity:
                        raise AssertionError(('fixed workload changed', layout, counts))
                    rows.append(times)
                steady[layout] = {'active': active, 'capacity': capacity, **summarize(rows,names)}
            varied = []
            for i in (0,1,14,15,16,29,30,31,34,37,40,44):
                times, counts = paired(pose(w,h,i), ('pose',i))
                varied.append({'pose':i,'ms':times,'counts':counts['radix4']})
            # Real scene replacement, not an update with unchanged values.
            shifted = {key: getattr(gs,key) for key in
                       ('means','scales','quaternions','opacities','sh_coeffs')}
            shifted['means'] = shifted['means'] + mx.array([0.1,-0.07,0.0])
            shifted['sh_coeffs'] = shifted['sh_coeffs'] + 0.2
            for engine in engines.values(): engine.update(shifted)
            paired(fixed, 'changed_scene')
            hidden = camera(w,h,0)
            hidden.W2C = hidden.W2C.at[2,3].add(-100.0)
            paired(hidden, 'fully_culled')
            case = {'gaussians':n,'resolution':[w,h],'tiles':tiles,
                    'specialized':experiment.final_pass_plan(tiles)['eligible'],
                    'steady':steady,'varied':varied,
                    'varied_summary':summarize([row['ms'] for row in varied],names),
                    'image_alpha_bitwise_parity':'PASS','changed_scene':'PASS','fully_culled':'PASS'}
            cases.append(case)
            print('PASS render', n, [w,h], 'specialized', case['specialized'],
                  {k:v['median_ms'] for k,v in steady.items()}, flush=True)
            del engines
    return {'variant':variant,'cases':cases,
            'timing':'matched capacities and camera; image+alpha GPU completion in timer; context and bit checks outside'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['sort','render'], required=True)
    p.add_argument('--slots', default='1000003,8000003,33154238')
    p.add_argument('--tiles', type=int, default=8160)
    p.add_argument('--scenes', default='4096,100000,700000')
    p.add_argument('--resolutions', default='1280x720,1920x1080')
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--variant', choices=experiment.VARIANTS, default='indices_only')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('output exists')
    slots = [int(x) for x in a.slots.split(',')]
    scenes = [int(x) for x in a.scenes.split(',')]
    resolutions = [tuple(map(int,x.split('x'))) for x in a.resolutions.split(',')]
    if a.rounds < 6 or not slots or min(slots)<1 or max(slots)>=2**31:
        p.error('invalid rounds/slots')
    if not scenes or min(scenes)<1 or max(scenes)>800000 or any(min(x)<1 or len(x)!=2 for x in resolutions):
        p.error('invalid scenes/resolutions')
    experiment.final_pass_plan(a.tiles)
    before = source_hashes()
    start = time.perf_counter()
    result = (sort_benchmark(slots,a.tiles,a.rounds) if a.mode=='sort' else
              render_benchmark(scenes,resolutions,a.rounds,a.variant))
    assert source_hashes() == before
    module, baseline = experiment.backend()
    assert module._radix_tile_argsort is baseline
    report = {'phase':85,'mode':a.mode,'status':'PASS','rounds':a.rounds,
              'elapsed_seconds':time.perf_counter()-start,
              'production_sources_sha256':before,
              'experiment_sha256':hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest(),
              'environment':{'python':platform.python_version(),'macos':platform.mac_ver()[0],
                  **{name:importlib.metadata.version(name) for name in ('mlx','numpy','mlx3d')}},
              **result, 'process_memory_after':memory(),
              'notes':['Production files, defaults and source-integrity checks unchanged.',
                       'Synthetic scenes; renderer parity is against current production, not a new C++ equivalence claim.',
                       'Wall timings include dispatch and synchronization; not hardware timestamps.',
                       'Warm steady-state experiment; no cold-start claim.',
                       'Within-run paired bootstrap intervals are exploratory; do not cover between-run/environment uncertainty.',
                       'Process memory includes all compared paths and allocator cache.']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT Phase 85',a.mode,'PASS saved',a.out,flush=True)


if __name__ == '__main__': main()
