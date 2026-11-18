#!/usr/bin/env python3
"""Independent stable-order, sentinel and fallback tests for Phase 85."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from radix_final3_experiment import (SENTINEL, VARIANTS, backend,
    final_pass_plan, stable_final3_argsort, renderer_sort_override)

PRODUCTION = ROOT / 'Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py'


def run_correctness(large_n=1000003):
    before = hashlib.sha256(PRODUCTION.read_bytes()).hexdigest()
    module, baseline = backend()
    rng = np.random.default_rng(85001)
    rows = []
    def check(raw, tiles, label, array=None):
        keys = mx.array(raw) if array is None else array
        expected = np.argsort(raw, kind='stable').astype(np.int32)
        reference = baseline(keys, tiles)
        mx.eval(reference)
        np.testing.assert_array_equal(np.asarray(reference), expected)
        for variant in VARIANTS:
            order = stable_final3_argsort(keys, tiles, variant=variant)
            mx.eval(order)
            np.testing.assert_array_equal(np.asarray(order), expected)
            np.testing.assert_array_equal(np.asarray(keys), raw)
        rows.append({'tiles': tiles, 'size': len(raw), 'pattern': label,
                     'specialized': final_pass_plan(tiles)['eligible']})
    sizes = (0, 1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 4096, 65537)
    for tiles in (1, 2, 255, 256, 4095, 4096, 8160, 8192, 8193, 65536):
        for n in sizes:
            raw = rng.integers(0, tiles, n, dtype=np.uint32)
            check(raw, tiles, 'random_unmasked')
            masked = raw.copy()
            masked[::7] = np.uint32(SENTINEL)
            check(masked, tiles, 'random_masked')
        for n in (257, 4096):
            check(np.full(n, SENTINEL, dtype=np.uint32), tiles, 'all_sentinel')
            check(np.full(n, tiles-1, dtype=np.uint32), tiles, 'all_equal_real')
            raw = np.sort(rng.integers(0, tiles, n, dtype=np.uint32))
            check(raw, tiles, 'sorted')
            check(raw[::-1].copy(), tiles, 'reverse_sorted')
    # Explicit Phase 84 counterexample: high real tiles MUST precede sentinels.
    collision = np.array([SENTINEL, 4096, 0, 8159, SENTINEL, 4095,
                          4096, SENTINEL, 8159, 0, 4095], dtype=np.uint32)
    check(np.tile(collision, 47), 8160, 'binary_alias_counterexample')
    # Noncontiguous input and widely interleaved duplicates.
    raw = rng.integers(0, 8160, 4098, dtype=np.uint32)
    raw[::13] = np.uint32(SENTINEL)
    check(raw[::2].copy(), 8160, 'strided_input', mx.array(raw)[::2])
    for tiles in (4096, 8160, 8192):
        for masked in (False, True):
            raw = rng.integers(0, tiles, large_n, dtype=np.uint32)
            if masked: raw[::7] = np.uint32(SENTINEL)
            check(raw, tiles, 'large_masked' if masked else 'large_unmasked')
    for tiles in (0, -1, SENTINEL):
        try: final_pass_plan(tiles)
        except ValueError: pass
        else: raise AssertionError('Invalid tile count accepted')
    for keys in (mx.array([1], dtype=mx.int32), mx.zeros((1,1), dtype=mx.uint32)):
        try: stable_final3_argsort(keys, 8160)
        except ValueError: pass
        else: raise AssertionError('Invalid shape/dtype accepted')
    try:
        with renderer_sort_override('indices_only'):
            raise RuntimeError('intentional restoration check')
    except RuntimeError:
        pass
    assert module._radix_tile_argsort is baseline
    assert hashlib.sha256(PRODUCTION.read_bytes()).hexdigest() == before
    result = {'phase': 85, 'status': 'PASS', 'case_count': len(rows),
              'variants': list(VARIANTS), 'cases': rows,
              'production_sha256': before,
              'oracle': 'exact NumPy stable permutation and unchanged production permutation',
              'checks': ['sentinel relative order', 'real tie order', 'input unchanged',
                         'empty and partial blocks', 'unsupported-domain fallback',
                         'strided input', 'exception-safe restoration', 'production file unchanged']}
    print('PASS Phase 85 exact stable ordering:', len(rows), 'cases ×', len(VARIANTS), 'variants', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--large-n', type=int, default=1000003)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('output exists')
    if a.large_n < 65537: p.error('large-n must be at least 65537')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(run_correctness(a.large_n), indent=2)+'\n')


if __name__ == '__main__': main()
