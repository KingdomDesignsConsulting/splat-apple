#!/usr/bin/env python3
"""Exact stable-order validation for the Metal four-bit radix experiment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class


def main():
    cls=_renderer_class()
    module=sys.modules[cls.__module__]
    rng=np.random.default_rng(6060)
    cases=0
    for tiles in (1,17,8192,65536):
        for n in (0,1,31,32,255,256,257,4096,65537,1000003):
            keys=rng.integers(0,tiles,n,dtype=np.uint32)
            keys[::17]=np.uint32(0xffffffff)
            array=mx.array(keys)
            actual=module._radix_tile_argsort(array,tiles)
            expected=np.argsort(keys,kind='stable').astype(np.int32)
            mx.eval(actual)
            np.testing.assert_array_equal(np.asarray(actual),expected)
            cases+=1
    print('PASS exact stable radix order:',cases,'cases',flush=True)


if __name__=='__main__':main()
