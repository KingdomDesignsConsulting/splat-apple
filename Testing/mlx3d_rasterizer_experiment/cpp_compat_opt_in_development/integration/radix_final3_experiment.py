"""Phase 85 only: stable, sentinel-safe three-category final radix pass.

Never imported by production. The first radix digits reuse the hash-verified
production kernels. The final digit is eligible only when valid real keys have
one or two distinct high digits. UINT_MAX always gets its own final category.
Input contract, like production: 1D uint32 keys in [0,num_tiles) or UINT_MAX.
"""
from contextlib import contextmanager
from functools import lru_cache
import operator
import sys

import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import _renderer_class

SENTINEL = 0xFFFFFFFF
VARIANTS = ("with_keys", "indices_only")


@lru_cache(maxsize=1)
def backend():
    cls = _renderer_class()
    module = sys.modules[cls.__module__]
    return module, module._radix_tile_argsort


def final_pass_plan(num_tiles):
    num_tiles = operator.index(num_tiles)
    if not 1 <= num_tiles < SENTINEL:
        raise ValueError("num_tiles must be in [1, UINT_MAX)")
    shift = ((num_tiles.bit_length() - 1) // 4) * 4
    real_categories = ((num_tiles - 1) >> shift) + 1
    return {"shift": shift, "real_categories": real_categories,
            "eligible": real_categories <= 2,
            "passes": shift // 4 + 1}


_FINAL_RANK = mx.fast.metal_kernel(
    name="phase85_final3_rank_hist",
    input_names=["keys", "params"], output_names=["ranks", "hist"],
    source=r"""
        uint i = thread_position_in_grid.x;
        uint local = i & 255u;
        uint block = i >> 8;
        uint lane = local & 31u;
        uint subgroup = local >> 5;
        uint n = (uint)params[0], blocks = (uint)params[1];
        uint key = i < n ? keys[i] : 0u;
        uint category = key == 0xffffffffu ? 2u : (key >> params[2]);
        threadgroup uint subgroup_counts[24];
        uint rank = 0;
        // All lanes, including padded lanes, reach every SIMD op and barrier.
        for (uint bin = 0; bin < 3; ++bin) {
            uint match = (uint)(i < n && category == bin);
            uint before = simd_prefix_exclusive_sum(match);
            uint count = simd_sum(match);
            if (lane == 0) subgroup_counts[subgroup * 3 + bin] = count;
            if (match) rank = before;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local < 3) {
            uint total = 0;
            for (uint sg = 0; sg < 8; ++sg) total += subgroup_counts[sg * 3 + local];
            hist[local * blocks + block] = (int)total;
        }
        if (i < n) {
            for (uint sg = 0; sg < subgroup; ++sg) rank += subgroup_counts[sg * 3 + category];
            ranks[i] = (int)rank;
        }
    """,
)

_SCATTER_INPUTS = ["keys", "indices", "ranks", "prefix", "starts", "params"]
_SCATTER_COMMON = r"""
    uint i = thread_position_in_grid.x;
    if (i >= (uint)params[0]) return;
    uint block = i >> 8, blocks = (uint)params[1];
    uint key = keys[i];
    uint category = key == 0xffffffffu ? 2u : (key >> params[2]);
    uint dst = (uint)(starts[category] + prefix[category * blocks + block] + ranks[i]);
    out_indices[dst] = indices[i];
"""
_FINAL_INDICES = mx.fast.metal_kernel(
    name="phase85_final3_scatter_indices", input_names=_SCATTER_INPUTS,
    output_names=["out_indices"], source=_SCATTER_COMMON,
)
_FINAL_WITH_KEYS = mx.fast.metal_kernel(
    name="phase85_final3_scatter_keys_indices", input_names=_SCATTER_INPUTS,
    output_names=["out_keys", "out_indices"],
    source=_SCATTER_COMMON + "\n out_keys[dst] = key;\n",
)


def stable_final3_argsort(keys, num_tiles, *, variant="indices_only"):
    """Return exactly the production stable permutation, with a cheaper last pass.

    Earlier stable LSD passes order the lower bits. The last stable partition
    separates high digits 0, 1, and sentinel; it never aliases sentinel with 1.
    Each category preserves preceding lane and block order, including sentinel
    input order. Unsupported tile domains delegate to the unchanged production
    sorter. No data-dependent host readback or sentinel-fraction assumption.
    """
    plan = final_pass_plan(num_tiles)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if keys.ndim != 1 or keys.dtype != mx.uint32:
        raise ValueError("keys must be a 1D uint32 array")
    n = int(keys.size)
    if n >= 2**31:
        raise OverflowError("signed 32-bit permutation limit exceeded")
    module, production_sort = backend()
    if n == 0 or not plan["eligible"]:
        return production_sort(keys, num_tiles)
    blocks = (n + 255) // 256
    indices = mx.arange(n, dtype=mx.int32)
    for shift in range(0, plan["shift"], 4):
        params = mx.array([n, blocks, shift], dtype=mx.int32)
        ranks, hist = module._radix_rank_hist_kernel(
            inputs=[keys, params], output_shapes=[(n,), (16, blocks)],
            output_dtypes=[mx.int32, mx.int32],
            grid=(blocks * 256, 1, 1), threadgroup=(256, 1, 1))
        prefix = mx.cumsum(hist, axis=1) - hist
        totals = mx.sum(hist, axis=1)
        starts = mx.cumsum(totals) - totals
        keys, indices = module._radix_scatter_kernel(
            inputs=[keys, indices, ranks, prefix, starts, params],
            output_shapes=[(n,), (n,)], output_dtypes=[mx.uint32, mx.int32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1))
    params = mx.array([n, blocks, plan["shift"]], dtype=mx.int32)
    ranks, hist = _FINAL_RANK(
        inputs=[keys, params], output_shapes=[(n,), (3, blocks)],
        output_dtypes=[mx.int32, mx.int32],
        grid=(blocks * 256, 1, 1), threadgroup=(256, 1, 1))
    prefix = mx.cumsum(hist, axis=1) - hist
    totals = mx.sum(hist, axis=1)
    starts = mx.cumsum(totals) - totals
    inputs = [keys, indices, ranks, prefix, starts, params]
    if variant == "with_keys":
        return _FINAL_WITH_KEYS(
            inputs=inputs, output_shapes=[(n,), (n,)],
            output_dtypes=[mx.uint32, mx.int32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1))[1]
    return _FINAL_INDICES(
        inputs=inputs, output_shapes=[(n,)], output_dtypes=[mx.int32],
        grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]


@contextmanager
def renderer_sort_override(variant):
    """Scoped, process-local experiment hook; restore even after exceptions.

    Production files/defaults and installed packages are never changed. Render
    completion must occur inside this context. It is not a concurrent API.
    """
    module, _ = backend()
    original = module._radix_tile_argsort
    def candidate(keys, tiles):
        return stable_final3_argsort(keys, tiles, variant=variant)
    module._radix_tile_argsort = candidate
    try:
        yield
    finally:
        module._radix_tile_argsort = original
