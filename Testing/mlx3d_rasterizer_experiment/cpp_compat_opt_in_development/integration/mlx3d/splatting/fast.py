"""Fast forward-only Gaussian Splatting rasterization.

A viewing-oriented rendering path in the spirit of
`fast-gaussian-rasterization <https://github.com/dendenxu/fast-gaussian-rasterization>`_,
which replaces the training rasterizer's generic pipeline with a
hardware-style "geometry shader" flow: compute everything per Gaussian once,
sort back-to-front once, then splat. This Metal/MLX adaptation keeps the
tile-based compositing (Metal compute has no hardware blend stage) but applies
the same ideas end to end:

- **One fused geometry kernel** replaces ~60 elementwise MLX ops: camera
  transform, EWA projection from a *cached* 3D covariance, conic, culling,
  and tile-bounding-box math run in a single pass per Gaussian.
- **Cross-frame caching** (their ``OptimizedGaussians`` TODO): activations
  (``exp`` / ``sigmoid``), the 3D covariance, and SH-evaluated colors are
  computed once and reused; colors refresh only when the camera has moved
  enough to matter.
- **Cheaper sorting**: Gaussians are globally depth-sorted once per frame
  (N keys), so tile duplicates are emitted already depth-ordered and only a
  *stable 32-bit* tile-key sort remains — instead of a 64-bit
  ``(tile, depth)`` composite sort over every duplicate.
- **Selectable duplicate sizing**: capacity/adaptive/auto modes submit one
  lazy frame graph after initial sizing. Adaptive mode reserves 5% headroom
  and shrinks a stale high-water mark for the next frame; any overflow grows
  and rerenders the current frame. Auto retains the original policy below
  1M duplicates. Experimental active/bucketed modes read a scalar mid-frame.
- **Forward-only compositing kernel**: no backward bookkeeping
  (``n_contrib``), half-precision splat colors (quantization error < 0.1%;
  conics stay fp32 to survive near-camera footprints), and an adjustable early-termination
  transmittance for extra speed.

No backward pass — use :func:`~mlx3d.splatting.render_gaussians` for
training. The natural consumers are viewers, flythrough export, benchmark
loops, and dynamic (4D) sequence playback via :meth:`FastGaussianRenderer.update`.

Example:
    >>> from mlx3d.splatting import GaussianModel, FastGaussianRenderer
    >>> model = GaussianModel.load_ply("point_cloud.ply")
    >>> renderer = FastGaussianRenderer(model)
    >>> out = renderer.render(camera)          # {"image", "alpha"}
"""

from __future__ import annotations

import mlx.core as mx

from ..cameras import Camera
from .sh import eval_sh
from .tiles import TILE_SIZE

__all__ = ["FastGaussianRenderer", "render_gaussians_fast"]

_BLOCK = TILE_SIZE * TILE_SIZE
_SENTINEL = 0xFFFFFFFF  # padding tile key; sorts after every real tile

# --------------------------------------------------------------------------
# Kernel 1 — "geometry shader": everything per Gaussian, fused in one pass.
# --------------------------------------------------------------------------
_GEOMETRY_SRC = """
    const uint g = thread_position_in_grid.x;
    const uint N = (uint)iparams[0];
    if (g >= N) return;

    const int width = iparams[1];
    const int height = iparams[2];
    const int tiles_x = iparams[3];
    const int tiles_y = iparams[4];
    const bool antialias = iparams[5] != 0;

    // fparams: R(9) t(3) fx fy cx cy tanfx tanfy blur znear
    const float fx = fparams[12], fy = fparams[13];
    const float cx = fparams[14], cy = fparams[15];
    const float tanfx = fparams[16], tanfy = fparams[17];
    const float blur = fparams[18], znear = fparams[19];

    const float px = means[3 * g + 0];
    const float py = means[3 * g + 1];
    const float pz = means[3 * g + 2];

    // world -> camera
    const float x = fparams[0] * px + fparams[1] * py + fparams[2] * pz + fparams[9];
    const float y = fparams[3] * px + fparams[4] * py + fparams[5] * pz + fparams[10];
    const float z = fparams[6] * px + fparams[7] * py + fparams[8] * pz + fparams[11];
    const float z_safe = metal::max(z, 1e-6f);
    const float inv_z = 1.0f / z_safe;

    const float u = fx * x * inv_z + cx;
    const float v = fy * y * inv_z + cy;

    // EWA Jacobian with padded-frustum clamping (matches the training path).
    const float txc = metal::clamp(x * inv_z, -1.3f * tanfx, 1.3f * tanfx) * z_safe;
    const float tyc = metal::clamp(y * inv_z, -1.3f * tanfy, 1.3f * tanfy) * z_safe;
    const float inv_z2 = inv_z * inv_z;
    const float j00 = fx * inv_z;
    const float j02 = -fx * txc * inv_z2;
    const float j11 = fy * inv_z;
    const float j12 = -fy * tyc * inv_z2;

    // T = J @ R (two rows of a 2x3).
    const float t00 = j00 * fparams[0] + j02 * fparams[6];
    const float t01 = j00 * fparams[1] + j02 * fparams[7];
    const float t02 = j00 * fparams[2] + j02 * fparams[8];
    const float t10 = j11 * fparams[3] + j12 * fparams[6];
    const float t11 = j11 * fparams[4] + j12 * fparams[7];
    const float t12 = j11 * fparams[5] + j12 * fparams[8];

    // cov2d = T @ Sigma @ T^T from the cached upper-triangular 3D covariance.
    const float s00 = cov3d[6 * g + 0], s01 = cov3d[6 * g + 1], s02 = cov3d[6 * g + 2];
    const float s11 = cov3d[6 * g + 3], s12 = cov3d[6 * g + 4], s22 = cov3d[6 * g + 5];
    const float w0x = t00 * s00 + t01 * s01 + t02 * s02;
    const float w0y = t00 * s01 + t01 * s11 + t02 * s12;
    const float w0z = t00 * s02 + t01 * s12 + t02 * s22;
    const float w1x = t10 * s00 + t11 * s01 + t12 * s02;
    const float w1y = t10 * s01 + t11 * s11 + t12 * s12;
    const float w1z = t10 * s02 + t11 * s12 + t12 * s22;
    const float a0 = w0x * t00 + w0y * t01 + w0z * t02;
    const float b  = w0x * t10 + w0y * t11 + w0z * t12;
    const float c0 = w1x * t10 + w1y * t11 + w1z * t12;

    const float a = a0 + blur;
    const float c = c0 + blur;
    const float det = a * c - b * b;
    const float det_safe = metal::max(det, 1e-12f);

    float comp = 1.0f;
    if (antialias) {
        comp = metal::sqrt(metal::max(a0 * c0 - b * b, 0.0f) / det_safe);
    }

    // 3-sigma radius from the larger eigenvalue.
    const float mid = 0.5f * (a + c);
    const float lam1 = mid + metal::sqrt(metal::max(mid * mid - det, 0.01f));
    float radius = metal::ceil(3.0f * metal::sqrt(metal::max(lam1, 0.0f)));

    const bool valid = (z > znear) && (det > 0.0f);
    if (!valid) radius = 0.0f;

    // Tile bounding box (inclusive), clipped to the screen.
    const float TILEF = 16.0f;
    int xmin = (int)metal::clamp(metal::floor((u - radius) / TILEF), 0.0f, (float)(tiles_x - 1));
    int xmax = (int)metal::clamp(metal::floor((u + radius) / TILEF), 0.0f, (float)(tiles_x - 1));
    int ymin = (int)metal::clamp(metal::floor((v - radius) / TILEF), 0.0f, (float)(tiles_y - 1));
    int ymax = (int)metal::clamp(metal::floor((v + radius) / TILEF), 0.0f, (float)(tiles_y - 1));

    const bool on_screen = (radius > 0.0f)
        && (u + radius >= 0.0f) && (u - radius < (float)width)
        && (v + radius >= 0.0f) && (v - radius < (float)height);
    const int count = on_screen ? (xmax - xmin + 1) * (ymax - ymin + 1) : 0;

    xy[2 * g + 0] = u;
    xy[2 * g + 1] = v;
    depths[g] = z;
    // Conic + opacity stay fp32: near-camera splats have huge footprints and
    // tiny conic coefficients that underflow half precision. Colors are in
    // [0, ~few] where half error is < 0.1%.
    conic_opac[4 * g + 0] = c / det_safe;
    conic_opac[4 * g + 1] = -b / det_safe;
    conic_opac[4 * g + 2] = a / det_safe;
    conic_opac[4 * g + 3] = opacities[g] * comp;
    rgbh[4 * g + 0] = (half)colors[3 * g + 0];
    rgbh[4 * g + 1] = (half)colors[3 * g + 1];
    rgbh[4 * g + 2] = (half)colors[3 * g + 2];
    rgbh[4 * g + 3] = (half)0.0f;
    bbox[4 * g + 0] = xmin;
    bbox[4 * g + 1] = xmax;
    bbox[4 * g + 2] = ymin;
    bbox[4 * g + 3] = ymax;
    counts[g] = count;
"""

# --------------------------------------------------------------------------
# Kernel 2 — duplicate expansion in depth order into a persistent capacity.
# --------------------------------------------------------------------------
_EXPAND_SRC = """
    const uint i = thread_position_in_grid.x;   // depth rank
    const uint N = (uint)iparams[0];
    if (i >= N) return;

    const int tiles_x = iparams[1];
    const int capacity = iparams[2];

    const int g = order[i];
    const int count = counts[g];
    if (count == 0) return;
    int slot = offsets[i];

    const int xmin = bbox[4 * g + 0];
    const int xmax = bbox[4 * g + 1];
    const int ymin = bbox[4 * g + 2];
    const int ymax = bbox[4 * g + 3];

    for (int ty = ymin; ty <= ymax; ty++) {
        for (int tx = xmin; tx <= xmax; tx++) {
            if (slot >= capacity) return;   // overflow: dropped this frame, grown next
            keys[slot] = (uint)(ty * tiles_x + tx);
            dup_ids[slot] = g;
            slot++;
        }
    }
"""

# --------------------------------------------------------------------------
# Kernel 3 — tile ranges from sorted keys by boundary detection (no atomics).
# --------------------------------------------------------------------------
_RANGES_SRC = """
    const uint i = thread_position_in_grid.x;
    const uint capacity = (uint)iparams[0];
    const uint num_tiles = (uint)iparams[1];
    if (i >= capacity) return;

    const uint k = keys[sort_idx[i]];
    const uint kp = (i == 0) ? 0xFFFFFFFFu : keys[sort_idx[i - 1]];

    if (i == 0 || k != kp) {
        if (k < num_tiles) tile_ranges[2 * k] = (int)i;         // start
        if (i > 0 && kp < num_tiles) tile_ranges[2 * kp + 1] = (int)i;  // end
    }
    if (i == capacity - 1 && k < num_tiles) tile_ranges[2 * k + 1] = (int)(i + 1);
"""

# --------------------------------------------------------------------------
# Kernel 4 — forward-only tile compositing (fp32 conics, fp16 colors).
# --------------------------------------------------------------------------
_RASTER_FAST_SRC = """
    constexpr int TILE = 16;
    constexpr int BLOCK = 256;

    const int width = iparams[0];
    const int height = iparams[1];
    const int tiles_x = iparams[2];
    const float t_min = fparams[0];

    uint2 tile = threadgroup_position_in_grid.xy;
    uint2 lid = thread_position_in_threadgroup.xy;
    uint tidx = thread_index_in_threadgroup;

    const int tile_id = tile.y * tiles_x + tile.x;
    const int px = tile.x * TILE + lid.x;
    const int py = tile.y * TILE + lid.y;
    const bool inside = (px < width) && (py < height);
    const float2 pixf = float2(px + 0.5f, py + 0.5f);

    const int range_start = tile_ranges[2 * tile_id];
    const int range_end = tile_ranges[2 * tile_id + 1];
    const int num = range_end - range_start;
    const int n_batches = (num + BLOCK - 1) / BLOCK;

    threadgroup float2 sm_xy[BLOCK];
    threadgroup float4 sm_co[BLOCK];   // conic.a, conic.b, conic.c, opacity
    threadgroup float3 sm_rgb[BLOCK];

    float T = 1.0f;
    float3 acc = float3(0.0f);
    bool done = !inside;

    for (int b = 0; b < n_batches; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int load = range_start + b * BLOCK + tidx;
        if (load < range_end) {
            const int g = dup_ids[sort_idx[load]];
            // Component loads support both MLX constant and device inputs.
            sm_xy[tidx] = float2(xy[2*g], xy[2*g+1]);
            sm_co[tidx] = float4(conic_opac[4*g], conic_opac[4*g+1],
                                conic_opac[4*g+2], conic_opac[4*g+3]);
            sm_rgb[tidx] = float3(rgbh[4*g], rgbh[4*g+1], rgbh[4*g+2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (done) continue;

        const int batch_size = min(BLOCK, num - b * BLOCK);
        for (int j = 0; j < batch_size; j++) {
            const float2 d = sm_xy[j] - pixf;
            const float4 co = sm_co[j];
            const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y)
                                - co.y * d.x * d.y;
            if (power > 0.0f) continue;
            const float alpha = min(0.99f, co.w * metal::exp(power));
            if (alpha < 1.0f / 255.0f) continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T < t_min) { done = true; T = next_T; break; }
            acc += sm_rgb[j] * (alpha * T);
            T = next_T;
        }
    }

    if (inside) {
        const int pid = py * width + px;
        image[3 * pid + 0] = acc.x + T * background[0];
        image[3 * pid + 1] = acc.y + T * background[1];
        image[3 * pid + 2] = acc.z + T * background[2];
        final_T[pid] = T;
    }
"""


# Explicit CPP compatibility geometry; default source is unchanged.
_CPP_COMPAT_GEOMETRY_SRC = '\n    const uint g = thread_position_in_grid.x;\n    const uint N = (uint)iparams[0];\n    if (g >= N) return;\n\n    const int width = iparams[1];\n    const int height = iparams[2];\n    const int tiles_x = iparams[3];\n    const int tiles_y = iparams[4];\n    const bool antialias = iparams[5] != 0;\n\n    // fparams: R(9) t(3) fx fy cx cy tanfx tanfy blur znear\n    const float fx = fparams[12], fy = fparams[13];\n    const float cx = fparams[14], cy = fparams[15];\n    const float tanfx = fparams[16], tanfy = fparams[17];\n    const float blur = fparams[18], znear = fparams[19];\n\n    const float px = means[3 * g + 0];\n    const float py = means[3 * g + 1];\n    const float pz = means[3 * g + 2];\n\n    // world -> camera\n    const float x = fparams[0] * px + fparams[1] * py + fparams[2] * pz + fparams[9];\n    const float y = fparams[3] * px + fparams[4] * py + fparams[5] * pz + fparams[10];\n    const float z = fparams[6] * px + fparams[7] * py + fparams[8] * pz + fparams[11];\n    const float z_safe = (z > znear ? z : 1.0f);\n    const float inv_z = 1.0f / z_safe;\n\n    const float u = fx * x * inv_z + cx;\n    const float v = fy * y * inv_z + cy;\n\n    // EWA Jacobian with padded-frustum clamping (matches the training path).\n    const float txc = (x * inv_z) * z_safe;\n    const float tyc = (y * inv_z) * z_safe;\n    const float inv_z2 = inv_z * inv_z;\n    const float j00 = fx * inv_z;\n    const float j02 = -fx * txc * inv_z2;\n    const float j11 = fy * inv_z;\n    const float j12 = -fy * tyc * inv_z2;\n\n    // T = J @ R (two rows of a 2x3).\n    const float t00 = j00 * fparams[0] + j02 * fparams[6];\n    const float t01 = j00 * fparams[1] + j02 * fparams[7];\n    const float t02 = j00 * fparams[2] + j02 * fparams[8];\n    const float t10 = j11 * fparams[3] + j12 * fparams[6];\n    const float t11 = j11 * fparams[4] + j12 * fparams[7];\n    const float t12 = j11 * fparams[5] + j12 * fparams[8];\n\n    // cov2d = T @ Sigma @ T^T from the cached upper-triangular 3D covariance.\n    const float s00 = cov3d[6 * g + 0], s01 = cov3d[6 * g + 1], s02 = cov3d[6 * g + 2];\n    const float s11 = cov3d[6 * g + 3], s12 = cov3d[6 * g + 4], s22 = cov3d[6 * g + 5];\n    const float w0x = t00 * s00 + t01 * s01 + t02 * s02;\n    const float w0y = t00 * s01 + t01 * s11 + t02 * s12;\n    const float w0z = t00 * s02 + t01 * s12 + t02 * s22;\n    const float w1x = t10 * s00 + t11 * s01 + t12 * s02;\n    const float w1y = t10 * s01 + t11 * s11 + t12 * s12;\n    const float w1z = t10 * s02 + t11 * s12 + t12 * s22;\n    const float a0 = w0x * t00 + w0y * t01 + w0z * t02;\n    const float b  = w0x * t10 + w0y * t11 + w0z * t12;\n    const float c0 = w1x * t10 + w1y * t11 + w1z * t12;\n\n    const float a = a0 + blur;\n    const float c = c0 + blur;\n    const float det = a * c - b * b;\n    const float det_safe = metal::max(det, 1e-12f);\n\n    float comp = 1.0f;\n    if (antialias) {\n        comp = metal::sqrt(metal::max(a0 * c0 - b * b, 0.0f) / det_safe);\n    }\n\n    // 3-sigma radius from the larger eigenvalue.\n    const float mid = 0.5f * (a + c);\n    const float lam1 = mid + metal::sqrt(metal::max(mid * mid - det, 0.01f));\n    float radius = metal::ceil(3.0f * metal::sqrt(metal::max(lam1, 0.0f)));\n\n    const bool valid = (z > znear) && (det > 0.0f);\n    if (!valid) radius = 0.0f;\n    radius = metal::min(radius, 1000.0f);\n\n    // Tile bounding box (inclusive), clipped to the screen.\n    const float TILEF = 16.0f;\n    int xmin = (int)metal::clamp(metal::floor((u - radius) / TILEF), 0.0f, (float)(tiles_x - 1));\n    int xmax = (int)metal::clamp(metal::floor((u + radius) / TILEF), 0.0f, (float)(tiles_x - 1));\n    int ymin = (int)metal::clamp(metal::floor((v - radius) / TILEF), 0.0f, (float)(tiles_y - 1));\n    int ymax = (int)metal::clamp(metal::floor((v + radius) / TILEF), 0.0f, (float)(tiles_y - 1));\n\n    bool on_screen = radius > 0.0f;\n    // Compatibility bounds may collapse fully offscreen circles onto an edge.\n    // Keep them only if their ellipse can reach a pixel in that rectangle.\n    if (on_screen && !((u + radius >= 0.0f) && (u - radius < width)\n                   && (v + radius >= 0.0f) && (v - radius < height))) {\n        const float qa = c / det_safe, qb = -b / det_safe, qc = a / det_safe;\n        const float lx = float(xmin * 16) + 0.5f - u;\n        const float hx = metal::min(float((xmax + 1) * 16), float(width)) - 0.5f - u;\n        const float ly = float(ymin * 16) + 0.5f - v;\n        const float hy = metal::min(float((ymax + 1) * 16), float(height)) - 0.5f - v;\n        float qmin = INFINITY;\n        for (int k = 0; k < 2; ++k) {\n            float dx = k ? hx : lx;\n            float dy = metal::clamp(-qb * dx / qc, ly, hy);\n            qmin = metal::min(qmin, qa*dx*dx + qc*dy*dy + 2.0f*qb*dx*dy);\n            dy = k ? hy : ly;\n            dx = metal::clamp(-qb * dy / qa, lx, hx);\n            qmin = metal::min(qmin, qa*dx*dx + qc*dy*dy + 2.0f*qb*dx*dy);\n        }\n        const float qlimit = 2.0f * metal::log(metal::max(255.0f * opacities[g], 1.0f));\n        // Small conservative tolerance protects the fp32 boundary decision.\n        if (metal::isfinite(qmin) && qmin > qlimit + 0.01f) on_screen = false;\n    }\n    const int count = on_screen ? (xmax - xmin + 1) * (ymax - ymin + 1) : 0;\n\n    xy[2 * g + 0] = u;\n    xy[2 * g + 1] = v;\n    depths[g] = z;\n    // Conic + opacity stay fp32: near-camera splats have huge footprints and\n    // tiny conic coefficients that underflow half precision. Colors are in\n    // [0, ~few] where half error is < 0.1%.\n    conic_opac[4 * g + 0] = c / det_safe;\n    conic_opac[4 * g + 1] = -b / det_safe;\n    conic_opac[4 * g + 2] = a / det_safe;\n    conic_opac[4 * g + 3] = opacities[g] * comp;\n    rgbh[4 * g + 0] = (half)colors[3 * g + 0];\n    rgbh[4 * g + 1] = (half)colors[3 * g + 1];\n    rgbh[4 * g + 2] = (half)colors[3 * g + 2];\n    rgbh[4 * g + 3] = (half)0.0f;\n    bbox[4 * g + 0] = xmin;\n    bbox[4 * g + 1] = xmax;\n    bbox[4 * g + 2] = ymin;\n    bbox[4 * g + 3] = ymax;\n    counts[g] = count;\n'
_CPP_COMPAT_RASTER_SRC = _RASTER_FAST_SRC.replace(
    'const float alpha = min(0.99f, co.w * metal::exp(power));',
    'const float alpha = min(0.99f, 1.0f - metal::exp(-co.w * metal::exp(power)));',
)
if _CPP_COMPAT_RASTER_SRC == _RASTER_FAST_SRC:
    raise RuntimeError('Expected opacity expression missing from installed Fast raster kernel')

_cpp_compat_geometry_kernel = mx.fast.metal_kernel(
    name='gs_fast_geometry_cpp_compat_opt_in',
    input_names=['means','cov3d','opacities','colors','fparams','iparams'],
    output_names=['xy','depths','conic_opac','rgbh','bbox','counts'],
    source=_CPP_COMPAT_GEOMETRY_SRC,
)
_cpp_compat_raster_kernel = mx.fast.metal_kernel(
    name='gs_fast_raster_cpp_exp_alpha_opt_in',
    input_names=['xy','conic_opac','rgbh','dup_ids','sort_idx','tile_ranges','background','fparams','iparams'],
    output_names=['image','final_T'],source=_CPP_COMPAT_RASTER_SRC,
)

_geometry_kernel = mx.fast.metal_kernel(
    name="gs_fast_geometry",
    input_names=["means", "cov3d", "opacities", "colors", "fparams", "iparams"],
    output_names=["xy", "depths", "conic_opac", "rgbh", "bbox", "counts"],
    source=_GEOMETRY_SRC,
)

_expand_kernel = mx.fast.metal_kernel(
    name="gs_fast_expand",
    input_names=["order", "offsets", "counts", "bbox", "iparams"],
    output_names=["keys", "dup_ids"],
    source=_EXPAND_SRC,
)

_ranges_kernel = mx.fast.metal_kernel(
    name="gs_fast_ranges",
    input_names=["keys", "sort_idx", "iparams"],
    output_names=["tile_ranges"],
    source=_RANGES_SRC,
)

_raster_fast_kernel = mx.fast.metal_kernel(
    name="gs_fast_raster",
    input_names=[
        "xy",
        "conic_opac",
        "rgbh",
        "dup_ids",
        "sort_idx",
        "tile_ranges",
        "background",
        "fparams",
        "iparams",
    ],
    output_names=["image", "final_T"],
    source=_RASTER_FAST_SRC,
)


_ranges_binary_kernel = mx.fast.metal_kernel(
    name="cpp_compat_tile_ranges_binary",
    input_names=["keys", "sort_idx", "iparams"],
    output_names=["tile_ranges"],
    source=r"""
        uint tile = thread_position_in_grid.x;
        int size = iparams[0];
        if (tile >= (uint)iparams[1]) return;
        int lo = 0, hi = size;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (keys[sort_idx[mid]] < tile) lo = mid + 1;
            else hi = mid;
        }
        int start = lo;
        hi = size;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (keys[sort_idx[mid]] <= tile) lo = mid + 1;
            else hi = mid;
        }
        tile_ranges[2 * tile] = start;
        tile_ranges[2 * tile + 1] = lo;
    """,
)


_radix_rank_hist_kernel = mx.fast.metal_kernel(
    name="cpp_compat_radix4_rank_hist",
    input_names=["keys", "params"],
    output_names=["ranks", "hist"],
    source=r"""
        // Apple Silicon: 32 lanes/SIMD group, 8 groups in this 256-thread block.
        uint i = thread_position_in_grid.x;
        uint local = i & 255u;
        uint block = i >> 8;
        uint lane = local & 31u;
        uint subgroup = local >> 5;
        uint n = (uint)params[0];
        uint blocks = (uint)params[1];
        uint digit = i < n ? ((keys[i] >> params[2]) & 15u) : 0u;
        threadgroup uint subgroup_counts[128];
        uint rank = 0;
        for (uint bin = 0; bin < 16; ++bin) {
            uint match = (uint)(i < n && digit == bin);
            uint before = simd_prefix_exclusive_sum(match);
            uint count = simd_sum(match);
            if (lane == 0) subgroup_counts[subgroup * 16 + bin] = count;
            if (match) rank = before;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local < 16) {
            uint total = 0;
            for (uint sg = 0; sg < 8; ++sg) total += subgroup_counts[sg * 16 + local];
            hist[local * blocks + block] = (int)total;
        }
        if (i < n) {
            for (uint sg = 0; sg < subgroup; ++sg) rank += subgroup_counts[sg * 16 + digit];
            ranks[i] = (int)rank;
        }
    """,
)

_radix_scatter_kernel = mx.fast.metal_kernel(
    name="cpp_compat_radix4_scatter",
    input_names=["keys", "indices", "ranks", "prefix", "starts", "params"],
    output_names=["out_keys", "out_indices"],
    source=r"""
        uint i = thread_position_in_grid.x;
        if (i >= (uint)params[0]) return;
        uint block = i >> 8;
        uint blocks = (uint)params[1];
        uint digit = (keys[i] >> params[2]) & 15u;
        uint dst = (uint)(starts[digit] + prefix[digit * blocks + block] + ranks[i]);
        out_keys[dst] = keys[i];
        out_indices[dst] = indices[i];
    """,
)


def _radix_tile_argsort(keys, num_tiles):
    """Stable tile-key argsort for Apple Silicon; real keys <num_tiles or sentinel.

    Per-block SIMD ranks preserve lane order; block-prefix offsets preserve block
    order. Stable low-to-high digit passes therefore retain global depth order
    within each tile. Include a distinct sentinel digit above every real tile.
    """
    n = int(keys.size)
    if n == 0:
        return mx.zeros((0,), dtype=mx.int32)
    blocks = (n + 255) // 256
    indices = mx.arange(n, dtype=mx.int32)
    for shift in range(0, max(1, int(num_tiles).bit_length()), 4):
        params = mx.array([n, blocks, shift], dtype=mx.int32)
        ranks, hist = _radix_rank_hist_kernel(
            inputs=[keys, params],
            output_shapes=[(n,), (16, blocks)],
            output_dtypes=[mx.int32, mx.int32],
            grid=(blocks * 256, 1, 1), threadgroup=(256, 1, 1),
        )
        prefix = mx.cumsum(hist, axis=1) - hist
        totals = mx.sum(hist, axis=1)
        starts = mx.cumsum(totals) - totals
        keys, indices = _radix_scatter_kernel(
            inputs=[keys, indices, ranks, prefix, starts, params],
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.uint32, mx.int32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1),
        )
    return indices


def final_pass_plan(num_tiles):
    """Eligibility of the sentinel-safe last radix pass."""
    shift = ((int(num_tiles).bit_length() - 1) // 4) * 4
    real_categories = ((int(num_tiles) - 1) >> shift) + 1
    return {"shift": shift, "real_categories": real_categories,
            "eligible": real_categories <= 2, "passes": shift // 4 + 1}


_FINAL_RANK = mx.fast.metal_kernel(
    name="cpp_compat_selective_final3_rank_hist",
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
    name="cpp_compat_selective_final3_scatter_indices", input_names=_SCATTER_INPUTS,
    output_names=["out_indices"], source=_SCATTER_COMMON,
)
_FINAL_WITH_KEYS = mx.fast.metal_kernel(
    name="cpp_compat_selective_final3_scatter_keys_indices", input_names=_SCATTER_INPUTS,
    output_names=["out_keys", "out_indices"],
    source=_SCATTER_COMMON + "\n out_keys[dst] = key;\n",
)


def _selective_final3_argsort(keys, num_tiles, *, scatter_group=256):
    """Return exactly the production stable permutation, with a cheaper last pass.

    Earlier stable LSD passes order the lower bits. The last stable partition
    separates high digits 0, 1, and sentinel; it never aliases sentinel with 1.
    Each category preserves preceding lane and block order, including sentinel
    input order. Unsupported tile domains delegate to the unchanged production
    sorter. No data-dependent host readback or sentinel-fraction assumption.
    """
    if scatter_group not in (256, 512):
        raise ValueError("scatter_group must be 256 or 512")
    plan = final_pass_plan(num_tiles)
    if keys.ndim != 1 or keys.dtype != mx.uint32:
        raise ValueError("keys must be a 1D uint32 array")
    n = int(keys.size)
    if n >= 2**31:
        raise OverflowError("signed 32-bit permutation limit exceeded")
    if n == 0 or not plan["eligible"]:
        return _radix_tile_argsort(keys, num_tiles)
    blocks = (n + 255) // 256
    indices = mx.arange(n, dtype=mx.int32)
    for shift in range(0, plan["shift"], 4):
        params = mx.array([n, blocks, shift], dtype=mx.int32)
        ranks, hist = _radix_rank_hist_kernel(
            inputs=[keys, params], output_shapes=[(n,), (16, blocks)],
            output_dtypes=[mx.int32, mx.int32],
            grid=(blocks * 256, 1, 1), threadgroup=(256, 1, 1))
        prefix = mx.cumsum(hist, axis=1) - hist
        totals = mx.sum(hist, axis=1)
        starts = mx.cumsum(totals) - totals
        keys, indices = _radix_scatter_kernel(
            inputs=[keys, indices, ranks, prefix, starts, params],
            output_shapes=[(n,), (n,)], output_dtypes=[mx.uint32, mx.int32],
            grid=(n, 1, 1), threadgroup=(scatter_group, 1, 1))
    params = mx.array([n, blocks, plan["shift"]], dtype=mx.int32)
    ranks, hist = _FINAL_RANK(
        inputs=[keys, params], output_shapes=[(n,), (3, blocks)],
        output_dtypes=[mx.int32, mx.int32],
        grid=(blocks * 256, 1, 1), threadgroup=(256, 1, 1))
    prefix = mx.cumsum(hist, axis=1) - hist
    totals = mx.sum(hist, axis=1)
    starts = mx.cumsum(totals) - totals
    inputs = [keys, indices, ranks, prefix, starts, params]
    return _FINAL_INDICES(
        inputs=inputs, output_shapes=[(n,)], output_dtypes=[mx.int32],
        grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]



def _cov3d_upper(quats: mx.array, scales: mx.array) -> mx.array:
    """Upper-triangular world covariance (N, 6): s00 s01 s02 s11 s12 s22."""
    q = quats / mx.maximum(mx.linalg.norm(quats, axis=-1, keepdims=True), 1e-12)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)
    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]
    m00, m01, m02 = r00 * sx, r01 * sy, r02 * sz
    m10, m11, m12 = r10 * sx, r11 * sy, r12 * sz
    m20, m21, m22 = r20 * sx, r21 * sy, r22 * sz
    s00 = m00 * m00 + m01 * m01 + m02 * m02
    s01 = m00 * m10 + m01 * m11 + m02 * m12
    s02 = m00 * m20 + m01 * m21 + m02 * m22
    s11 = m10 * m10 + m11 * m11 + m12 * m12
    s12 = m10 * m20 + m11 * m21 + m12 * m22
    s22 = m20 * m20 + m21 * m21 + m22 * m22
    return mx.stack([s00, s01, s02, s11, s12, s22], axis=-1)


class FastGaussianRenderer:
    """Forward-only, cache-heavy Gaussian renderer for viewing and playback.

    Precomputes activations, 3D covariances, and SH colors once; per frame it
    uses fused Metal kernels and selectable stable tile sorting. Capacity,
    adaptive and auto sizing avoid mid-frame CPU synchronization after setup.
    Roughly matching the reference renderer at the pixel level (identical
    compositing math; splat colors are quantized to fp16 and early
    termination is slightly more aggressive).

    Args:
        model (GaussianModel): the scene (or anything exposing ``params``,
            ``scales_act``, ``opacities_act``, ``sh``, ``active_sh_degree``).
        sh_degree: cap the evaluated SH degree (``None`` = the model's active
            degree; ``0`` renders view-independent color, fastest).
        t_min: transmittance early-termination threshold. The training kernel
            uses ``1e-4``; the default here (``1/255``) stops as soon as any
            further splat could no longer change an 8-bit pixel.
        color_refresh: fraction of the scene radius the camera must move
            before view-dependent SH colors are re-evaluated (``0`` =
            every frame).
        antialias: Mip-Splatting opacity compensation (as in the training path).
        duplicate_sort_mode: capacity (reference), adaptive (5% reserve with
            safe growth/shrink), auto (adaptive above 1M), or experimental
            active/bucketed host-sized sorting.
        tile_range_mode: scan, binary lower/upper bounds, or auto (binary for
            lists of at least 1M slots).
        tile_sort_impl: mlx stable argsort, radix4, auto, selective_final3, or opt-in selective_final3_scatter512 (radix for lists
            of at least 4M slots; targets Apple Silicon SIMD width 32).

    Not differentiable. For training use ``render_gaussians``.
    """

    def __init__(
        self,
        model=None,
        *,
        means: mx.array | None = None,
        quats: mx.array | None = None,
        scales: mx.array | None = None,
        opacities: mx.array | None = None,
        colors: mx.array | None = None,
        sh: mx.array | None = None,
        sh_degree: int | None = None,
        t_min: float = 1.0 / 255.0,
        color_refresh: float = 0.005,
        antialias: bool = False,
        geometry_policy: str = "default",
        alpha_mode: str = "linear",
        duplicate_sort_mode: str = "capacity",
        tile_range_mode: str = "scan",
        tile_sort_impl: str = "mlx",
    ):
        self.t_min = float(t_min)
        self.color_refresh = float(color_refresh)
        self.antialias = bool(antialias)
        if geometry_policy not in ('default', 'cpp_compat'):
            raise ValueError('geometry_policy must be default or cpp_compat')
        if alpha_mode not in ('linear', 'exponential'):
            raise ValueError('alpha_mode must be linear or exponential')
        if geometry_policy == 'cpp_compat' and alpha_mode != 'exponential':
            raise ValueError('cpp_compat geometry requires alpha_mode=exponential for validated CPP parity')
        self.geometry_policy = geometry_policy
        self.alpha_mode = alpha_mode
        if duplicate_sort_mode not in ("capacity", "active", "bucketed", "adaptive", "auto", "guarded"):
            raise ValueError("duplicate_sort_mode must be capacity, active, bucketed, adaptive, auto or guarded")
        if tile_range_mode not in ("scan", "binary", "auto"):
            raise ValueError("tile_range_mode must be scan, binary or auto")
        self.tile_range_mode = tile_range_mode
        if tile_sort_impl not in ("mlx", "radix4", "auto", "selective_final3", "selective_final3_scatter512"):
            raise ValueError("tile_sort_impl must be mlx, radix4, auto, selective_final3 or selective_final3_scatter512")
        self.tile_sort_impl = tile_sort_impl
        self._frame_needs_mask = True
        self.duplicate_sort_mode = duplicate_sort_mode
        self._frame_duplicate_count = None
        self.last_frame_counts = {}
        self._sh_degree_cap = sh_degree

        self._capacity = 0
        self._cached_colors: mx.array | None = None
        self._cached_color_center: mx.array | None = None

        if model is not None:
            self.update(
                means=model.params["means"],
                quats=model.params["quats"],
                scales=model.scales_act,
                opacities=model.opacities_act,
                sh=model.sh,
                sh_degree=(
                    model.active_sh_degree if sh_degree is None else min(sh_degree, model.sh_degree)
                ),
            )
        else:
            if means is None or opacities is None or (colors is None and sh is None):
                raise ValueError(
                    "Provide either `model` or (means, quats, scales, opacities, colors|sh)."
                )
            self.update(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                sh=sh,
                sh_degree=3 if sh_degree is None else sh_degree,
            )

    def _use_adaptive_capacity(self, needed):
        # Below 1M duplicates keep the original reserve: extra growth and
        # specialized kernels have measurable overhead on tiny scenes.
        return self.duplicate_sort_mode == "adaptive" or (
            self.duplicate_sort_mode == "auto" and needed >= (1 << 20))

    # ------------------------------------------------------------- scene state
    def update(
        self,
        means: mx.array,
        quats: mx.array | None = None,
        scales: mx.array | None = None,
        opacities: mx.array | None = None,
        colors: mx.array | None = None,
        sh: mx.array | None = None,
        sh_degree: int | None = None,
        cov3d: mx.array | None = None,
    ) -> None:
        """Replace scene arrays (e.g. per timestep of a 4D sequence).

        ``scales`` / ``opacities`` are the *activated* values. Pass ``cov3d``
        (N, 6 upper-triangular) directly to skip the quaternion math when a
        sequence precomputes covariances. Caches are rebuilt lazily.
        """
        self.means = means.astype(mx.float32)
        self.n = int(means.shape[0])
        if cov3d is not None:
            self.cov3d = cov3d.astype(mx.float32)
        else:
            if quats is None or scales is None:
                raise ValueError("Provide quats+scales or cov3d.")
            self.cov3d = _cov3d_upper(quats.astype(mx.float32), scales.astype(mx.float32))
        if opacities is not None:
            self.opacities = opacities.astype(mx.float32)
        if (colors is None) == (sh is None) and colors is not None:
            raise ValueError("Provide only one of `colors` or `sh`.")
        if colors is not None:
            self._colors_static = colors.astype(mx.float32)
            self._sh = None
        elif sh is not None:
            self._sh = sh.astype(mx.float32)
            self._colors_static = None
        if sh_degree is not None:
            self.sh_degree = int(sh_degree)

        # A cheap scene radius for the color-refresh heuristic.
        if self.n > 0:
            lo = mx.min(self.means, axis=0)
            hi = mx.max(self.means, axis=0)
            self._scene_radius = float(mx.linalg.norm(hi - lo).item()) * 0.5 + 1e-6
        else:
            self._scene_radius = 1.0

        self._cached_colors = None
        self._cached_color_center = None
        # Evaluate everything the render graph will capture so frames start hot.
        mx.eval(self.means, self.cov3d, self.opacities)

    # ----------------------------------------------------------------- colors
    def _view_colors(self, camera: Camera) -> mx.array:
        if self._colors_static is not None:
            return self._colors_static
        # The cache key lives in plain Python floats: viewers render on HTTP
        # handler threads, and MLX cannot evaluate lazy arrays created on a
        # different thread, so no cross-frame MLX graph may survive here.
        center = camera.camera_center
        center_f = tuple(float(c) for c in center)
        if self._cached_colors is not None and self.color_refresh > 0:
            moved = sum((a - b) ** 2 for a, b in zip(center_f, self._cached_color_center)) ** 0.5
            if moved < self.color_refresh * self._scene_radius:
                return self._cached_colors
        deg = self.sh_degree if self._sh_degree_cap is None else self._sh_degree_cap
        deg = min(deg, self.sh_degree)
        dirs = self.means - center
        dirs = dirs / mx.maximum(mx.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
        colors = mx.maximum(eval_sh(deg, self._sh, dirs), 0.0)
        mx.eval(colors)
        self._cached_colors = colors
        self._cached_color_center = center_f
        return colors

    # ----------------------------------------------------------------- render
    def render(self, camera: Camera, background: mx.array | None = None) -> dict[str, mx.array]:
        """Render one frame. Returns ``{"image", "alpha"}`` (not differentiable)."""
        w, h = camera.width, camera.height
        tiles_x = (w + TILE_SIZE - 1) // TILE_SIZE
        tiles_y = (h + TILE_SIZE - 1) // TILE_SIZE
        num_tiles = tiles_x * tiles_y
        if background is None:
            background = mx.zeros((3,))
        background = background.astype(mx.float32)
        if self.n == 0:
            image = mx.broadcast_to(background[None, None, :], (h, w, 3))
            return {"image": image, "alpha": mx.zeros((h, w))}
        colors = self._view_colors(camera)

        R, t = camera.R, camera.t
        fparams = mx.concatenate(
            [
                R.reshape(-1).astype(mx.float32),
                t.reshape(-1).astype(mx.float32),
                mx.array(
                    [
                        camera.fx,
                        camera.fy,
                        camera.cx,
                        camera.cy,
                        0.5 * w / camera.fx,
                        0.5 * h / camera.fy,
                        0.3,
                        camera.znear,
                    ],
                    dtype=mx.float32,
                ),
            ]
        )
        iparams_geo = mx.array(
            [self.n, w, h, tiles_x, tiles_y, int(self.antialias)], dtype=mx.int32
        )

        geometry_kernel = (_cpp_compat_geometry_kernel if self.geometry_policy == 'cpp_compat'
                           else _geometry_kernel)
        xy, depths, conic_opac, rgbh, bbox, counts = geometry_kernel(
            inputs=[self.means, self.cov3d, self.opacities, colors, fparams, iparams_geo],
            output_shapes=[
                (self.n, 2),
                (self.n,),
                (self.n, 4),
                (self.n, 4),
                (self.n, 4),
                (self.n,),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float16, mx.int32, mx.int32],
            grid=(self.n, 1, 1),
            threadgroup=(min(256, self.n), 1, 1),
        )

        # Global back-to-front order (front-to-back for the compositor):
        # duplicates emitted in this order are depth-sorted within every tile.
        order = mx.argsort(depths).astype(mx.int32)
        counts_sorted = counts[order]
        offsets = (mx.cumsum(counts_sorted) - counts_sorted).astype(mx.int32)
        total = mx.sum(counts)  # stays on GPU; checked only after the frame

        self._frame_duplicate_count = None
        if self.duplicate_sort_mode in ("active", "bucketed"):
            # MLX sort shapes are host-sized. Pay one scalar readback rather
            # than sorting unused capacity; grow before expansion, never drop.
            needed = int(total.item())
            if needed < 0 or needed >= 2**31:
                raise OverflowError("Tile duplicate count exceeds signed 32-bit indexing")
            self._frame_duplicate_count = needed
            if self.duplicate_sort_mode == "bucketed":
                # Bounded tail (<1M entries) stabilizes GPU allocation/sort
                # shapes across nearby cameras without a fixed scene limit.
                quantum = 1 << 20
                self._frame_duplicate_count = ((needed + quantum - 1) // quantum) * quantum if needed else 1
            if self._frame_duplicate_count >= 2**31:
                raise OverflowError("Rounded duplicate buffer exceeds signed 32-bit indexing")
            self._frame_needs_mask = self._frame_duplicate_count != needed or needed == 0
            if self._capacity == 0 or needed > self._capacity:
                self._capacity = max(needed, 1) * 5 // 4 + 1024
        elif self.duplicate_sort_mode == "guarded":
            # One preflight readback per frame, but preserve the sort buffer
            # shape while it is large enough. This prevents wasted full
            # composite/radix passes on overflow without active-sized sorts.
            needed = int(total.item())
            if needed < 0 or needed >= 2**31:
                raise OverflowError("Tile duplicate count exceeds signed 32-bit indexing")
            if self._capacity == 0 or needed > self._capacity:
                proposed = max(needed, 1) * 105 // 100 + 1024
                if proposed >= 2**31:
                    proposed = 2**31 - 1
                self._capacity = proposed
        elif self._capacity == 0:
            # First frame: one-time sync to size the duplicate buffers.
            needed = int(total.item())
            if self._use_adaptive_capacity(needed):
                self._capacity = needed * 105 // 100 + 1024
            else:
                self._capacity = max(needed + 1024, 1) * 5 // 4

        for _attempt in range(2):
            image, final_T = self._composite(
                xy,
                conic_opac,
                rgbh,
                order,
                offsets,
                counts,
                bbox,
                background,
                total,
                w,
                h,
                tiles_x,
                tiles_y,
                num_tiles,
            )
            # The capacity check rides on the frame's own eval: no extra sync.
            mx.eval(image, final_T, total)
            needed = int(total.item())
            self.last_frame_counts = {
                "active": needed,
                "capacity": self._capacity,
                "sorted": max(self._frame_duplicate_count, 1)
                    if self._frame_duplicate_count is not None else self._capacity,
            }
            if needed <= self._capacity:
                # Reclaim stale high-water capacity for the NEXT frame. The
                # current image is already complete; a later camera jump still
                # takes the overflow-and-rerender path, preserving correctness.
                if (self.duplicate_sort_mode in ("adaptive", "auto")
                        and self._capacity >= (1 << 20)
                        and needed * 4 < self._capacity * 3):
                    self._capacity = needed * 105 // 100 + 1024
                break
            # Rare (e.g. zooming in): grow and re-render this frame correctly.
            if self._use_adaptive_capacity(needed):
                self._capacity = needed * 105 // 100 + 1024
            else:
                self._capacity = needed * 5 // 4 + 1024
        return {"image": image, "alpha": 1.0 - final_T}

    def _composite(
        self,
        xy,
        conic_opac,
        rgbh,
        order,
        offsets,
        counts,
        bbox,
        background,
        total,
        w,
        h,
        tiles_x,
        tiles_y,
        num_tiles,
    ):
        active = self._frame_duplicate_count
        capacity = max(active, 1) if active is not None else self._capacity
        iparams_exp = mx.array([self.n, tiles_x, capacity], dtype=mx.int32)
        keys, dup_ids = _expand_kernel(
            inputs=[order, offsets, counts, bbox, iparams_exp],
            output_shapes=[(capacity,), (capacity,)],
            output_dtypes=[mx.uint32, mx.int32],
            grid=(self.n, 1, 1),
            threadgroup=(min(256, self.n), 1, 1),
            init_value=0,
        )
        # Offsets are a perfect prefix sum, so slots [0, total) are exactly the
        # written ones. Mask the tail with the sentinel *on the GPU* — `total`
        # is a 0-d array, so this costs no synchronization.
        if active is None or self._frame_needs_mask:
            positions = mx.arange(capacity, dtype=mx.int32)
            keys = mx.where(positions < total, keys, mx.array(_SENTINEL, dtype=mx.uint32))

        # Stable 32-bit sort: ties keep depth order from the expansion pass.
        # Downstream kernels read through `sort_idx`, so no gathered copies of
        # the sorted arrays are ever materialized.
        use_radix = self.tile_sort_impl in ("radix4", "selective_final3", "selective_final3_scatter512") or (
            self.tile_sort_impl == "auto" and capacity >= (4 << 20))
        use_final3 = (self.tile_sort_impl in ("selective_final3", "selective_final3_scatter512") and
                      capacity >= 8_000_000 and num_tiles >= 8_000 and
                      final_pass_plan(num_tiles)["eligible"])
        sort_idx = (_selective_final3_argsort(
                        keys, num_tiles, scatter_group=(512 if self.tile_sort_impl == "selective_final3_scatter512" else 256))
                    if use_final3
                    else _radix_tile_argsort(keys, num_tiles) if use_radix
                    else mx.argsort(keys).astype(mx.int32))

        iparams_rg = mx.array([capacity, num_tiles], dtype=mx.int32)
        use_binary_ranges = self.tile_range_mode == "binary" or (
            self.tile_range_mode == "auto" and capacity >= (1 << 20))
        range_kernel = _ranges_binary_kernel if use_binary_ranges else _ranges_kernel
        (tile_ranges,) = range_kernel(
            inputs=[keys, sort_idx, iparams_rg],
            output_shapes=[(num_tiles, 2)],
            output_dtypes=[mx.int32],
            grid=(num_tiles if use_binary_ranges else capacity, 1, 1),
            threadgroup=(256, 1, 1),
            init_value=0,  # empty tiles keep start == end
        )

        iparams_r = mx.array([w, h, tiles_x], dtype=mx.int32)
        fparams_r = mx.array([self.t_min], dtype=mx.float32)
        raster_kernel = (_cpp_compat_raster_kernel if self.alpha_mode == 'exponential'
                         else _raster_fast_kernel)
        image, final_T = raster_kernel(
            inputs=[
                xy,
                conic_opac,
                rgbh,
                dup_ids,
                sort_idx,
                tile_ranges,
                background,
                fparams_r,
                iparams_r,
            ],
            output_shapes=[(h, w, 3), (h, w)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(tiles_x * TILE_SIZE, tiles_y * TILE_SIZE, 1),
            threadgroup=(TILE_SIZE, TILE_SIZE, 1),
            init_value=0,
        )
        return image, final_T


def render_gaussians_fast(
    camera: Camera,
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    opacities: mx.array,
    colors: mx.array | None = None,
    sh: mx.array | None = None,
    sh_degree: int = 3,
    background: mx.array | None = None,
    t_min: float = 1.0 / 255.0,
    antialias: bool = False,
) -> dict[str, mx.array]:
    """One-shot fast forward-only render (signature mirrors ``render_gaussians``).

    For loops (viewers, flythroughs, 4D playback) construct a
    :class:`FastGaussianRenderer` once instead — it caches activations,
    covariances, colors, and buffer capacity across frames.

    Not differentiable; ``scales`` / ``opacities`` are activated values.
    """
    renderer = FastGaussianRenderer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        sh=sh,
        sh_degree=sh_degree,
        t_min=t_min,
        antialias=antialias,
    )
    return renderer.render(camera, background=background)
