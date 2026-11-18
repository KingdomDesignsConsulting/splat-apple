#!/usr/bin/env python3
"""Phase 37: independent Fast Metal geometry policies and regression metrics.

Uses fixed Phase-25 population RNG, native CPP stage1 tile membership, and an
in-memory monkey patch of MLX3D Fast's geometry output. Production untouched.
Requires benchmark helpers from this Testing directory.
"""
import argparse, copy, importlib, json, sys, gc, time, hashlib, platform
from pathlib import Path
import numpy as np
import mlx.core as mx
from bench_bonsai_trained import load_trained_ply, json_default
from bench_bonsai_alpha_transmittance import C0, DEFAULT_PLY, DEFAULT_DATA
from bench_bonsai_cpp_boundary_consistency import inv_cov
from bench_bonsai_multicamera import metrics, save_png

VARIANTS = ('staged_cpp_compat', 'baseline', 'no_clamp', 'no_clamp_cap', 'no_clamp_edges', 'no_clamp_cap_edges', 'no_clamp_cap_edges_safe_z', 'no_clamp_cap_edges_pruned_safe_z', 'no_clamp_support_safe_z')
GEOMETRY_SOURCES = {}


def geometry_kernels(mx, fm):
    src = fm._GEOMETRY_SRC
    GEOMETRY_SOURCES['baseline'] = src
    patx = 'metal::clamp(x * inv_z, -1.3f * tanfx, 1.3f * tanfx)'
    paty = 'metal::clamp(y * inv_z, -1.3f * tanfy, 1.3f * tanfy)'
    if src.count(patx) != 1 or src.count(paty) != 1:
        raise RuntimeError('Unexpected installed MLX3D Jacobian source; aborting rather than patching a wrong expression')
    kernels = {}
    for name, mul in [(name, None) for name in VARIANTS if name not in ('baseline', 'staged_cpp_compat')]:
        if mul is None:
            new = src.replace(patx, '(x * inv_z)').replace(paty, '(y * inv_z)')
        else:
            new = src.replace(patx, f'metal::clamp(x * inv_z, -{mul} * tanfx, {mul} * tanfx)').replace(paty, f'metal::clamp(y * inv_z, -{mul} * tanfy, {mul} * tanfy)')
        if '_cap' in name:
            new = new.replace('if (!valid) radius = 0.0f;', 'if (!valid) radius = 0.0f;\n    radius = metal::min(radius, 1000.0f);')
        if '_edges' in name:
            start = new.index('    const bool on_screen =')
            stop = new.index('    const int count =', start)
            new = new[:start] + '    const bool on_screen = radius > 0.0f;\n' + new[stop:]
        if '_safe_z' in name:
            new = new.replace('metal::max(z, 1e-6f)', '(z > znear ? z : 1.0f)')
        if '_support' in name:
            new = new.replace('float radius = metal::ceil(3.0f * metal::sqrt(metal::max(lam1, 0.0f)));', '''const float profile_cut = -metal::log(1.0f - 1.0f / 255.0f);
    const float support2 = metal::max(0.0f, -2.0f * metal::log(profile_cut / metal::max(opacities[g], profile_cut)));
    float radius = metal::ceil(metal::sqrt(metal::max(lam1 * support2, 0.0f)));''')
        if '_pruned' in name:
            new = new.replace('const bool on_screen = radius > 0.0f;', 'bool on_screen = radius > 0.0f;')
            marker='    const int count = on_screen ?'
            pruning='''    // Compatibility bounds may collapse fully offscreen circles onto an edge.
    // Keep them only if their ellipse can reach a pixel in that rectangle.
    if (on_screen && !((u + radius >= 0.0f) && (u - radius < width)
                   && (v + radius >= 0.0f) && (v - radius < height))) {
        const float qa = c / det_safe, qb = -b / det_safe, qc = a / det_safe;
        const float lx = float(xmin * 16) + 0.5f - u;
        const float hx = metal::min(float((xmax + 1) * 16), float(width)) - 0.5f - u;
        const float ly = float(ymin * 16) + 0.5f - v;
        const float hy = metal::min(float((ymax + 1) * 16), float(height)) - 0.5f - v;
        float qmin = INFINITY;
        for (int k = 0; k < 2; ++k) {
            float dx = k ? hx : lx;
            float dy = metal::clamp(-qb * dx / qc, ly, hy);
            qmin = metal::min(qmin, qa*dx*dx + qc*dy*dy + 2.0f*qb*dx*dy);
            dy = k ? hy : ly;
            dx = metal::clamp(-qb * dy / qa, lx, hx);
            qmin = metal::min(qmin, qa*dx*dx + qc*dy*dy + 2.0f*qb*dx*dy);
        }
        const float qlimit = 2.0f * metal::log(metal::max(255.0f * opacities[g], 1.0f));
        // Small conservative tolerance protects the fp32 boundary decision.
        if (metal::isfinite(qmin) && qmin > qlimit + 0.01f) on_screen = false;
    }
'''
            assert new.count(marker)==1
            new=new.replace(marker,pruning+marker)
        GEOMETRY_SOURCES[name] = new
        kernels[name] = mx.fast.metal_kernel(name='gs_fast_geom_phase37_'+name,
            input_names=['means','cov3d','opacities','colors','fparams','iparams'],
            output_names=['xy','depths','conic_opac','rgbh','bbox','counts'],source=new)
    import importlib.util
    staged_path = Path('Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/staging/mlx3d/splatting/fast.py').resolve()
    if not staged_path.is_file():
        raise FileNotFoundError(f'Staged renderer not built: {staged_path}')
    spec = importlib.util.spec_from_file_location('mlx3d.splatting._fast_phase41_regression', staged_path)
    staged = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = staged
    spec.loader.exec_module(staged)
    kernels['staged_cpp_compat'] = staged._cpp_compat_geometry_kernel
    GEOMETRY_SOURCES['staged_cpp_compat'] = staged._CPP_COMPAT_GEOMETRY_SRC
    # The existing phase37 harness uses an independently compiled exp-alpha shader
    # for every variant. The staged shader is byte-equivalent by construction.
    assert staged._CPP_COMPAT_RASTER_SRC == fm._RASTER_FAST_SRC.replace(
        'const float alpha = min(0.99f, co.w * metal::exp(power));',
        'const float alpha = min(0.99f, 1.0f - metal::exp(-co.w * metal::exp(power)));')
    return kernels


def exact_bounds(tiles, ids, n, tx, ty):
    """Reconstruct exact per-Gaussian rectangles from native CPP interaction pairs."""
    tiles=np.asarray(tiles).reshape(-1).astype(np.int64)
    ids=np.asarray(ids).reshape(-1).astype(np.int64)
    good=(ids>=0)&(ids<n)&(tiles>=0)&(tiles<tx*ty)
    tiles=tiles[good];ids=ids[good]
    xs=tiles%tx;ys=tiles//tx
    lo_x=np.full(n,tx,np.int32);hi_x=np.full(n,-1,np.int32)
    lo_y=np.full(n,ty,np.int32);hi_y=np.full(n,-1,np.int32)
    np.minimum.at(lo_x,ids,xs.astype(np.int32))
    np.maximum.at(hi_x,ids,xs.astype(np.int32))
    np.minimum.at(lo_y,ids,ys.astype(np.int32))
    np.maximum.at(hi_y,ids,ys.astype(np.int32))
    present=hi_x>=0
    boxes=np.stack((lo_x,hi_x,lo_y,hi_y),axis=-1)
    boxes[~present]=0
    return np.ascontiguousarray(boxes), present, int(len(ids))

def bounds_from_radius(centers, radius, width, height, mode):
    """Independent tile rectangle reconstruction. No CPP interaction list read here."""
    tx=(width+15)//16;ty=(height+15)//16
    centers=np.asarray(centers, dtype=np.float64)
    radius=np.asarray(radius,dtype=np.float64).reshape(-1)
    finite=np.isfinite(centers).all(axis=1)&np.isfinite(radius)&(radius>0)
    x=centers[:,0]; y=centers[:,1]
    if mode=='floor':
        ix0=np.floor((x-radius)/16);ix1=np.floor((x+radius)/16)
        iy0=np.floor((y-radius)/16);iy1=np.floor((y+radius)/16)
    elif mode=='ceil':
        ix0=np.floor((x-radius)/16);ix1=np.ceil((x+radius)/16)-1
        iy0=np.floor((y-radius)/16);iy1=np.ceil((y+radius)/16)-1
    else:raise ValueError(mode)
    raw=np.stack((ix0,ix1,iy0,iy1),axis=1)
    raw=np.nan_to_num(raw,nan=-1e7,posinf=1e7,neginf=-1e7)
    box=np.stack((np.clip(raw[:,0],0,tx-1),np.clip(raw[:,1],0,tx-1),np.clip(raw[:,2],0,ty-1),np.clip(raw[:,3],0,ty-1)),axis=1).astype(np.int32)
    visible=finite&(x+radius>=0)&(x-radius<width)&(y+radius>=0)&(y-radius<height)&(box[:,1]>=box[:,0])&(box[:,3]>=box[:,2])
    count=np.where(visible,(box[:,1]-box[:,0]+1)*(box[:,3]-box[:,2]+1),0).astype(np.int32)
    box[~visible]=0
    return np.ascontiguousarray(box),np.ascontiguousarray(count)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply',type=Path,default=Path(DEFAULT_PLY))
    p.add_argument('--data-dir',default=DEFAULT_DATA)
    p.add_argument('--img-folder',default='images_4')
    p.add_argument('--camera',type=int,default=72)
    p.add_argument('--counts',default='400000,700000')
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--trained',action='store_true')
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--color',action='store_true')
    p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--population-max',type=int,default=800000)
    p.add_argument('--normalized-scene',action='store_true')
    p.add_argument('--displacement',type=float,default=2.)
    p.add_argument('--target-tile-x',type=int,default=0)
    p.add_argument('--target-tile-y',type=int,default=32)
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--offset',type=float,default=.5)
    p.add_argument('--t-min',type=float,default=1/255)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    variants=a.variants.split(',')
    if any(v not in VARIANTS for v in variants):p.error('Unknown variant')
    if a.runs<1:p.error('--runs must be positive')
    if a.output_dir.exists():p.error('Use a new output directory to preserve previous results')
    counts=[int(v.strip()) for v in a.counts.split(',') if v.strip()]
    if not counts or min(counts)<4096 or max(counts)>a.population_max:p.error('counts outside [4096,population-max]')
    if not a.ply.is_file():p.error('PLY does not exist')
    sys.path.insert(0,str(Path.cwd()))
    from mlx_gs.io.colmap import load_colmap_dataset
    from mlx_gs.core.gaussians import Gaussians
    from mlx_gs.renderer.renderer import render,_render_stage1
    from mlx_gs.renderer import rasterizer_metal
    from mlx3d.cameras import Camera as Camera3D
    from mlx3d.splatting import GaussianModel,FastGaussianRenderer
    fm=importlib.import_module('mlx3d.splatting.fast')
    old_expr='const float alpha = min(0.99f, co.w * metal::exp(power));'
    new_expr='const float alpha = min(0.99f, 1.0f - metal::exp(-co.w * metal::exp(power)));'
    if fm._RASTER_FAST_SRC.count(old_expr)!=1:raise RuntimeError('Fast raster source changed')
    exp_kernel=mx.fast.metal_kernel(name='gs_fast_expalpha_clamp_phase34',input_names=['xy','conic_opac','rgbh','dup_ids','sort_idx','tile_ranges','background','fparams','iparams'],output_names=['image','final_T'],source=fm._RASTER_FAST_SRC.replace(old_expr,new_expr))
    kernels=geometry_kernels(mx,fm)
    orig_g=fm._geometry_kernel;orig_r=fm._raster_fast_kernel
    old_capacity=rasterizer_metal._INTERACTION_CAPACITY_ENABLED
    rasterizer_metal.set_tile_expansion_impl('mlx_metal_fill')
    rasterizer_metal.set_sort_key_impl('u32');rasterizer_metal.set_sort_impl('u32_radix')
    rasterizer_metal._INTERACTION_CAPACITY_ENABLED=False
    source=load_trained_ply(a.ply);n_orig=len(source[0]);N=a.population_max
    if a.trained:counts=[n_orig];N=n_orig
    perm=np.random.default_rng(a.seed).permutation(n_orig)
    index=perm[np.arange(N,dtype=np.int64)%n_orig]
    arrays=[np.ascontiguousarray(v[index]) for v in source]
    repeated=np.arange(N)>=n_orig
    rng=np.random.default_rng(np.random.SeedSequence([a.seed,N]))
    dirs=rng.standard_normal((int(repeated.sum()),3)).astype(np.float32)
    if len(dirs):dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)
    radii=np.exp(np.mean(np.clip(arrays[1][repeated],-20,20),axis=1))
    delta=np.zeros((N,3),np.float32);delta[repeated]=dirs*radii[:,None]
    _,_,cameras,_=load_colmap_dataset(a.data_dir,a.img_folder)
    cam=cameras[a.camera]
    centers=np.asarray([-np.asarray(c.W2C)[:3,:3].T@np.asarray(c.W2C)[:3,3] for c in cameras])
    centroid=centers.mean(0) if a.normalized_scene else np.zeros(3)
    norm=float(1/(np.linalg.norm(centers-centroid,axis=1).mean()+1e-6)) if a.normalized_scene else 1.
    raw=np.asarray(cam.W2C,np.float32);R=raw[:3,:3];T=np.asarray((R@centroid+raw[:3,3])*norm,np.float32)
    cp=copy.copy(cam);w2c=np.eye(4,dtype=np.float32);w2c[:3,:3]=R;w2c[:3,3]=T;cp.W2C=mx.array(w2c)
    c3=Camera3D(R=mx.array(R),t=mx.array(T),fx=float(cam.fx),fy=float(cam.fy),cx=float(cam.cx)+a.offset,cy=float(cam.cy)+a.offset,width=int(cam.W),height=int(cam.H))
    W,H=int(cam.W),int(cam.H);tx=(W+15)//16;ty=(H+15)//16
    if not (0<=a.target_tile_x<tx and 0<=a.target_tile_y<ty):p.error('tile outside grid')
    ys=slice(a.target_tile_y*16,min(H,(a.target_tile_y+1)*16))
    xs=slice(a.target_tile_x*16,min(W,(a.target_tile_x+1)*16))
    tile_id=a.target_tile_y*tx+a.target_tile_x
    report={'phase':37,'experiment':'independent_metal_geometry_policy','population_max':N,'displacement':a.displacement,'camera':a.camera,'tile':(a.target_tile_x,a.target_tile_y),'variants':variants,'arguments':vars(a),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fast_source_sha256':hashlib.sha256(fm._GEOMETRY_SRC.encode()).hexdigest(),'python':platform.python_version(),'device':str(mx.device_info()),'notes':['Independent Metal kernel variants; no CPP geometry injected.','Stage1 named ordering corrected: centers,covariance,radii,valid_mask,depths,colors.','Exp-alpha compositor prerequisite is shared by every variant, including baseline.','Timed cached render calls have no diagnostic hooks; first-frame capture time is recorded separately.','All kernel replacements are runtime-only and original functions restored.'],'cases':[]}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    for name in variants:(a.output_dir/f'geometry_{name}.metal').write_text(GEOMETRY_SOURCES[name])
    try:
      for n in counts:
        pos=np.ascontiguousarray(arrays[0][:n]+np.float32(a.displacement)*delta[:n])
        sh=np.full((n,1,3),(.999999-.5)/C0,np.float32)
        v=[mx.array(z) for z in (pos,arrays[1][:n],arrays[2][:n],arrays[3][:n],sh)]
        gs=Gaussians(means=v[0],scales=v[1],quaternions=v[2],opacities=v[3],sh_coeffs=v[4]);mx.eval(*v)
        cpp_start=time.perf_counter()
        cpp_img=render(gs,cp,rasterizer_type='cpp');mx.eval(cpp_img)
        cpp_cold_ms=(time.perf_counter()-cpp_start)*1000
        cpp_alpha=np.asarray(cpp_img,np.float32).mean(-1)
        params={'means':v[0],'scales':v[1],'quaternions':v[2],'opacities':v[3],'sh_coeffs':v[4]}
        cd={'H':cp.H,'W':cp.W,'fx':cp.fx,'fy':cp.fy,'cx':cp.cx,'cy':cp.cy,'W2C':cp.W2C}
        stage=_render_stage1(params,cd);mx.eval(*stage)
        cpp_centers,cpp_covariance,cpp_radii,cpp_valid,cpp_depths,cpp_colors=stage
        assert cpp_radii.dtype==mx.float32 and cpp_valid.dtype==mx.bool_
        ct,ci=rasterizer_metal.get_tile_interactions(stage[0],stage[2],stage[3],stage[4],H,W,16);mx.eval(ct,ci)
        cpp_boxes,cpp_present,cpp_interactions=exact_bounds(np.asarray(ct),np.asarray(ci),n,tx,ty)
        cpp_xy=np.asarray(stage[0]).reshape(n,2).astype(np.float32)
        cpp_radius=np.asarray(stage[2]).reshape(n).astype(np.float64)
        cpp_depth=np.asarray(stage[4]).reshape(n).astype(np.float32)
        cpp_inv,valid_cov=inv_cov(stage[1]);cpp_inv=cpp_inv.astype(np.float32)
        cpp_opacity=(1/(1+np.exp(-np.clip(np.asarray(v[3]).reshape(-1).astype(np.float64),-60,60)))).astype(np.float32)
        model=GaussianModel({'means':v[0],'scales':v[1],'quats':v[2],'opacities':mx.reshape(v[3],(n,)),'sh_dc':v[4],'sh_rest':mx.zeros((n,0,3),dtype=mx.float32)},sh_degree=0)
        model.active_sh_degree=0
        cpp_color=None
        if a.color:
            gs.sh_coeffs=mx.array(arrays[4][:n])
            cpp_color=render(gs,cp,rasterizer_type='cpp');mx.eval(cpp_color)
            cpp_color=np.asarray(cpp_color,np.float32)
            model.params['sh_dc']=mx.array(arrays[4][:n])
        save_png(a.output_dir/f'n{n}_cpp_alpha.png',cpp_alpha)
        np.savez_compressed(a.output_dir/f'n{n}_cpp_arrays.npz',alpha=cpp_alpha,rgb=cpp_color if cpp_color is not None else cpp_alpha[:,:,None])
        if cpp_color is not None:save_png(a.output_dir/f'n{n}_cpp_rgb.png',cpp_color)
        cpp_tiles=np.asarray(ct).ravel();cpp_ids=np.asarray(ci).ravel().astype(np.int64)
        cpp_target_ids=np.unique(cpp_ids[cpp_tiles==tile_id])
        rng_sample=np.random.default_rng(a.seed+n)
        sample_ids=np.sort(rng_sample.choice(cpp_target_ids,min(a.samples,len(cpp_target_ids)),replace=False))
        cpp_conic_sample=cpp_inv[sample_ids]
        cases=[]
        for name in variants:
          cap={}
          def geo_spy(*args,**kw):
            geometry_kernel = kernels[name] if name in kernels else orig_g
            out=list(geometry_kernel(*args,**kw));mx.eval(*out)
            xy,depth,co,rgb,bbox,count=out
            baseline_box=np.asarray(bbox).reshape(n,4).astype(np.int32)
            baseline_count=np.asarray(count).reshape(n).astype(np.int32)
            cap['conic_values']=np.asarray(co).reshape(n,4)[:,:3].copy()
            cap['geometry_counts']=np.asarray(count).reshape(n).astype(np.int64)
            cap['geometry_bbox']=np.asarray(bbox).reshape(n,4).astype(np.int32)
            cap['baseline_count']=baseline_count
            cap['baseline_bbox']=baseline_box
            return (xy,depth,co,rgb,bbox,count)
          def raster_spy(*args,**kw):
            ins=kw.get('inputs',args[0] if args else None)
            if ins is None or len(ins)!=9:raise RuntimeError('Fast raster interface changed')
            dup=np.asarray(ins[3]);order=np.asarray(ins[4]);ranges=np.asarray(ins[5]).reshape(tx*ty,2)
            beg,end=map(int,ranges[tile_id]);cap['actual_tile_ids']=dup[order[beg:end].astype(np.int64)].astype(np.int64) if end>beg else np.empty(0,np.int64)
            cap['raster_xy']=ins[0];cap['raster_conic']=ins[1]
            cap['actual_tile_n']=max(0,end-beg)
            return exp_kernel(*args,**kw)
          fm._geometry_kernel=geo_spy;fm._raster_fast_kernel=raster_spy
          try:
            first_start=time.perf_counter()
            renderer=FastGaussianRenderer(model,sh_degree=0,antialias=False,t_min=a.t_min)
            out=renderer.render(c3);mx.eval(out['alpha'])
            capture_ms=(time.perf_counter()-first_start)*1000
          finally:
            fm._geometry_kernel=orig_g;fm._raster_fast_kernel=orig_r
          alpha=np.asarray(out['alpha'],np.float32)
          rgb=np.asarray(out['image'],np.float32)
          np.savez_compressed(a.output_dir/f'n{n}_{name}_arrays.npz',alpha=alpha,rgb=rgb)
          save_png(a.output_dir/f'n{n}_{name}_alpha.png',alpha)
          save_png(a.output_dir/f'n{n}_{name}_error_x4.png',np.minimum(1,4*np.abs(alpha-cpp_alpha)))
          if a.color:save_png(a.output_dir/f'n{n}_{name}_rgb.png',rgb)
          fm._geometry_kernel=kernels.get(name,orig_g);fm._raster_fast_kernel=exp_kernel
          samples=[]
          try:
            renderer.render(c3)
            for _ in range(a.runs):
                t0=time.perf_counter();timed=renderer.render(c3);mx.eval(timed['alpha'],timed['image'])
                samples.append((time.perf_counter()-t0)*1000)
          finally:fm._geometry_kernel=orig_g;fm._raster_fast_kernel=orig_r
          ids=cap['actual_tile_ids']
          base_n=cap['baseline_count']
          actual_n=cap['geometry_counts']
          tile_ids=np.flatnonzero((cap['geometry_bbox'][:,0]<=a.target_tile_x)&(cap['geometry_bbox'][:,1]>=a.target_tile_x)&(cap['geometry_bbox'][:,2]<=a.target_tile_y)&(cap['geometry_bbox'][:,3]>=a.target_tile_y)&(actual_n>0))
          # Report the diagnostic that the console line expects. The same CPP
          # target-tile sample is used for every clamp policy at this density.
          measured_conics = cap['conic_values'][sample_ids].astype(np.float64)
          reference_conics = cpp_conic_sample.astype(np.float64)
          relative_errors = np.linalg.norm(measured_conics-reference_conics,axis=1) / np.maximum(np.linalg.norm(reference_conics,axis=1),1e-12)
          relative_errors = relative_errors[np.isfinite(relative_errors)]
          conic_rel_p90 = float(np.percentile(relative_errors,90)) if relative_errors.size else float('nan')
          rec={'name':name,'cpp_tile_sample_conic_rel_p90':conic_rel_p90,
               'alpha_metrics':metrics(cpp_alpha[:,:,None],alpha[:,:,None]),
               'rgb_metrics':metrics(cpp_color,rgb) if cpp_color is not None else None,
               'cached_render_ms':samples,'cached_render_median_ms':float(np.median(samples)),
               'capture_first_frame_ms':capture_ms,
               'nonfinite_conic_gaussians':int(np.count_nonzero(~np.isfinite(cap['conic_values']).all(axis=1))),
               'cpp_tile_sample_conic_n':int(relative_errors.size),'global_rmse':float(np.sqrt(np.mean((alpha-cpp_alpha)**2))),
               'tile_rmse':float(np.sqrt(np.mean((alpha[ys,xs]-cpp_alpha[ys,xs])**2))),
               'tile_interactions':int(cap['actual_tile_n']),'tile_ids_unique':int(np.unique(ids).size),
               'geometry_total_interactions':int(actual_n.sum()),'geometry_tile_expected_n':int(len(tile_ids)),
               'geometry_zero_gaussians':int(np.count_nonzero(actual_n==0)),
               'input_counts_changed':int(np.count_nonzero(actual_n!=base_n)),
               'input_bounds_changed':int(np.count_nonzero(np.any(cap['geometry_bbox']!=cap['baseline_bbox'],axis=1))),
               'raster_center_shape':list(cap['raster_xy'].shape),'raster_conic_shape':list(cap['raster_conic'].shape),
               'target_tile_cpp_overlap':int(np.intersect1d(ids,cpp_target_ids).size),
               'target_tile_cpp_missing':int(np.setdiff1d(cpp_target_ids,ids).size), 'target_tile_fast_extra':int(np.setdiff1d(ids,cpp_target_ids).size)}
          if int(cap['actual_tile_n'])!=len(tile_ids):
            raise RuntimeError(f'{name}: expansion tile count {cap["actual_tile_n"]} != bbox-derived {len(tile_ids)}')
          cases.append(rec)
          print(f'n={n} {name:22s} tile={rec["tile_rmse"]:.6f} global={rec["global_rmse"]:.6f} '
                f'tile_ids={rec["tile_interactions"]} total={rec["geometry_total_interactions"]} '
                f'changed_count={rec["input_counts_changed"]} conic_rel_p90={rec["cpp_tile_sample_conic_rel_p90"]:.5g}',flush=True)
          del renderer,out,alpha,cap
          gc.collect()
        report['cases'].append({'n':n,'cpp_cold_ms':cpp_cold_ms,'cpp_interactions':cpp_interactions,'cpp_present_gaussians':int(cpp_present.sum()),'valid_cpp_cov':int(valid_cov.sum()),'results':cases})
        (a.output_dir/'results.json').write_text(json.dumps(report,indent=2,default=json_default)+'\n')
        del model,gs,cpp_img,stage,ct,ci
        gc.collect()
    finally:
      fm._geometry_kernel=orig_g;fm._raster_fast_kernel=orig_r
      rasterizer_metal._INTERACTION_CAPACITY_ENABLED=old_capacity
    print('Saved:',a.output_dir/'results.json')

if __name__=='__main__':main()
