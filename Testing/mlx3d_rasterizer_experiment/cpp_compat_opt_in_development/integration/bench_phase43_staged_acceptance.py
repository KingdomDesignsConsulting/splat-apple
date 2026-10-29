"""Interleave cached baseline/candidate frames to reduce clock and workload timing bias."""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np
from bench_bonsai_trained import load_trained_ply, json_default
from bench_bonsai_alpha_transmittance import DEFAULT_DATA
from mlx_gs.io.colmap import load_colmap_dataset
from mlx3d.cameras import Camera as Camera3D
from mlx3d.splatting import GaussianModel,FastGaussianRenderer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--ply',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--rounds',type=int,default=12)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
(a.output_dir/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
fm=importlib.import_module('mlx3d.splatting.fast')
staged_path=Path('Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/staging/mlx3d/splatting/fast.py').resolve()
if not staged_path.is_file(): raise FileNotFoundError(staged_path)
spec=importlib.util.spec_from_file_location('mlx3d.splatting._fast_phase43_validation',staged_path)
staged=importlib.util.module_from_spec(spec);sys.modules[spec.name]=staged;spec.loader.exec_module(staged)
assert staged._GEOMETRY_SRC==fm._GEOMETRY_SRC
assert staged._RASTER_FAST_SRC==fm._RASTER_FAST_SRC
source=load_trained_ply(a.ply);original=len(source[0]);seed=12345
_,_,cameras,_=load_colmap_dataset(DEFAULT_DATA,'images_4')
centers=np.asarray([-np.asarray(c.W2C)[:3,:3].T@np.asarray(c.W2C)[:3,3] for c in cameras])
centroid=centers.mean(0);norm=1/(np.linalg.norm(centers-centroid,axis=1).mean()+1e-6)
report={'phase':43,'arguments':vars(a),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notes':['Random order within each paired round; 4 warmup renders per path.','Installed default and staged default are tested for bitwise parity.','Performance compares staged baseline geometry with exponential alpha vs staged cpp_compat geometry with exponential alpha at t_min=1/255.','Reported ratios are paired per-round candidate/baseline; raw samples retained.'], 'cases':[]}
specs=[(original,original,0,'trained'),(400000,800000,2,'displaced'),(700000,800000,2,'displaced'),
       (1600000,1600000,2,'displaced'),(400000,800000,0,'normal'),(700000,800000,0,'normal'),
       (1600000,1600000,0,'normal')]
try:
  for n,N,displacement,scene in specs:
    idx=np.random.default_rng(seed).permutation(original)[np.arange(N)%original]
    arrays=[np.ascontiguousarray(z[idx]) for z in source]
    repeat=np.arange(N)>=original
    rng=np.random.default_rng(np.random.SeedSequence([seed,N]))
    dirs=rng.standard_normal((int(repeat.sum()),3)).astype(np.float32)
    if len(dirs):dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)
    delta=np.zeros((N,3),np.float32)
    delta[repeat]=dirs*np.exp(np.mean(np.clip(arrays[1][repeat],-20,20),axis=1))[:,None]
    pos=np.ascontiguousarray(arrays[0][:n]+np.float32(displacement)*delta[:n])
    model=GaussianModel(dict(means=mx.array(pos),scales=mx.array(arrays[1][:n]),quats=mx.array(arrays[2][:n]),opacities=mx.array(arrays[3][:n,0]),sh_dc=mx.array(arrays[4][:n]),sh_rest=mx.zeros((n,0,3))),sh_degree=0)
    model.active_sh_degree=0
    for camera in (72,0):
      cam=cameras[camera];raw=np.asarray(cam.W2C,np.float32);R=raw[:3,:3];T=np.asarray((R@centroid+raw[:3,3])*norm,np.float32)
      c3=Camera3D(R=mx.array(R),t=mx.array(T),fx=float(cam.fx),fy=float(cam.fy),cx=float(cam.cx)+.5,cy=float(cam.cy)+.5,width=int(cam.W),height=int(cam.H))
      # Check full-image default parity for the same scene/camera before performance measurements.
      inst=fm.FastGaussianRenderer(model,sh_degree=0,antialias=False,t_min=1/255)
      staged_default=staged.FastGaussianRenderer(model,sh_degree=0,antialias=False,t_min=1/255)
      inst_out=inst.render(c3);stage_out=staged_default.render(c3)
      mx.eval(inst_out['image'],inst_out['alpha'],stage_out['image'],stage_out['alpha'])
      if not (np.array_equal(np.asarray(inst_out['alpha']),np.asarray(stage_out['alpha'])) and np.array_equal(np.asarray(inst_out['image']),np.asarray(stage_out['image']))):
        raise AssertionError(f'Staged default differs from installed default: {scene}, camera {camera}, n={n}')
      variants=('baseline','candidate')
      renderers={'baseline':staged.FastGaussianRenderer(model,sh_degree=0,antialias=False,t_min=1/255,
                     geometry_policy='default',alpha_mode='exponential'),
                 'candidate':staged.FastGaussianRenderer(model,sh_degree=0,antialias=False,t_min=1/255,
                     geometry_policy='cpp_compat',alpha_mode='exponential')}
      for _ in range(4):
        for name in variants:
          out=renderers[name].render(c3);mx.eval(out['image'],out['alpha'])
      times={k:[] for k in variants};orders=[]
      order_rng=np.random.default_rng(seed+n+camera+int(displacement))
      for _ in range(a.rounds):
        order=list(order_rng.permutation(list(variants)));orders.append(order)
        for name in order:
          t0=time.perf_counter();out=renderers[name].render(c3);mx.eval(out['image'],out['alpha'])
          times[name].append((time.perf_counter()-t0)*1000)
      ratios=np.asarray(times['candidate'])/np.asarray(times['baseline'])
      rec={'scene':scene,'n':n,'population_max':N,'displacement':displacement,'camera':camera,
           'default_bitwise_equal':True,'samples_ms':times,'orders':orders,'median_ms':{k:float(np.median(v)) for k,v in times.items()},
           'paired_ratio_median':float(np.median(ratios)),'paired_ratio_p10':float(np.percentile(ratios,10)),
           'paired_ratio_p90':float(np.percentile(ratios,90))}
      report['cases'].append(rec)
      (a.output_dir/'results.json').write_text(json.dumps(report,indent=2,default=json_default)+'\n')
      print(scene,n,camera,rec['median_ms'],'paired ratio',rec['paired_ratio_median'],flush=True)
finally:
  pass
