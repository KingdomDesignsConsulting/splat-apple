#!/usr/bin/env python3
"""Build a staged, opt-in MLX3D Fast source file, never overwrite site-packages.

Run with --source /path/to/installed/mlx3d/splatting/fast.py --out ./staging/fast.py
The output is a reviewable full source copy. An unchanged default uses precisely
its original Metal geometry and raster kernel. The CPP geometry and exp-alpha
kernels are compiled lazily only when requested.
"""
import argparse, hashlib, pathlib, re, subprocess, tempfile

EXPECTED_SHA = '4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc'
BASE = pathlib.Path(__file__).resolve().parent

def one(source, old, new, label):
    count=source.count(old)
    if count!=1: raise RuntimeError(f'{label}: expected 1 occurrence, found {count}')
    return source.replace(old,new,1)

def build(src, out, enforce_hash=True):
    original=src.read_bytes()
    sha=hashlib.sha256(original).hexdigest()
    if enforce_hash and sha!=EXPECTED_SHA:
        raise RuntimeError(f'Installed source SHA {sha} differs from validated {EXPECTED_SHA}. Refusing unverified patch.')
    s=original.decode('utf-8')
    # Apply exactly the validated pruned experimental geometry changes, not to
    # the installed file but to a private temporary copy.
    with tempfile.TemporaryDirectory() as td:
        target=pathlib.Path(td)/'mlx3d/splatting/fast.py'
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        patch=BASE/'cpp_compat_pruned.patch'
        proc=subprocess.run(['patch','--batch','--forward','-p1','-i',str(patch)],cwd=td,capture_output=True,text=True)
        if proc.returncode: raise RuntimeError('Validated patch failed: '+proc.stdout+' '+proc.stderr)
        patched=target.read_text()
    start='"""\n\n# --------------------------------------------------------------------------\n# Kernel 2'
    # Extract only updated shader source; source boilerplate remains default.
    prefix='_GEOMETRY_SRC = """'
    i=patched.index(prefix)+len(prefix);j=patched.index('"""',i)
    compat_geo=patched[i:j]
    i0=s.index(prefix)+len(prefix);j0=s.index('"""',i0)
    original_geo=s[i0:j0]
    if original_geo==compat_geo: raise RuntimeError('No geometry patch difference found')
    # Insert additional kernel sources before kernel constructor definitions.
    hook='\n_geometry_kernel = mx.fast.metal_kernel('
    definition='''
# Explicit CPP compatibility geometry; default source is unchanged.
_CPP_COMPAT_GEOMETRY_SRC = '''+repr(compat_geo)+'''
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
'''
    s=one(s,hook,'\n'+definition+hook,'kernel definitions')
    # Extend the public renderer API with opt-in modes (default unchanged).
    s=one(s,'        antialias: bool = False,\n    ):',
          '        antialias: bool = False,\n        geometry_policy: str = "default",\n        alpha_mode: str = "linear",\n    ):','constructor signature')
    s=one(s,'        self.antialias = bool(antialias)', '''        self.antialias = bool(antialias)
        if geometry_policy not in ('default', 'cpp_compat'):
            raise ValueError('geometry_policy must be default or cpp_compat')
        if alpha_mode not in ('linear', 'exponential'):
            raise ValueError('alpha_mode must be linear or exponential')
        if geometry_policy == 'cpp_compat' and alpha_mode != 'exponential':
            raise ValueError('cpp_compat geometry requires alpha_mode=exponential for validated CPP parity')
        self.geometry_policy = geometry_policy
        self.alpha_mode = alpha_mode''','mode validation')
    s=one(s,'        xy, depths, conic_opac, rgbh, bbox, counts = _geometry_kernel(',
          '''        geometry_kernel = (_cpp_compat_geometry_kernel if self.geometry_policy == 'cpp_compat'
                           else _geometry_kernel)
        xy, depths, conic_opac, rgbh, bbox, counts = geometry_kernel(''','geometry dispatch')
    s=one(s,'        image, final_T = _raster_fast_kernel(',
          '''        raster_kernel = (_cpp_compat_raster_kernel if self.alpha_mode == 'exponential'
                         else _raster_fast_kernel)
        image, final_T = raster_kernel(''','raster dispatch')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(s)
    compile(s,str(out),'exec')
    print('Staged:',out,'bytes:',out.stat().st_size)
    print('Validated source sha256:',sha)
    print('Default paths retain original Metal sources, kernels, and thresholds.')
    print('CPP compatibility: FastGaussianRenderer(model, geometry_policy="cpp_compat", alpha_mode="exponential", t_min=0.0001)')
    print('Unchanged default: FastGaussianRenderer(model)')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=pathlib.Path)
    ap.add_argument('--out',required=True,type=pathlib.Path)
    ap.add_argument('--allow-source-mismatch',action='store_true',help='Not recommended; bypasses exact-source provenance guard')
    a=ap.parse_args()
    if a.source.resolve()==a.out.resolve(): ap.error('Refusing in-place modification')
    build(a.source,a.out,not a.allow_source_mismatch)
