"""Shared, fail-closed infrastructure for Phases 122–126 qualification."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import zipfile


def cached_wheelhouse(requirements, destination, manifest_path):
    """Recover exact wheels from pip's cache without importing caller packages.

    Validate archive CRCs and every hashed RECORD entry. This establishes local
    cache content integrity, not remote publisher authenticity. The MLX3D wheel
    additionally has the independently pinned whole-wheel and Fast digests.
    """
    import base64,csv,email,io,shutil
    from pip._vendor.packaging.tags import sys_tags,parse_tag
    supported=list(sys_tags())
    wanted={re.sub(r'[-_.]+','-',r.split('==')[0]).lower():r.split('==')[1] for r in requirements}
    found={}
    destination.mkdir()
    for path in (Path.home()/'Library/Caches/pip').rglob('*.body'):
        try:
            if not zipfile.is_zipfile(path): continue
            with zipfile.ZipFile(path) as z:
                meta_name=next((n for n in z.namelist() if n.endswith('.dist-info/METADATA')),None)
                if not meta_name: continue
                meta=email.message_from_bytes(z.read(meta_name))
                name=re.sub(r'[-_.]+','-',meta['Name']).lower()
                if name not in wanted or name in found or meta['Version']!=wanted[name]: continue
                stem=meta_name.rsplit('/',1)[0]
                wheel=email.message_from_bytes(z.read(stem+'/WHEEL'))
                tags=set().union(*(parse_tag(t) for t in wheel.get_all('Tag',[])))
                tag=next((t for t in supported if t in tags),None)
                if tag is None: continue
                if z.testzip() is not None: raise AssertionError('Cached wheel CRC failure')
                for member,hashed,size in csv.reader(io.StringIO(z.read(stem+'/RECORD').decode())):
                    if not hashed: continue
                    alg,expected=hashed.split('=',1)
                    data=z.read(member)
                    actual=base64.urlsafe_b64encode(hashlib.new(alg,data).digest()).decode().rstrip('=')
                    if actual!=expected or (size and len(data)!=int(size)):
                        raise AssertionError('Cached wheel RECORD mismatch: '+name)
                filename=stem.removesuffix('.dist-info')
                if wheel.get('Build'): filename+='-'+wheel['Build']
                target=destination/(filename+'-'+str(tag)+'.whl')
                shutil.copyfile(path,target)
                found[name]={'version':meta['Version'],'filename':target.name,
                             'sha256':digest(target),'crc_and_record':'PASS','source':'local pip download cache'}
                print('VERIFIED_CACHED_WHEEL',name,meta['Version'],found[name]['sha256'],flush=True)
        except (OSError,zipfile.BadZipFile):
            continue
    manifest={'requirements':requirements,'wheels':found,
              'missing':sorted(set(wanted)-set(found)),
              'scope':'cache integrity verified; not a fresh online download'}
    write_report(manifest_path,manifest)
    return found

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FAST = HERE / 'mlx3d/splatting/fast.py'
ADAPTER = ROOT / 'mlx_gs/renderer/mlx3d_cpp_compat.py'
EXPECTED_LOCAL = '56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710'
EXPECTED_INSTALLED = '4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc'
EXPECTED_WHEEL = 'bdfdffe97d2a933afb6bdac2c01ae609c119bf1e22fe129469f4ab03a59439bf'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invariants():
    import ast
    assert digest(FAST) == EXPECTED_LOCAL, 'Validated Fast digest changed'
    native = ast.parse(FAST.read_text())
    adapter = ast.parse(ADAPTER.read_text())
    def default(tree, cls, function, parameter):
        owner = tree if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        f = next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==function)
        mapping = dict(zip([a.arg for a in f.args.args][-len(f.args.defaults):],f.args.defaults))
        mapping.update(zip([a.arg for a in f.args.kwonlyargs],f.args.kw_defaults))
        return ast.literal_eval(mapping[parameter])
    values = {'native_fast':default(native,'FastGaussianRenderer','__init__','tile_sort_impl'),
              'one_shot_adapter':default(adapter,None,'render_cpp_compat','tile_sort_impl'),
              'persistent_adapter':default(adapter,'PersistentCppCompatRenderer','__init__','tile_sort_impl')}
    assert values == {'native_fast':'mlx','one_shot_adapter':'auto','persistent_adapter':'auto'}, values
    return {'fast_sha256':digest(FAST),'adapter_sha256':digest(ADAPTER),'defaults':values}


def clean_env():
    env = os.environ.copy()
    for name in list(env):
        if name in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','CONDA_PREFIX') or name.startswith('SPLAT_APPLE_'):
            env.pop(name,None)
    env['PYTHONNOUSERSITE']='1'
    env['PYTHONDONTWRITEBYTECODE']='1'
    return env


def inventory():
    return dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))


def output_paths(path):
    path = Path(path).resolve()
    if path.exists(): raise FileExistsError(path)
    logs = path.parent / (path.stem + '_artifacts')
    logs.mkdir(exist_ok=False)
    return path,logs


def write_report(path, report):
    with Path(path).open('x') as handle:
        json.dump(report,handle,indent=2)
        handle.write('\n')


def command(args, log=None, *, cwd=ROOT, env=None, timeout=900):
    args = [str(a) for a in args]
    print('+ '+shlex.join(args),flush=True)
    if log is None:
        p=subprocess.run(args,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
        return p.returncode,p.stdout,p.stderr
    with Path(log).open('x') as handle:
        handle.write('COMMAND: '+shlex.join(args)+'\n'); handle.flush()
        try:
            p=subprocess.run(args,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            handle.write('\nTIMEOUT\n')
            return 124


def gpu_gate():
    rc,out,err=command(['ioreg','-r','-c','AGXAccelerator','-d','1'])
    readings=re.findall(r'"Device Utilization %"\s*=\s*(\d+)',out)
    if rc or not readings or max(map(int,readings))>30:
        raise RuntimeError('GPU test blocked: utilization '+str(readings or 'unknown')+'; limit 30%')
    result={'utc':datetime.now(timezone.utc).isoformat(),'utilization_pct':max(map(int,readings))}
    print('GPU_GATE',result,flush=True)
    return result


def metadata():
    return {'python':sys.version,'platform':platform.platform(),
            'utc':datetime.now(timezone.utc).isoformat()}


@contextmanager
def isolated_runtime(logs, *, with_mlx3d=False, native_build=False):
    """Install only into a disposable venv; fixed final Phase121 runtime stack."""
    if platform.system()!='Darwin' or platform.machine()!='arm64' or sys.version_info[:2]!=(3,13):
        raise RuntimeError('Qualified recipe requires macOS arm64 Python 3.13')
    env=clean_env()
    with tempfile.TemporaryDirectory(prefix='splat_qualification_') as temp:
        tmp=Path(temp)
        def step(name,args):
            rc=command(args,logs/(name+'.log'),env=env)
            if rc: raise RuntimeError(f'{name} failed (exit {rc}); inspect {name}.log')
        step('create_venv',[sys.executable,'-m','venv',tmp/'venv'])
        py=tmp/'venv/bin/python'
        packages=['mlx==0.32.4','mlx-metal==0.32.4','numpy==2.4.2']
        if native_build: packages+=['setuptools==80.10.2','nanobind==2.11.0']
        if with_mlx3d: packages+=['pillow==13.0.0','tqdm==4.70.1']
        wheelhouse=tmp/'wheelhouse'
        cached=cached_wheelhouse(packages+(['mlx3d==0.3.0'] if with_mlx3d else []),wheelhouse,logs/'wheel_cache_manifest.json')
        base_names=[re.sub(r'[-_.]+','-',r.split('==')[0]).lower() for r in packages]
        index_args=(['--no-index','--find-links',wheelhouse,'--only-binary=:all:']
                    if all(n in cached for n in base_names) else ['--index-url','https://pypi.org/simple','--retries','1','--timeout','15'])
        step('install_base',[py,'-m','pip','--isolated','install',*index_args,*packages])
        if with_mlx3d:
            wheels=tmp/'wheels'; wheels.mkdir()
            if 'mlx3d' in cached:
                import shutil
                shutil.copyfile(wheelhouse/cached['mlx3d']['filename'],wheels/cached['mlx3d']['filename'])
            else:
                step('download_mlx3d',[py,'-m','pip','--isolated','download','--no-deps','--only-binary=:all:',
                                      '--index-url','https://pypi.org/simple','--retries','1','--timeout','15','--dest',wheels,'mlx3d==0.3.0'])
            files=list(wheels.glob('mlx3d-0.3.0-*.whl'))
            if len(files)!=1 or digest(files[0])!=EXPECTED_WHEEL:
                raise AssertionError('Exact verified mlx3d wheel digest differs')
            with zipfile.ZipFile(files[0]) as z:
                assert hashlib.sha256(z.read('mlx3d/splatting/fast.py')).hexdigest()==EXPECTED_INSTALLED
            step('install_mlx3d',[py,'-m','pip','--isolated','install','--no-deps',files[0]])
        yield py,tmp,env
