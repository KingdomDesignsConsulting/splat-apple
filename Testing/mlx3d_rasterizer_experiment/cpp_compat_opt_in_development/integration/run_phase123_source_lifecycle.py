#!/usr/bin/env python3
"""Phase123: CPU-only loader lifecycle on disposable adapter/source copies."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
from unittest.mock import patch,Mock
from qualification_common import ADAPTER,invariants,metadata,write_report

BASELINE='installed baseline fixture\n'
VALID="class FastGaussianRenderer:\n    generation='initial'\n"
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix='phase123_') as temp:
        root=Path(temp)/'repo'
        adapter=root/'mlx_gs/renderer/mlx3d_cpp_compat.py';adapter.parent.mkdir(parents=True)
        adapter.write_bytes(ADAPTER.read_bytes())
        source=root/'Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py'
        source.parent.mkdir(parents=True);source.write_text(VALID)
        installed=Path(temp)/'installed_fast.py';installed.write_text(BASELINE)
        mlx=ModuleType('mlx');mlx.__path__=[]
        core=ModuleType('mlx.core');mlx.core=core
        spec=importlib.util.spec_from_file_location('_phase123_adapter',adapter)
        subject=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'mlx':mlx,'mlx.core':core}):spec.loader.exec_module(subject)
        importer=Mock(return_value=SimpleNamespace(__file__=str(installed)))
        subject.importlib=SimpleNamespace(import_module=importer,util=importlib.util)
        subject._ORIGINAL_INSTALLED_SHA=sha(BASELINE)
        subject._VALIDATED_SOURCE_SHA=sha(VALID)
        subject._MODULE_NAME='_phase123_disposable_fast'
        try:yield subject,source,installed,importer
        finally:
            subject._renderer_class.cache_clear()
            sys.modules.pop(subject._MODULE_NAME,None)


def rejection(subject,error_type):
    try:subject._renderer_class()
    except error_type as exc:
        assert subject._renderer_class.cache_info().currsize==0
        assert subject._MODULE_NAME not in sys.modules,'stale/partial owned module remains after rejected load'
        return {'error_type':type(exc).__name__,'message':str(exc)}
    raise AssertionError('Expected rejection did not occur')


def scenarios():
    def installed_mismatch(s,source,installed,importer):
        installed.write_text('changed')
        return rejection(s,RuntimeError)
    def local_mismatch(s,source,installed,importer):
        source.write_text('changed')
        return rejection(s,RuntimeError)
    def missing_source(s,source,installed,importer):
        source.unlink()
        return rejection(s,FileNotFoundError)
    def execution_failure(s,source,installed,importer):
        text="partial=True\nraise RuntimeError('fixture failure')\n"
        source.write_text(text);s._VALIDATED_SOURCE_SHA=sha(text)
        return rejection(s,RuntimeError)
    def missing_export(s,source,installed,importer):
        text='partial=True\n'
        source.write_text(text);s._VALIDATED_SOURCE_SHA=sha(text)
        return rejection(s,AttributeError)
    def interrupted_execution(s,source,installed,importer):
        text="partial=True\nraise SystemExit('fixture interrupt')\n"
        source.write_text(text);s._VALIDATED_SOURCE_SHA=sha(text)
        return rejection(s,SystemExit)
    def recovery(s,source,installed,importer):
        installed.write_text('changed');rejection(s,RuntimeError)
        installed.write_text(BASELINE)
        result=s._renderer_class()
        assert result.generation=='initial' and s._renderer_class.cache_info().currsize==1
        return {'recovered':True}
    def cache_reuse(s,source,installed,importer):
        first=s._renderer_class()
        for _ in range(5):assert s._renderer_class() is first
        assert importer.call_count==1
        return {'cache_hits':s._renderer_class.cache_info().hits,'class_identity_retained':True}
    def postload_change(s,source,installed,importer):
        first=s._renderer_class()
        source.write_text('changed');installed.write_text('changed')
        assert s._renderer_class() is first and importer.call_count==1
        s._renderer_class.cache_clear()
        rejection(s,RuntimeError)
        installed.write_text(BASELINE);source.write_text(VALID)
        second=s._renderer_class()
        assert second is not first and second.generation=='initial'
        return {'cached_class_ignores_disk_changes_by_contract':True,'explicit_revalidation_rejected':True,'recovery_new_class':True}
    def postload_missing(s,source,installed,importer):
        s._renderer_class();s._renderer_class.cache_clear();source.unlink()
        return rejection(s,FileNotFoundError)
    return [('installed_sha_mismatch',installed_mismatch),('repository_sha_mismatch',local_mismatch),
            ('missing_source',missing_source),('execution_failure_cleanup',execution_failure),
            ('missing_export_cleanup',missing_export),('interrupted_execution_cleanup',interrupted_execution),
            ('initial_rejection_then_recovery',recovery),('successful_cache_reuse',cache_reuse),
            ('postload_changes_revalidate_recover',postload_change),('postload_missing_source_cleanup',postload_missing)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():p.error('output exists')
    before=invariants();rows=[]
    for name,fn in scenarios():
        with fixture() as args:
            try:details=fn(*args);row={'name':name,'status':'PASS','details':details}
            except Exception as exc:row={'name':name,'status':'FAIL','error':{'type':type(exc).__name__,'message':str(exc)}}
        rows.append(row);print(row['status'],name,flush=True)
    assert invariants()==before
    report={'phase':123,'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'environment':metadata(),'checks':rows,'invariants':before,
            'scope':'Exact adapter code copied into temporary repo; fixture installed/local Fast bytes and fixture digests; no GPU execution or site-packages changes.',
            'contract':'Hashes are checked on uncached class load. Cached classes do not revalidate mutable disk contents. Restart/revalidate after changes.'}
    write_report(a.out,report);print('RESULT PHASE123',report['status'],a.out)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
