#!/usr/bin/env python3
"""Phase112: opt-in isolation and fail-closed source integrity regression.

Checks that original default sorting is 'mlx', explicit legacy modes remain
accepted, new mode requires explicit selection, and the validated source hash
is enforced under a controlled in-process mock. No source files are modified.
The temporary mock is restored and the cache cleared before exiting.
"""
import argparse,hashlib,inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from unittest.mock import patch
import mlx.core as mx
from mlx_gs.renderer import mlx3d_cpp_compat as adapter

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():p.error(f'output exists: {a.out}')
 original=adapter._renderer_class()
 original_params=inspect.signature(original).parameters
 persistent_params=inspect.signature(adapter.PersistentCppCompatRenderer).parameters
 oneshot_params=inspect.signature(adapter.render_cpp_compat).parameters
 expected_defaults={
  'native_fast':original_params['tile_sort_impl'].default,
  'persistent_adapter':persistent_params['tile_sort_impl'].default,
  'one_shot_adapter':oneshot_params['tile_sort_impl'].default}
 if expected_defaults!={'native_fast':'mlx','persistent_adapter':'auto','one_shot_adapter':'auto'}:
  raise AssertionError(('legacy defaults changed',expected_defaults))
 # The Fast renderer requires an initialized scene even for option tests.
 # A single Gaussian keeps this constructor validation inexpensive.
 fixture=dict(
  means=mx.array([[0.,0.,1.]],dtype=mx.float32),
  quats=mx.array([[1.,0.,0.,0.]],dtype=mx.float32),
  scales=mx.array([[0.01,0.01,0.01]],dtype=mx.float32),
  opacities=mx.array([0.5],dtype=mx.float32),
  colors=mx.array([[0.5,0.5,0.5]],dtype=mx.float32),
  sh_degree=0,
 )
 for mode in ('mlx','radix4','auto','selective_final3','selective_final3_scatter512'):
  test=original(**fixture,tile_sort_impl=mode)
  if test.tile_sort_impl!=mode:raise AssertionError(('mode not stored',mode))
 for bad in ('','selective_final3_scatter1024',None):
  try:original(**fixture,tile_sort_impl=bad)
  except ValueError:pass
  else:raise AssertionError(('invalid mode accepted',bad))
 source=ROOT/'Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py'
 verified_hash=hashlib.sha256(source.read_bytes()).hexdigest()
 if verified_hash!=adapter._VALIDATED_SOURCE_SHA:
  raise AssertionError(('source SHA mismatch',verified_hash,adapter._VALIDATED_SOURCE_SHA))
 # Assert guard refuses an altered local source without touching any file.
 # _renderer_class checks the installed baseline before the checked-in source.
 actual_sha=hashlib.sha256
 class FakeDigest:
  def __init__(self,b):
   self.b=b
  def hexdigest(self):
   return '0'*64 if self.b==source.read_bytes() else actual_sha(self.b).hexdigest()
 def fake_sha(data=b''):
  return FakeDigest(data)
 rejected=False
 try:
  adapter._renderer_class.cache_clear()
  with patch.object(adapter.hashlib,'sha256',fake_sha):
   try:adapter._renderer_class()
   except RuntimeError as e:
    if 'hash mismatch' not in str(e):raise
    rejected=True
  if not rejected:raise AssertionError('tampered-source check not rejected')
 finally:
  adapter._renderer_class.cache_clear()
 if adapter._renderer_class() is not original:
  # The module class identity can change when reloaded, but the restored
  # loader must return a valid FastGaussianRenderer with the same API.
  if adapter._renderer_class().__name__!=original.__name__:
   raise AssertionError('source guard recovery did not load renderer')
 report={'phase':112,'status':'PASS','legacy_default_values':expected_defaults,
  'valid_modes':['mlx','radix4','auto','selective_final3','selective_final3_scatter512'],
  'invalid_modes_rejected':True,'source_digest':verified_hash,
  'tampered_source_rejected':rejected,'source_restoration':'PASS',
  'notes':['In-process mock only; repository files not mutated.',
           'No production mode/default changes.']}
 a.out.write_text(json.dumps(report,indent=2)+'\n')
 print('RESULT phase112 PASS',json.dumps(report),flush=True)
if __name__=='__main__':main()
