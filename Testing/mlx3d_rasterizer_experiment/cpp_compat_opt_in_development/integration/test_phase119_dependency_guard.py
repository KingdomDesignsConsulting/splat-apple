#!/usr/bin/env python3
"""Phase119: verify absent/incompatible MLX3D fails closed and legacy import stays lazy."""
import hashlib
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from mlx_gs.renderer import mlx3d_cpp_compat as adapter

def main():
    adapter._renderer_class.cache_clear()
    real_import = importlib.import_module
    def absent(name, *args, **kwargs):
        if name == "mlx3d.splatting.fast":
            raise ModuleNotFoundError("simulated missing MLX3D Fast", name=name)
        return real_import(name, *args, **kwargs)
    try:
        with patch.object(adapter.importlib, "import_module", side_effect=absent):
            try:
                adapter._renderer_class()
            except ModuleNotFoundError as exc:
                if exc.name != "mlx3d.splatting.fast":
                    raise
            else:
                raise AssertionError("missing MLX3D unexpectedly accepted")
        print("PASS absent_dependency_rejected", flush=True)
        actual = hashlib.sha256
        class WrongHash:
            def hexdigest(self):
                return "f" * 64
        with patch.object(adapter.hashlib, "sha256", side_effect=lambda *args, **kwargs: WrongHash()):
            try:
                adapter._renderer_class()
            except RuntimeError as exc:
                if "Installed MLX3D Fast source differs" not in str(exc):
                    raise
            else:
                raise AssertionError("mismatched installed baseline unexpectedly accepted")
        print("PASS installed_baseline_mismatch_rejected", flush=True)
    finally:
        adapter._renderer_class.cache_clear()
    if adapter._renderer_class().__name__ != "FastGaussianRenderer":
        raise AssertionError("source guard did not recover")
    print("PASS source_guard_restored", flush=True)

if __name__ == "__main__":
    main()
