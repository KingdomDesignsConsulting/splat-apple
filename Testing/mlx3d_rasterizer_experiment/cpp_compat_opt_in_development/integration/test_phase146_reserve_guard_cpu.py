"""CPU-only Phase146 policy/ownership tests: DOES NOT test MLX/Metal renderer."""
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
import unittest

class FakeCombined:
    def __init__(self, gaussians, *, supported_device=False, **kwargs):
        self.supported = supported_device is True
        self._renderer = None
        self.rendered = 0
        self.update(gaussians)

    def update(self, gaussians):
        if not isinstance(gaussians, int) or gaussians <= 0:
            raise ValueError("bad scene")
        reserve = self._renderer._capacity if self._renderer else 0
        self._renderer = SimpleNamespace(n=gaussians, _capacity=reserve)

    def _render_prepared(self, camera, background):
        self.rendered += 1
        if self._renderer._capacity == 0:
            self._renderer._capacity = 12345
        return self._renderer._capacity

sys.modules["phase142_combined"] = SimpleNamespace(CombinedRenderer=FakeCombined)
spec = importlib.util.spec_from_file_location(
    "phase146_test_target", Path(__file__).with_name("phase146_reserve_guard.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eligible = module.eligible_transition_reset
Renderer = module.ExperimentalReserveResetRenderer

class Policy(unittest.TestCase):
    def test_positive(self):
        self.assertTrue(eligible(supported=True, after_update=True, gaussians=1024,
            width=640, height=360, capacity=1<<22))

    def test_negative_boundaries(self):
        basic = dict(supported=True, after_update=True, gaussians=1024,
                     width=640, height=360, capacity=1<<22)
        for key, value in (("supported", False), ("after_update", False),
            ("gaussians", 0), ("gaussians", 4097), ("width", 1920),
            ("height", 1080), ("capacity", (1<<22)-1)):
            trial = dict(basic, **{key:value})
            self.assertFalse(eligible(**trial), trial)

    def test_reset_only_after_replacement(self):
        x = Renderer(700000, supported_device=True)
        x._renderer._capacity = 54_526_961
        cam = SimpleNamespace(width=640, height=360)
        x._render_prepared(cam, None)
        self.assertFalse(x.last_reserve_reset["applied"])
        x.update(1024)
        x._render_prepared(cam, None)
        self.assertTrue(x.last_reserve_reset["applied"])
        self.assertEqual(x.last_reserve_reset["capacity_before"],54_526_961)
        x._renderer._capacity = 54_526_961
        x._render_prepared(cam, None)
        self.assertFalse(x.last_reserve_reset["applied"])

    def test_no_reset_on_large_camera(self):
        x = Renderer(700000, supported_device=True)
        x._renderer._capacity=54_526_961
        x.update(1024)
        x._render_prepared(SimpleNamespace(width=2560,height=1440),None)
        self.assertFalse(x.last_reserve_reset["applied"])

    def test_no_reset_unattested(self):
        x = Renderer(700000)
        x._renderer._capacity=54_526_961
        x.update(1024)
        x._render_prepared(SimpleNamespace(width=640,height=360),None)
        self.assertFalse(x.last_reserve_reset["applied"])

    def test_atomic_rejected_update(self):
        x=Renderer(700000,supported_device=True)
        x._renderer._capacity=54_526_961
        before=x._renderer
        with self.assertRaises(ValueError):
            x.update(0)
        self.assertIs(before,x._renderer)
        self.assertFalse(x._transition_pending)

if __name__ == "__main__":
    unittest.main()
