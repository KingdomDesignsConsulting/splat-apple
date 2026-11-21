"""Regression tests: Phase146 contract worker must not masquerade as a lifecycle guard.

CPU-only; no MLX, Metal, or checkpoint mutation. The full original GPU evidence
is reaggregated independently by reaggregate_phase146_verified.py.
"""
import unittest
from run_phase146_reserve_guard import summarize

TAGS = ["initial","small_first_0","small_warm_0","fullhd_restore_0",
        "highres_return_0","highres_warm_0"]

def lifecycle(block, mode, *, match=True):
    checksums = {tag:tag for tag in TAGS}
    if not match:
        checksums["small_first_0"] = "DIFFERENT"
    frames = [{"tag":tag, "wall_ms":float(i),}
              for i,tag in enumerate(TAGS)]
    return {"result":{"job":{"kind":"lifecycle","block":block,"mode":mode},
            "status":"PASS","frames":frames, "output_checksums":checksums,
            "initialization_ms":{"construct":10.0},
            "after_release":{"peak":1000,"cache":2000}}}

def contract():
    return {"result":{"job":{"kind":"contracts","block":0,"mode":"guard"},
            "status":"PASS","contracts":{"count":65},
            "output_checksums":{"contract":"different-from-lifecycle"}}}

class SummaryRegression(unittest.TestCase):
    def matrix(self, mismatch=False):
        rows = [contract()]
        for block in range(4):
            rows.append(lifecycle(block,"keep"))
            rows.append(lifecycle(block,"guard",match=not (mismatch and block==0)))
        return rows

    def test_contract_guard_cannot_collide_with_block_zero_lifecycle_guard(self):
        result = summarize(self.matrix(), "146")
        self.assertEqual(len(result), 4)
        self.assertEqual([row["block"] for row in result], [0,1,2,3])

    def test_real_lifecycle_frame_mismatch_remains_fatal(self):
        with self.assertRaisesRegex(AssertionError, "Transition bitwise parity failed"):
            summarize(self.matrix(mismatch=True), "146")

    def test_contract_count_fail_closed(self):
        rows = self.matrix()
        rows[0]["result"]["contracts"]["count"] = 64
        with self.assertRaisesRegex(AssertionError, "contracts"):
            summarize(rows, "146")

if __name__=="__main__":
    unittest.main()
