"""CPU tests for shared leases, gate acceptance, matrix and evidence."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from phase138_checkpoint import stable_gpu_gate
from phase140_gpu_lease import (AdmissionBusy, atomic_json, checkpoint, lease,
                                legacy_pids, require_inherited_lease, verify)
from run_phase140_143 import jobs_for

class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "machine.lock"

    def probe(self):
        return lease([], self.path, check=lambda: None)

    def test_competing_runners_and_normal_release(self):
        with self.probe():
            with self.assertRaisesRegex(AdmissionBusy, "GPU_LOCK_BUSY"):
                with self.probe():
                    self.fail("lease overlap")
        with self.probe():
            self.assertTrue(self.path.exists())

    def test_unlocked_existing_file(self):
        self.path.write_text("old process details")
        with self.probe():
            self.assertEqual(self.path.read_text(), "old process details")

    def test_termination_releases_lease(self):
        program = ("import fcntl,sys,time; f=open(sys.argv[1],'a+b');"
                   "fcntl.flock(f,fcntl.LOCK_EX);print('ready',flush=True);time.sleep(90)")
        child = subprocess.Popen([sys.executable, "-c", program, str(self.path)],
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            with self.assertRaises(AdmissionBusy):
                with self.probe():
                    self.fail("lease overlap")
        finally:
            child.terminate()
            child.wait(timeout=10)
            child.stdout.close()
        with self.probe():
            self.assertTrue(self.path.exists())

    def test_surviving_child_retains_exclusivity(self):
        with self.probe() as fd:
            child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(90)"],
                                     pass_fds=(fd,))
        try:
            with self.assertRaises(AdmissionBusy):
                with self.probe():
                    self.fail("second GPU worker could start")
        finally:
            child.terminate()
            child.wait(timeout=10)
        with self.probe():
            self.assertTrue(self.path.exists())

    def test_direct_worker_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "inherited lease"):
            require_inherited_lease({}, self.path)

    def test_legacy_and_ancestors(self):
        table = (" 123 1 /bin/zsh\n 124 123 python conda run python run_phase140_143.py\n"
                 " 125 124 python run_phase140_143.py\n"
                 " 178 1 /bin/python3 /some path/run_phase138_mode_isolated_memory.py\n"
                 " 180 1 /bin/zsh -c rg run_phase138_mode_isolated_memory.py")
        self.assertEqual(legacy_pids(table, 125), [178])
        def detect():
            raise AdmissionBusy("GPU_LOCK_BUSY: legacy")
        with self.assertRaisesRegex(AdmissionBusy, "legacy"):
            with lease([], self.path, check=detect):
                self.fail("legacy check was bypassed")
        self.assertFalse(self.path.exists())

class AdmissionTests(unittest.TestCase):
    def gate(self, inputs):
        history, waits = {"gpu_gate_attempts": []}, []
        def read():
            v = next(inputs)
            if v > 30:
                raise RuntimeError(f"GPU test blocked: utilization {v}")
            return {"utc": "test", "utilization_pct": v}
        return stable_gpu_gate(history, reader=read, sleeper=waits.append, attempts=4,
                               settle_seconds=4), history, waits

    def test_high_resets_streak_and_settles(self):
        gate, history, waits = self.gate(iter([28,31,30,27]))
        self.assertEqual([r["utilization_pct"] for r in gate["confirmations"]], [30,27])
        self.assertEqual(waits, [4,2,10,2])
        self.assertEqual(len(history["gpu_gate_attempts"][0]), 4)

    def test_fluctuating_never_admitted(self):
        with self.assertRaisesRegex(RuntimeError, "unstable"):
            self.gate(iter([30,40,29,31]))

    def test_entirely_busy(self):
        with self.assertRaisesRegex(RuntimeError, "unstable"):
            self.gate(iter([31,45,73,34]))

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.job = jobs_for("smoke")[0]
        self.manifest = {"source": "a", "dependencies": "b"}
        self.gate = {"confirmations": [{"status":"PASS","utilization_pct":29},
                                       {"status":"PASS","utilization_pct":30}], "utilization_pct":30}
        self.gate["attempts"] = self.gate["confirmations"]
        atomic_json(self.path/"one.json", {"job":self.job,"status":"PASS","output_checksums":{"frame":"a"}})
        (self.path/"one.log").write_text("successful child")

    def completed(self):
        return checkpoint(self.path, "one", self.job, self.manifest, self.gate, 20)

    def read(self, manifest=None):
        return verify(self.path, "one", self.job, manifest or self.manifest)

    def test_partial_and_resume(self):
        (self.path/".partial-interrupted").write_text("{")
        self.assertIsNone(self.read())
        row = self.completed()
        self.assertEqual(self.read()[0], row)
        self.assertEqual([j["mode"] for j in jobs_for("smoke") if j != row["job"]], ["active"])

    def test_corrupt_raw_evidence(self):
        self.completed()
        (self.path/"one.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.read()

    def test_corrupt_checkpoint(self):
        self.completed()
        (self.path/"one.checkpoint.json").write_text("{")
        with self.assertRaises(json.JSONDecodeError):
            self.read()

    def test_source_and_dependency_changes(self):
        self.completed()
        for changed in ({"source":"different","dependencies":"b"},
                        {"source":"a","dependencies":"different"}):
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                self.read(changed)

    def test_output_collision(self):
        with self.assertRaises(FileExistsError):
            atomic_json(self.path/"one.json", {})
        self.assertTrue(json.loads((self.path/"one.json").read_text())["output_checksums"])

    def test_counterbalance(self):
        jobs = jobs_for("140")
        for workload in ("small","fullhd","highres"):
            for density in ("sparse","moderate","dense"):
                subset = [j for j in jobs if j["workload"] == workload and j["density"] == density]
                self.assertEqual([j["mode"] for j in subset], ["auto","active","active","auto"])
                self.assertEqual(len({j["seed"] for j in subset}), 2)

if __name__ == "__main__":
    unittest.main()
