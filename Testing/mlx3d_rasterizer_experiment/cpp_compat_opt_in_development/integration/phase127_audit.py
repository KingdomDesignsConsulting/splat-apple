import ast
import json
from pathlib import Path
import sys
from qualification_common import HERE, ROOT, EXPECTED_INSTALLED, invariants, write_report

def main():
    current = invariants()
    performance = json.loads((HERE / "qualification_evidence/phase125/phase125_integration_performance_4.json").read_text())
    release = json.loads((HERE / "qualification_evidence/phase126/phase126_release_gate_3.json").read_text())
    assert performance["status"] == release["status"] == "PASS"
    assert performance["invariants"] == release["invariants"] == current
    assert len(performance["boundaries"]["cases"]) == 6
    assert len(performance["gpu_gates"]) == 13
    assert all(0 <= row["utilization_pct"] <= 30 for row in performance["gpu_gates"])
    assert {w["name"] for w in performance["workloads"]} == {"small", "fullhd", "highres"}
    assert all(len(w["blocks"]) == 4 and all(b["status"] == "PASS" for b in w["blocks"])
               for w in performance["workloads"])
    assert release["decision"] == "CONDITIONALLY READY" and not release["blockers"]
    assert len(release["checks"]) == 6 and all(r["status"] == "PASS" for r in release["checks"])
    renderer = ast.parse((ROOT / "mlx_gs/renderer/renderer.py").read_text())
    render = next(node for node in renderer.body if getattr(node, "name", "") == "render")
    assert ast.literal_eval(render.args.defaults[-1]) == "python"
    write_report(Path(sys.argv[1]), {
        "phase": 127, "status": "PASS", "decision": release["decision"],
        "reviewed_head": "b491bcdc84370267ea375ed8312d30ec2ae2c4eb",
        "invariants": current, "expected_installed_fast": EXPECTED_INSTALLED,
        "inputs": ["phase125_integration_performance_4.json", "phase126_release_gate_3.json"],
        "scope": "Mac mini M4, source checkout, forward rendering",
        "github": {"draft": True, "merged": False, "inline_review_threads": 0,
                   "combined_commit_status_entries": 0, "ci_pass": "unestablished"},
    })

if __name__ == "__main__":
    main()
