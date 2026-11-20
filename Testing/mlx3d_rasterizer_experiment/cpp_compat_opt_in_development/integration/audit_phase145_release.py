"""Phase 145: match held release baselines and exact experimental sources to evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from qualification_common import (
    HERE, ROOT, EXPECTED_INSTALLED, EXPECTED_LOCAL, EXPECTED_WHEEL,
    command, digest, invariants, write_report,
)
from phase140_gpu_lease import verify
from run_phase140_143 import label

RELEASE = "006330276a6e6737cea22f7bdd2ce5e21c2286e8"
PHASE143 = "0e13dbe78d5e94219e5e43bf1c844f0a0165458a"

def git_bytes(ref, path):
    rc, data, _ = command(["git","show",f"{ref}:{path}"])
    if rc:
        raise RuntimeError(f"Cannot read {ref}:{path}")
    return data.encode()

def read_json(path):
    return json.loads(path.read_text())

def audit(lifecycle, loader):
    release_adapter = "mlx_gs/renderer/mlx3d_cpp_compat.py"
    release_dispatch = "mlx_gs/renderer/renderer.py"
    frozen_fast = str((HERE/"mlx3d/splatting/fast.py").relative_to(ROOT))
    shas = {str(path):hashlib.sha256(git_bytes(RELEASE,path)).hexdigest()
            for path in (release_adapter,release_dispatch,frozen_fast)}
    current = invariants()
    assert shas[frozen_fast] == current["fast_sha256"] == EXPECTED_LOCAL
    assert digest(ROOT/release_dispatch) == shas[release_dispatch]
    assert current["adapter_sha256"] == hashlib.sha256(git_bytes(PHASE143,release_adapter)).hexdigest()
    assert digest(HERE/"phase142_combined.py") == hashlib.sha256(git_bytes(
        PHASE143,str((HERE/"phase142_combined.py").relative_to(ROOT)))).hexdigest()
    for name in ("setup.py","setup_mlx.py"):
        assert digest(ROOT/name) == hashlib.sha256(git_bytes(RELEASE,name)).hexdigest()
    scene_gate = HERE/"qualification_evidence/overnight/phase131_m4_extended_1"
    release_proofs = {}
    for name in ("legacy","loader","api","performance","retained"):
        slot = scene_gate/name
        checkpoint = read_json(slot/"checkpoint.json")
        path = slot/checkpoint["attempt"]/"result.json"
        assert digest(path) == checkpoint["result_sha256"]
        row = read_json(path)
        assert row["status"] == "PASS"
        assert row["invariants"]["adapter_sha256"] == shas[release_adapter]
        release_proofs[name] = dict(path=str(path.relative_to(HERE)), sha256=digest(path))
        if name=="legacy":
            assert row["native_build"]["returncode"] == 0
            assert {r["backend"] for r in row["checks"] if r["status"]=="PASS" and r["mlx3d_absent"]} == {"python","cpp","missing_opt_in"}
        if name=="api":
            assert len(row["contracts"]["checks"])==65
            assert all(r["status"]=="PASS" for r in row["contracts"]["checks"])
            assert row["api_gate"]["status"]=="PASS" and len(row["api_gate"]["checks"])==5
        if name=="performance":
            assert len(row["boundaries"]["cases"])==6
            assert len(row["workloads"])==3 and all(len(x["blocks"])==4 for x in row["workloads"])
    installation=read_json(HERE/"qualification_evidence/phase121/phase121_summary.json")
    assert installation["status"]=="PASS" and len(installation["steps"])==7
    assert installation["wheel_sha256"]==EXPECTED_WHEEL
    assert installation["wheel_fast_sha256"]==EXPECTED_INSTALLED
    checked_loader=read_json(loader)
    assert checked_loader["status"]=="PASS" and len(checked_loader["checks"])==10
    assert checked_loader["invariants"]==current
    phase143=HERE/"qualification_evidence/phase140/phase143_completed"
    artifacts=phase143/"phase143_artifacts"
    manifest=read_json(artifacts/"manifest.json")
    assert manifest["invariants"]==current
    assert manifest["scripts"]["phase142_combined.py"]==digest(HERE/"phase142_combined.py")
    compare=read_json(phase143/"phase143_comparison.json")
    assert compare["correctness_and_matrix_status"]=="PASS" and compare["contracts_passed"]==65
    life=read_json(lifecycle)
    assert life["manifest"]["invariants"]==current
    assert life["manifest"]["scripts"]["phase142_combined.py"]==digest(HERE/"phase142_combined.py")
    verified_lifecycle=0
    if life["status"]=="PASS":
        for job in life["manifest"]["jobs"]:
            check,result=verify(lifecycle.with_name(lifecycle.stem + "_artifacts"),label(job),job,life["manifest"])
            assert len(result["frames"])==31 and all(len(v)==64 for v in result["output_checksums"].values())
            verified_lifecycle+=1
        assert verified_lifecycle==8
    return dict(status="PASS" if life["status"]=="PASS" else "INCOMPLETE",
         release_ref=RELEASE, release_source_sha256=shas,
         release_qualification="CONDITIONALLY READY for tested Mac M4 source checkout",
         release_proofs=release_proofs, experimental_baseline=PHASE143,
         experimental_invariants=current, policy_sha256=digest(HERE/"phase142_combined.py"),
         installed_fast_sha256=EXPECTED_INSTALLED, checked_wheel_sha256=EXPECTED_WHEEL,
         loader_status=checked_loader["status"], new_loader_report_sha256=digest(loader),
         baseline_verification="Retained exact source matches; no fresh release GPU rerun",
         experimental_contracts=compare["contracts_passed"], experimental_comparison=compare,
         lifecycle_workers=verified_lifecycle,lifecycle_report_sha256=digest(lifecycle),
         disposition="No experimental runtime integration into qualified release branch")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lifecycle",required=True,type=Path)
    p.add_argument("--loader",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    r=audit(a.lifecycle,a.loader)
    write_report(a.out,r)
    print("RESULT PHASE145",r["status"])
