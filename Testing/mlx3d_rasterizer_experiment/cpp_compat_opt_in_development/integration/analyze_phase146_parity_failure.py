#!/usr/bin/env python3
"""Offline Phase146 parity forensics: read existing worker JSON, run NO GPU work."""
import argparse
import json
from pathlib import Path
import statistics

def analyze(artifacts):
    blocks = []
    mismatches = []
    for block in range(4):
        pair = {}
        for mode in ("keep", "guard"):
            file = artifacts / f"lifecycle_transition_moderate_{block}_{mode}.json"
            data = json.loads(file.read_text())
            assert data["status"] == "PASS" and data["job"]["block"] == block
            assert data["job"]["mode"] == mode and data["job"]["kind"] == "lifecycle"
            assert len(data["frames"]) == 31, (file, len(data["frames"]))
            pair[mode] = data
        keep = {x["tag"]:x for x in pair["keep"]["frames"]}
        guard = {x["tag"]:x for x in pair["guard"]["frames"]}
        assert keep.keys() == guard.keys(), f"Frame matrix mismatch block {block}"
        rows = []
        for tag in keep:
            a, z = keep[tag], guard[tag]
            same = a["rgb"] == z["rgb"]
            item = {"block": block, "frame": tag, "parity": "MATCH" if same else "DIFF",
                    "camera": a["camera"], "camera_guard": z["camera"],
                    "keep_rgb": a["rgb"], "guard_rgb": z["rgb"],
                    "keep_ms": a["wall_ms"], "guard_ms": z["wall_ms"],
                    "delta_ms": z["wall_ms"]-a["wall_ms"],
                    "keep_counts": a["counts"], "guard_counts": z["counts"],
                    "keep_routes": a["routes"], "guard_routes": z["routes"],
                    "keep_inherited": a["inherited"], "guard_inherited": z["inherited"],
                    "guard_decision": z["guard_decision"],
                    "keep_reserve_after": a["active_after_frame_reserve"],
                    "guard_reserve_after": z["active_after_frame_reserve"],
                    "keep_memory": a["memory"], "guard_memory": z["memory"]}
            rows.append(item)
            if not same:
                mismatches.append(item)
        blocks.append({"block":block, "matched":sum(x["parity"]=="MATCH" for x in rows),
                       "total":len(rows), "first_mismatch":next((x for x in rows if x["parity"]=="DIFF"),None),
                       "frames":rows})
    return {"status":"PARITY_PASS" if not mismatches else "PARITY_FAIL",
            "blocks":blocks, "mismatch_count":len(mismatches),
            "first_mismatch_by_block":[{"block":b["block"],"first_mismatch":b["first_mismatch"]} for b in blocks],
            "mismatch_frames":[{"block":x["block"],"frame":x["frame"]} for x in mismatches]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.out and args.out.exists():
        parser.error(f"Output exists; refusing overwrite: {args.out}")
    result=analyze(args.artifacts)
    print(f"PHASE146 PARITY FORENSICS {result['status']} -- {result['mismatch_count']} differing frames")
    for block in result["blocks"]:
        first=block["first_mismatch"]
        print(f"BLOCK {block['block']} {block['matched']}/{block['total']} frames bitwise matched")
        if first:
            print("  FIRST",first["frame"],"keep",first["keep_rgb"],"guard",first["guard_rgb"])
            print("  CAMERA",first["camera"],"COUNTS",first["keep_counts"],first["guard_counts"])
            print("  ROUTES",first["keep_routes"],first["guard_routes"])
            print("  RESERVE",first["keep_inherited"],first["guard_inherited"],
                  "guard_decision",first["guard_decision"])
            print("  TIMING DELTA",round(first["delta_ms"],3),"ms")
    if args.out:
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        print("WROTE",args.out)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
