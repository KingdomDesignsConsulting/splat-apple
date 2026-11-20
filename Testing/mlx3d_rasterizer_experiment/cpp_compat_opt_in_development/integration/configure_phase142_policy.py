"""Record or apply Phase 141's narrow trial bands before starting Phase 143."""
import argparse
import json
from pathlib import Path
import re
from phase140_gpu_lease import lease

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if data["status"] != "PASS":
        parser.error("Phase 140/141 evidence is incomplete. Active bands remain disabled.")
    bands = tuple(tuple(row) for row in data["trial_bands"])
    if not all(len(row)==5 and row[:2] in ((1920,1080),(2560,1440))
               and all(type(v) is int and v > 0 for v in row)
               and row[3] <= row[4] for row in bands):
        parser.error("Unexpected trial band")
    path = Path(__file__).with_name("phase142_combined.py")
    with lease([]):
        current = path.read_text()
        new, count = re.subn(r"(?m)^ACTIVE_BANDS = .*$", "ACTIVE_BANDS = " + repr(bands), current)
        if count != 1:
            parser.error("The policy has no single active-band declaration")
        if current != new:
            path.write_text(new)
    print("Phase 142 trial active bands:", bands)

if __name__ == "__main__":
    main()
