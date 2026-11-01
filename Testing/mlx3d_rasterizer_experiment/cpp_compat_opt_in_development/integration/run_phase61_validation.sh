#!/usr/bin/env bash
# Phase 61: reproducible post-push validation of the default optimized renderer.
# Run from anywhere inside the feature worktree; preserves pre-existing files.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$(git branch --show-current)" != "feature/cpp-compat-renderer" ]]; then
  echo "ERROR: wrong branch" >&2; exit 2
fi
OUT="${1:-phase61_validation_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$OUT" ]]; then echo "ERROR: destination exists: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
INTEGRATION="$ROOT/Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration"
git show 4dfde5d96f7dca3e1da1535915facc020ecc7f5c:Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration/mlx3d/splatting/fast.py > "$OUT/frozen_fast_reference.py"
export OUT
python - <<'PY'
import hashlib,os
from pathlib import Path
p=Path(os.environ["OUT"])/"frozen_fast_reference.py"
h=hashlib.sha256(p.read_bytes()).hexdigest()
expected="b6aa0444116920d1ae5e526eaf5821e4970bf681c74ce04096fb7f57ae9c11d1"
if h!=expected: raise SystemExit(f"Reference SHA mismatch: {h}")
print("PASS frozen source SHA",h,flush=True)
PY
{
  echo "HEAD=$HEAD"
  echo "timestamp=$(date -u +%FT%TZ)"
  uname -a
  python --version
  git status --short
} > "$OUT/environment.txt"
python "$INTEGRATION/test_phase60_radix.py" 2>&1 | tee "$OUT/radix_test.log"
python "$INTEGRATION/test_phase58_regression.py" --reference "$OUT/frozen_fast_reference.py" 2>&1 | tee "$OUT/regression_test.log"
for n in 100000 700000; do
  echo "Starting paired $n / 1920x1080"
  python "$INTEGRATION/bench_phase59_adaptive.py" \
    --reference "$OUT/frozen_fast_reference.py" \
    --n "$n" --width 1920 --height 1080 \
    --rounds 20 --survey 45 \
    --candidate-sort auto --candidate-ranges auto --candidate-algorithm auto \
    --out "$OUT/paired_${n}.json" 2>&1 | tee "$OUT/paired_${n}.log"
done
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ["OUT"])
for n in (100000,700000):
 d=json.loads((p/f"paired_{n}.json").read_text())
 print(f"{n}: steady={d['steady']} varied={d['varied_camera']} memory={d.get('process_memory_after')}")
PY
echo "PASS Phase 61 outputs stored in $OUT"
