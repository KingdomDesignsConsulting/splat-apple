#!/usr/bin/env bash
# Resume Phase 85 render validation without repeating successful prior stages.
# Usage: bash run_phase85_resume_render.sh <existing_phase85_validation_directory>
set -euo pipefail
root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$root"
integration="$root/Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration"
out="${1:?Pass the existing phase85_validation_YYYYMMDD_HHMMSS directory}"
if [[ ! -f "$out/correctness.json" || ! -f "$out/sort.json" ]]; then
  echo "Expected correctness.json and sort.json in $out; refusing incomplete resume" >&2
  exit 2
fi
if [[ -e "$out/render.json" ]]; then
  echo "Render output already exists at $out/render.json; refusing overwrite" >&2
  exit 2
fi
# Sample an idle GPU over consecutive checks, rather than a single stale
# utilization reading immediately after the sorting workload.
consecutive=0
for attempt in {1..24}; do
  pct="$(ioreg -r -c AGXAccelerator -d 1 | sed -nE 's/.*"Device Utilization %"=([0-9]+).*/\1/p' | head -n 1)"
  if [[ "$pct" =~ ^[0-9]+$ ]] && (( pct <= 30 )); then
    consecutive=$((consecutive+1))
    if (( consecutive >= 2 )); then
      echo "GPU idle confirmed on two consecutive samples (${pct}% last)."
      python "$integration/bench_phase85_final3.py" --mode render --out "$out/render.json"
      echo "Phase 85 renderer validation complete: $out/render.json"
      exit 0
    fi
  else
    consecutive=0
  fi
  echo "GPU utilization ${pct:-unknown}% (attempt $attempt/24); awaiting two <=30% readings."
  sleep 5
done
echo "GPU did not remain at or below 30%; renderer benchmark NOT run. Retry when idle." >&2
exit 75
