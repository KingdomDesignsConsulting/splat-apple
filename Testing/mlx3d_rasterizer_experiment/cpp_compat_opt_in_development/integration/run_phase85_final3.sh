#!/usr/bin/env bash
# Run with Python from gs-mps. Every shell command is echoed for inspection.
set -euxo pipefail
phase85_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$phase85_root"
phase85_integration="$phase85_root/Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration"
phase85_out="${1:-phase85_validation_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$phase85_out" ]]; then echo 'Output already exists' >&2; exit 2; fi
mkdir -p "$phase85_out"
phase85_gpu_gate() {
  phase85_gpu_pct=$(ioreg -r -c AGXAccelerator -d 1 | sed -nE 's/.*"Device Utilization %"=([0-9]+).*/\1/p')
  if [[ ! "$phase85_gpu_pct" =~ ^[0-9]+$ ]] || (( phase85_gpu_pct > 30 )); then
    echo "GPU test skipped: utilization ${phase85_gpu_pct:-unknown}%; limit 30%." >&2
    exit 75
  fi
}
phase85_gpu_gate
python "$phase85_integration/test_phase85_final3.py" --out "$phase85_out/correctness.json"
phase85_gpu_gate
python "$phase85_integration/bench_phase85_final3.py" --mode sort --out "$phase85_out/sort.json"
phase85_gpu_gate
python "$phase85_integration/bench_phase85_final3.py" --mode render --out "$phase85_out/render.json"
echo "Phase 85 complete: $phase85_out"
