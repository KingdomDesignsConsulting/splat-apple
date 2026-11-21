#!/usr/bin/env bash
# Run from the dedicated Phase 140 worktree. Pass a Phase 140 artifact
# directory to resume; every report receives a fresh path.
set -euo pipefail
set -x
integration_dir=Testing/mlx3d_rasterizer_experiment/cpp_compat_opt_in_development/integration
batch_dir=$(mktemp -d "$PWD/phase140-143-batch.XXXXXXXX")
resume_args=()
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [phase140_artifact_dir]" >&2
  exit 2
fi
if [ "$#" -eq 1 ]; then
  resume_args=(--resume-from "$1")
fi
conda run --no-capture-output -n splat_apple python "$integration_dir/run_phase140_143.py" --phase 140 --attest-m4 "${resume_args[@]}" --out "$batch_dir/phase140.json"
python3 "$integration_dir/analyze_phase141_break_even.py" "$batch_dir/phase140.json" --out "$batch_dir/phase141.json"
python3 "$integration_dir/configure_phase142_policy.py" "$batch_dir/phase141.json"
python3 "$integration_dir/test_phase142_policy.py" -v
conda run --no-capture-output -n splat_apple python "$integration_dir/run_phase143_balanced.py" --phase 143 --attest-m4 --out "$batch_dir/phase143.json"
python3 "$integration_dir/analyze_phase143_combined.py" "$batch_dir/phase143.json" --out "$batch_dir/phase143_comparison.json"
