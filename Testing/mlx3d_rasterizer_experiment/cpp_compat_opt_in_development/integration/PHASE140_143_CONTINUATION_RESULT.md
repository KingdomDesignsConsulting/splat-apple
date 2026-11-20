# Phase 140–143 continuation

All workers used the shared machine-wide lock and two consecutive GPU readings at or below 30%.

| Phase | Result | New verified | Reused | Busy readings | Peak admitted |
|---|---|---:|---:|---:|---:|
| phase140 | PASS | 31 | 5 | 18 | 30% |
| phase143 | PASS | 61 | 0 | 23 | 30% |

## Phase 141 active sizing screen

Deltas are active minus original auto. Negative times favor active sizing.

| Workload | Density | Warmed difference (ms) | Peak MLX difference (MiB) | Retained cache difference (MiB) | Screen |
|---|---|---:|---:|---:|---|
| fullhd | dense | -11.833 | -268.8 | +1393.5 | reject |
| fullhd | moderate | +2.286 | -219.7 | +434.3 | reject |
| fullhd | sparse | +0.082 | -90.2 | +430.4 | reject |
| highres | dense | -0.151 | -464.6 | +5.1 | reject |
| highres | moderate | +6.685 | -339.1 | -48.7 | reject |
| highres | sparse | -4.437 | -118.9 | +211.7 | reject |
| small | dense | +0.464 | -0.6 | -0.4 | reject |
| small | moderate | +0.373 | -0.6 | -0.3 | reject |
| small | sparse | +0.755 | -0.2 | -0.1 | reject |

Phase 141: **PASS**. Configured trial bands: `[]`.

## Phase 143 mode-isolated comparison

65/65 contracts passed. Matrix and dispatch status: **PASS**.

| Workload | Mode | Blocks | Warmed Δ (ms) | First-frame Δ (ms) | Peak MLX Δ (MiB) | Cache Δ (MiB) |
|---|---|---:|---:|---:|---:|---:|
| fullhd | active | 4 | +4.664 | -11.691 | -218.9 | +1319.6 |
| fullhd | adaptive | 4 | -1.872 | -8.019 | -139.1 | -265.3 |
| fullhd | auto | 4 | +0.000 | +0.000 | +0.0 | +0.0 |
| fullhd | combined | 4 | -9.757 | -4.838 | -139.1 | -265.3 |
| fullhd | scatter512 | 4 | -9.248 | -2.796 | -139.1 | -265.0 |
| highres | active | 4 | +11.301 | -18.016 | -355.0 | -12.6 |
| highres | adaptive | 4 | +0.879 | -2.938 | +0.0 | +145.3 |
| highres | auto | 4 | +0.000 | +0.000 | +0.0 | +0.0 |
| highres | combined | 4 | +0.733 | -3.669 | +0.0 | +1.3 |
| highres | scatter512 | 4 | +0.712 | -3.754 | +0.0 | +128.2 |
| small | active | 4 | +0.258 | -0.479 | -0.6 | -0.3 |
| small | adaptive | 4 | -0.165 | -0.511 | +0.0 | +0.0 |
| small | auto | 4 | +0.000 | +0.000 | +0.0 | +0.0 |
| small | combined | 4 | -0.288 | -0.159 | +0.0 | +0.0 |
| small | scatter512 | 4 | -0.199 | +0.199 | +1.0 | +0.9 |

## Qualification decision

**PASS for the scoped experimental sorting policy on the tested M4.** Automatic active sizing remains disabled. No density met the Phase 141 screen, and Phase 143 used no active bands. All 60 rendering workers matched the paired auto RGB checksums across initial, warm, replacement, and restored frames. The original 65 API contracts and five native dispatch boundaries passed. There were 97 verified checkpoints across Phases 140 and 143, including the five reused checkpoints. The two admitted readings were at least two seconds apart and the minimum admitted free system memory was 63%.

On the moderate Full HD fixture, the experimental wrapper improved in all four independent blocks (−11.179 to −6.313 ms versus auto). It used the same scatter512 route as the explicit comparison and produced a similar gain. The trace selected that route at the inclusive 32M and 34M endpoints and fell back immediately outside them and at 2560×1440. High-resolution wrapper timing was mixed, with a median +0.733 ms versus auto. Stable small workloads used automatic sorting, while the first small replacement frame could inherit the old reserve and use radix4 before the resize took effect. Full HD restores returned to the specialized route when the actual sorted capacity qualified.

The recommendation is a narrow, opt-in Full HD sorting experiment. Existing release defaults remain in place. The data do not qualify automatic active sizing or support attributing process-wide memory to an individual kernel. The full raw evidence includes active memory, RSS, captured selection times, and gate readings. PR #1 remains draft and unmerged.

Evidence: [complete density and break-even reports](qualification_evidence/phase140/phase140_density_completed/) and [Phase 143 reports and artifacts](qualification_evidence/phase140/phase143_completed/).
