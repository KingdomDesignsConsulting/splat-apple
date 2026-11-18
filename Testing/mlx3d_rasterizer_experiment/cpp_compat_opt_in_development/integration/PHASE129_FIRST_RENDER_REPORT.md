# Phase 129 first-render investigation

Four fresh-process observations per workload completed with bitwise first/warm/one-shot RGB parity. Process-first measurements retain existing driver caches.

| Median wall time (ms) | Small | Full HD | High resolution |
|---|---:|---:|---:|
| Import verification and Fast load | 31.2 | 31.3 | 35.1 |
| Persistent constructor | 23.8 | 42.3 | 44.5 |
| First render | 7.4 | 144.1 | 217.3 |
| Warmed persistent frame | 2.0 | 115.7 | 183.9 |
| One-shot after warmup | 3.8 | 134.1 | 205.3 |

First render includes initial buffer sizing, enqueue and synchronization. Metal compilation and allocation could not be separated without lower-level instrumentation. Earlier Phase 95 rehearsal spent extra time and memory, so this batch did not reintroduce automatic prewarming. The counterbalanced active-size follow-up in Phase 133 reduced first-frame time in all four large-workload blocks, with a warmed small-scene regression. It is a workload-specific lead.

**Optimization acceptance INCOMPLETE** for a general first-render change. [Machine-readable results](overnight_results/phase129_first_render_1.json).
