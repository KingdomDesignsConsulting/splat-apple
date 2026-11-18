# Phase 128 adapter overhead

The measurement batch completed on the three M4 workloads. The first harness run omitted the exponential-alpha constructor argument and stopped on the documented native restriction. Its raw failure is retained. The corrected run used the pinned disposable runtime.

| Median wall time (ms) | Small | Full HD | High resolution |
|---|---:|---:|---:|
| Scene metadata subset on CPU | 0.003 | 0.032 | 0.015 |
| Evaluated activation | 0.417 | 2.104 | 1.915 |
| Complete scene preparation and validation | 0.361 | 2.981 | 2.892 |
| Camera preparation | 0.222 | 0.300 | 0.234 |
| GPU finite check | 0.222 | 0.318 | 0.267 |
| Host finite-check prototype | 0.003 | 0.011 | 0.005 |
| Native constructor | 0.708 | 12.113 | 11.903 |
| Persistent update | 1.120 | 15.225 | 14.696 |

Independent samples include host dispatch and evaluation boundaries. They cannot be subtracted to derive an exclusive GPU validation cost. The CPU metadata probes measure a subset. Public/direct one-shot order alternated and all RGB pairs matched bitwise. Timing differences across those full renders varied by workload. The host prototype matched the current finite-check verdict on finite, NaN and infinity probes. It can save about 0.22–0.31 ms in that isolated operation.

**Optimization acceptance INCOMPLETE.** The experiment did not put the host check into the adapter or rerun the 65 contracts against a changed adapter. The next comparison should include lazy GPU-produced matrices and all supported matrix dtypes. See [raw summary](overnight_results/phase128_adapter_overhead_1.json).
