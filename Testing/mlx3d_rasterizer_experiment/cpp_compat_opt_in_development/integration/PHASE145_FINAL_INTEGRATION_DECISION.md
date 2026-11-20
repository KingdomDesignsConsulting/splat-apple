# Phase 145 final integration decision

**PASS for the source-matched integration audit.** Release head [0063302](https://github.com/KingdomDesignsConsulting/splat-apple/commit/006330276a6e6737cea22f7bdd2ce5e21c2286e8) remains **CONDITIONALLY READY for the tested M4 forward-rendering source checkout**. No experimental runtime change was added to that checkout. Draft PR #1 remains unmerged.

| Scope | Evidence and decision |
|---|---|
| Exact release source | Phase 121–126 and 131 archived runtime evidence verified against the release adapter hash. Legacy Python and newly built CPP ran with MLX3D absent. Installation, loader guards, 65 API contracts, five API checks, six sort boundaries and the 12-block bitwise performance matrix passed. These are retained executions on identical release source, not fresh release GPU runs. |
| Host camera validation | Separate Phase 128 candidate and exact adapter digest, 65 contracts and 12 paired worker comparisons, small-scene benefit. Preserve as a reviewable experimental patch. |
| Narrow adaptive scatter512 | The already tested experimental adapter and selector are byte-identical to Phase 143 at [0e13dbe](https://github.com/KingdomDesignsConsulting/splat-apple/commit/0e13dbe78d5e94219e5e43bf1c844f0a0165458a). The 65 API contracts, five actual dispatch boundaries, 60 bitwise rendering workers and fallbacks passed there. Ten fresh CPU loader lifecycle checks passed on the current experimental source. Automatic active sizing remains disabled. |
| Reserve reset | Phase 144 experimental harness at [cd0fd1a](https://github.com/KingdomDesignsConsulting/splat-apple/commit/cd0fd1a8277711d1c8654eb9b4fa6391279735f7) reproduced an 89.104 ms median first-small-frame reduction. Report as a transition-only prototype. Production update integration and broader regressions remain unqualified. |

| Measured comparison against control | First frame Δ | Warm frame Δ | Peak MLX Δ | Retained MLX cache Δ |
|---|---:|---:|---:|---:|
| Phase 143 experimental selector, small | −0.159 ms | −0.288 ms | 0.0 MiB | 0.0 MiB |
| Phase 143 experimental selector, moderate Full HD | −4.838 ms | −9.757 ms | −139.1 MiB | −265.3 MiB |
| Phase 143 experimental selector, high resolution | −3.669 ms | +0.733 ms | 0.0 MiB | +1.3 MiB |
| Phase 144 reset, 640×360 after large scene replacement | −89.104 ms replacement | −0.531 ms | 0.0 MiB over the full cycle | about +0.002 MiB after release |

Four matched blocks per comparison. Peak, active, cache, and RSS are process-level data, with no per-kernel memory attribution. Full HD adaptive and explicit scatter512 use the same specialized route in the qualifying window. These deltas cannot be added or transferred to arbitrary scenes. The Phase 144 report contains the separate startup, restore and memory chronology.

## Source and supported packaging

- Release adapter SHA-256 `e8f9079908e0d08b28f4af125ed544c76d25b23ed572a273b8b20316db7192ec`.
- Experimental host adapter SHA-256 `2554ff81b3ae12a40a1b7cc07b6cb79769dc0c4036216cf3bf6e8e91ada6425e`.
- Experimental selector SHA-256 `73eaf01a1cc5595efac68395fef8fe4e7a2caef5e904e56fdb0f6b31c74bd783`.
- Repository Fast SHA-256 `56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710`.
- Installed Fast SHA-256 `4ef36ac2cea20b52c5b72684b18cdcedea24c472b73a6326a26027e402fdbcbc`.
- Verified MLX3D wheel SHA-256 `bdfdffe97d2a933afb6bdac2c01ae609c119bf1e22fe129469f4ab03a59439bf`.

The legacy dispatch file, native extension setup and protected Fast snapshot match the release source. Both adapter entry points default to auto. Native Fast defaults to MLX. The explicit selector falls back for an unattested device or out-of-band geometry and remains separate from ordinary rendering. APIs, source checkout loading, and pinned dependencies are documented in [CPP_COMPAT_API.md](CPP_COMPAT_API.md) and the [installation guide](MLX3D_REPRODUCIBLE_INSTALL.md). The adapter resolves its protected source via checkout-relative paths. The extension build is separate from this opt-in. Disposable environments verified wheel identity and left the existing installation intact. Benchmark paths and artifacts are evidence, not distributable runtime inputs.

The experimental host-camera and Full HD sorting policies are ready for review as separately scoped changes, with focused integration checks required for any new production diff. The original tested M4 configuration remains the release candidate. General production readiness, other Apple Silicon, training or gradients, and concurrent shared renderer use are outside qualification. No default promotion is supported.

[Reproducible Phase 144 experiment](PHASE144_FIRST_FRAME_CACHE_FINDINGS.md) · [Machine-readable audit and archived evidence](qualification_evidence/phase144_145/). No further GPU work is required for this audited source decision.
