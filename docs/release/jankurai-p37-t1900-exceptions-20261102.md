# P37 exact loader dispositions — 2026-11-02

These exact-file Jankurai dispositions cover only the HLT-033 findings assigned
to `luna_jk_dup_enc_2_x1` by watcher T1900. They do not change Gitleaks or Python
test behavior, and they do not include any directory or glob.

| Exact file | Finding fingerprint(s) | Integrity evidence |
| --- | --- | --- |
| `research/flywheel_selfplay/h75_closed_loop_evolution/scoring/score_panel.py` | `sha256:89b945a9d3126fa68266767d3dc3b3852f186d701290e90d0668af4e3f28717a` | `load_reference` reads the supplied helper bytes and compares SHA-256 with `ELO_SHA256` before compiling or executing them. `h75_closed_loop_evolution/SOURCE_PINS.json` pins `python/src/jope/north_star/elo_tracker.py` to that same digest. |
| `research/flywheel_selfplay/h94_w6_prospective_hedge/run_prospective_hedge.py` | `sha256:b35dd3972fefebb20ecf3dd123add56a84e05708ee8e20860cf9e2e693e72eae`, `sha256:b9f6645cbee38489ed72b809e3bc0259e8d026b9f01133eb9ed0028ab5953589` | The context model pickle is SHA-256 checked against `CTX_MODELS_SHA` before loading. The program model pickle is now checked against `PROG_MODELS_SHA` before loading; a newly fitted cache is also checked before use. The H94 run receipt records the same program-model digest `5938a35497ccde1a1489c4d6b13d122196aed16739a3272afea06d57d5fff692`. |
| `research/flywheel_selfplay/h96_w4_active_reveal/run_w4.py` | `sha256:b7ada47bcb2a2597cc0745df16d5ff18fd73fbb65d0781139acd82e60b855462` | The program-model pickle is checked against `PROG_MODELS_SHA` before loading. The retained H96 dry-run receipt records that same digest. |

The pinned bytes remain the only accepted inputs at these load sites. H94 now
fails closed if the generated program-model cache has any other digest. H228's
`research/flywheel_selfplay/h228_family_invariant/runner.py` is intentionally
not included: it is listed by `[[evidence_snapshot]]` and remains byte-for-byte
untouched under the user evidence-policy pointer. Its HLT-033 finding remains
visible because evidence snapshots do not exempt HLT-033.
