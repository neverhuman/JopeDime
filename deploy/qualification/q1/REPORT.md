# Q1 final attempted qualification — INCOMPLETE

**Q1 disposition: INCOMPLETE. Regression guidance default: OFF under OWNER-R21.**
The signed study reached all 582 logical terminal identities, but only 60 have a valid complete response. The other 522 identities have driver records with `ValueError: backend process failed`, missing the required response fields and known charge counts. The unchanged preregistered analyzer exited 1 with `ValueError: invalid response fields/schema`. It produced no analysis JSON, effect estimate, confidence interval or subgroup effect. This is not a superiority FAIL or a Q1 PASS.

## Frozen study and acceptance

The owner-signed plan is `d71ebeda6fe7c5575025a063807662b66c93c5f743689dc96bc5bae0fe26d19a`: 97 authoritative regression families, one dataset per family, seeds 0/1/2, heuristic and learned guidance, and 24 charged rung-0 admissions per policy/seed. Scientific source is `ccaf14b210b3b55a350218e60f6b49ef76524cf3`; metadata/runtime source is `fa07e1336c8fc7c04994faadf8f6efabded0a195`. Candidate bundle, DOPE, evaluator, vocabulary, protocol and training-family closure remain pinned in the final receipt. Execution used the authenticated CPU evaluator; no GPU, model update, cohort replacement, tuning, new evaluation or retry was introduced during final analysis.

A Q1 PASS requires every signed unit to have a valid complete response before the preregistered primary and subgroup calculations. The 10,000-resample superiority and non-inferiority calculations were not run on this incomplete panel. The R20 census amendment preserves the three-family wide subgroup; its threshold was not changed after outcomes.

| Frozen subgroup | Families | Required by signed plan | Effect / bound |
|---|---:|---:|---|
| large | 27 | 5 | Not computed |
| narrow | 94 | 5 | Not computed |
| small | 70 | 5 | Not computed |
| wide | 3 | 3 | Not computed |

## Execution and response accounting

| Evidence | Terminal identities | Valid complete responses | Unresolved invalid responses |
|---|---:|---:|---:|
| Original ledger, after approved durable-response mapping | 17 | 17 | 0 |
| Previously completed resume prefix | 14 | 14 | 0 |
| Attempt-09 execution | 551 | 29 | 522 |
| Full signed logical panel | 582 | 60 | 522 |

Attempt-09 ended at `2026-11-02T21:37:26.284744+00:00` with `all_resume_units_terminal`, 551/551 terminal units and no resource stop reason. It stopped its remote engine successfully, removed the local/remote daemon records and closed its tunnel. The final resume ledger SHA-256 is `ca7025cc262c3134768f9daea5e31dcc2d22c8afa9ffa7429e5a94f51ac101f5`.

The original infrastructure-interruption FINISHED row is preserved. GOV03 approved recovery of its already-durable 24/24 response (`afd7424379a950a678bdbfd667c665b4067b6a849e9e0be5b21d586d7a66048b`) through append-only mapping `43686b2bc7839853cb74f2a0bbce36e5c40ad3e9e239382a5af1e16c7a30db30`. This resolves an existing original identity and adds zero identities or charges. The original started-only identity overlaps an already-finished resume identity; it must not be counted twice.

There are **1,440 known returned charges across the 60 valid logical responses**, with 1440 completed candidate attempts reported by those responses. The 522 invalid logical responses have unknown charges/completion counts, and prior interrupted or started-only infrastructure work remains unknown. **Total physical charges and evaluator CPU/wall time are not established.** Failures retain their reserved/recorded budget position; they are not zero-cost observations or permission for a reset or free retry.

Attempt-09 controller elapsed time was `1823.201573` seconds. The unchanged post-execution analyzer took `0.314898` seconds and refused the panel. These are controller/analysis elapsed observations, not total evaluator compute. Elo and percentile measurements are unavailable and were not inferred from loss receipts.

## Final evidence and review handoff

- Final receipt: [receipts/q1-attempt09-final.json](receipts/q1-attempt09-final.json), SHA-256 `49fe1da4d8fab088b7d5b3d80f335df2b2cbb6e8b084dec447044292455b8855`.
- Exact command, exit status, source pins, raw ledger hashes and provenance: the final receipt and `/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-final-t2141/`.
- Full invalid-identity/reason inventory: `INVALID-UNITS.json`, SHA-256 `ea9fb91c9e42ebdd6ad16c5b3f5c74201de0879b3b32938d862c7f30e017b10c`. Original raw events and every selection origin remain in `PROVENANCE.json`; no failed row was overwritten or hidden.
- GOV03's unchanged assembly barrier remains `WAITING_FOR_ALL_582_VALID_COMPLETE_RESPONSES`. Its independent review and API01's delegated guidance-default receipt are pending at publication.

Guidance stays **OFF**. No new signature, acceptance waiver, serving promotion or successful qualification is claimed. Any further evaluation or recovery requires its own original-protocol disposition; this report authorizes none. QLT-04 post-execution evaluation/reporting is delivered with an INCOMPLETE scientific result; REL-07 must preserve that disposition rather than treating terminal counts as a Q1 pass.
