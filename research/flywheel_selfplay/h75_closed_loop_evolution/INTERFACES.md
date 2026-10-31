# H75 CPU support interfaces

Protocol authority: [PREREG.md](PREREG.md) and [protocol.json](protocol.json).
These contracts describe required records; they do not certify an execution.
OPUS owns runner integration; GOV03 owns `scoring/`; OBJ02 owns `feedback/`.
Neither support helper may choose arms, change thresholds or launch fits.

## Feedback records (OBJ02)

Use JSON-compatible records with these required fields; missing/unknown
identity is rejected, never inferred from path names:

- `schema`: `jope.h75-closed-loop.outcome/v1`;
  `run_id`, `source_git_sha`, `prereg_sha256`, `execution_manifest_sha256`.
- `cell`: A/B/C/D; `fold`:0..4; `seed`:701/702/703;
  `cycle`:1/2; `phase`: `fit_acquisition`;
  `role`: `DL-Train`; `context_sha256`, `family_id`.
- `table_sha256`, `original_rows_sha256`, `cut_id`,
  `evaluator_sha256`, `vocabulary_sha256`, `evaluation_protocol`;
  `evaluation_scope`: `full_train_cv`; `nq`:0;
  `dope_sha256` matching the protocol.
- `program_sha256`, `dsl`, `parent_program_sha256s`, `operator`,
  `proposal_seed`, `selection_policy`, `selection_probability`;
  `status`: `completed`, `failed`, `timeout` or `unknown`;
  `gate`, `fitness`, `logratio` (null for absent measurements).
- `source_object`: key, version_id and sha256;
  `record_id`: stable execution identity, not a new ID for a retried upload.

The caller supplies a frozen **admission manifest**, not just row-declared
booleans: exact allowed run/cell/fold/seed, fit-context/family roster, forbidden
scoring families/alias components, per-context table/row/cut/evaluator identities,
source object hashes and program hashes. Verify joined values against it;
assert SHA256(UTF-8 DSL)==program_sha256. A helper can validate a supplied
manifest but cannot independently prove that its creator excluded all hidden
source data; its receipt must state that boundary.

Partition by run/cell/fold/seed/context; never concatenate a global target
buffer across folds. Deduplicate exact record IDs; reject conflicting payloads
for the same identity. Check finite completed outcomes and boolean gate.
Failed/timeout/unknown rows remain in an audit output with `target=null` and
`eligible_for_loss=false`; a finite sentinel alone cannot become completed.

For eligible completed rows: `target = fitness` if gate passes, otherwise2.0;
`eligible_for_loss=true` only for D fresh feedback. A/C are shadow only; B is
search-memory only. Preserve original unrounded outcomes and status regardless
of target mapping. No winner-only filtering. Compute
`apparent_hwm = completed and gate and fitness < pre_cycle_best_fitness - 1e-6`
using the manifest's same-context pre-cycle incumbent. This is an apparent
original-cut label, never a confirmed or serving HWM. Output program-keyed
records plus counts/hashes; OPUS joins canonical feat412/proj16 and verifies
alignment before training. Unknown feature/status provenance blocks that row.

Target/history masks belong to OPUS: feedback rows must not become revealed
training anchors or enter their own feature channels. OBJ02 tests malformed
identity, held-family entry, duplicate conflict, gate-failure mapping and
failed-record masking with tiny fixtures. It must not load datasets/checkpoints
or write the shared HWM S3 index.

## Score records (GOV03)

Use `schema=jope.h75-closed-loop.score/v1`, run/prereg/execution manifest hashes,
`panel` (`legacy` or `confirmed_U8`), `policy_id`, `fold`, `seed`,
`context_sha256`, `family_id`, `role=DL-Train`, `selected_program_sha256`,
`evaluation_identity_sha256`, `status=completed` and finite `utility`.
For legacy, utility is -fitness or -2 according to the actual gate. For
confirmed_U8, it is the mean of the three frozen confirmation-cut utilities
of the program selected before those cuts were read; include the three cut
identities. Never average favorable cuts only or rerank using confirmation.

The caller supplies a frozen complete panel and exact context/family/fold/seed
roster with paired evaluation identities. Legacy panel:
A,B,C,D,replay_B,replay_D. Confirmed U8 panel: A,B,C,D,G,R. Reject missing,
duplicate/conflicting, nonfinite or foreign-role rows; never complete-case
silently or fill a missing utility with zero. If execution is incomplete,
return counts plus INCOMPLETE, not an official full-panel Elo.

Average seeds within context, contexts within family, then use canonical
`python/src/jope/north_star/elo_tracker.py` with A anchored1500, tie1e-4,
prior.01,400iterations. Distinguish the context-mean `val_reg_inner` from equal
family-weighted mean. Bootstrap paired families with shared indices; intervals
are descriptive conditional development evidence. Return unrounded scores,
family counts, W/T/L, source hash and resampling settings. The scorer does not
apply a promotion decision; OPUS/ASTRA apply the seven-contrast preregistered
conjunction. Reject panel mixing and method-generated fake family replication.
