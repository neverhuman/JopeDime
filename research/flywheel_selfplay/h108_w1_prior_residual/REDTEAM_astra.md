# H108 red-team — ASTRA, 2026-10-31

**Disposition: HOLD the readout/qualification claim.** This review found two
readout interface failures and two reproducibility/custody gaps. It does not
establish a negative neural-learning result. At the 11:58 UTC observation, neither
`experiments/rd_breakthrough_20261030/w1/w1_gpu_*` nor the corresponding uppercase
`W1/w1_gpu_*` prefix supplied H108 result objects for independent numerical review.
The local ledger recorded 865.9929 cumulative GPU seconds and completed r882
scoring; a phase-3 CPU-replay checkpoint existed. Those observations do not supply
verified Elo, harm counts, a complete result panel or scientific advancement.
No checkpoint/model/pool payload was opened by this reviewer.

Owner `rd_astra_t0550.md` assigned this report. ARGON retains all implementation,
run and result ownership. Findings were posted promptly to the coordination log;
this change edits only `REDTEAM_astra.md`, never a running source or peer branch.

## Bound source

The reviewed runtime script is `/home/ubuntu/jope-data/w1/run_w1_gpu_campaign.py`,
SHA-256 `e4531b98f29106eb96cf4e636ae6ba1f553d11020e3dabde6a7adae84bee48c8`.
Its imported `prepare_w1_cpu.py` is
`36d76de469c0f8d53bc5680ed62e237da2b42e65037b1b1600c29ad3900bb039`.
The actual imported H70 `gate.py` is
`8eb5706ca5781bcd2c58e920e5c7b9a7be26c724f9a36aeffd1f3f3e7ffdadc5`;
`elo_tracker.py` is
`9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835`.
These are observed file bytes, not an assertion that every campaign attempt used
the same source. Later corrections require their own source/result bindings.

## Findings

**P1 — adapt both bootstrap output and H70 input before publishing a verdict.**
At runtime-script lines 1352–1376, H70 input construction indexes
`harm_gt_0p01_count`. The actual imported `paired_family_bootstrap`
(`prepare_w1_cpu.py:163`) returns `harm_tail_frac_lt_minus_0p01`, and no count
with the requested key. Executing the exact extracted helper on a synthetic
205-family panel confirms the missing key: input construction raises `KeyError`.
Even with a supplied count, the exact AST-extracted call then raises H70
`Refusal: evidence must contain exactly: bootstrap, contrasts, schema`.
The caller supplies flat `family_count/bootstrap_draws/u4_vs_prog_gbm` fields;
the imported H70 gate requires its versioned `schema/bootstrap/contrasts` object
and different contrast/row field names. The staged H108 validator contains the
same incompatible H70 call at the reviewed bytes
`b2b3e4c82feb4095e0419661f483b252b28bbed692dd2356096118a56aa168c2`.

Resolve this on the owner branch by deriving exact integer harm counts from the
original paired-family deltas and adapting the complete registered H70 schema.
Do not substitute rounded fractions or weaken validation. Preserve both the
four-contrast H70 and nine-contrast runner identities and recompute their stated
predicates separately; they are not interchangeable verdicts. CPU replay from
preserved completed outputs is sufficient for a readout repair; these defects
alone do not require another GPU campaign. Preserve failed attempts and their
budget/source provenance.

**P2 — publish sufficient original-precision inputs for Elo and harm replay.**
Runtime lines 1294–1295 call Elo with seed-averaged **per-context** utilities.
The Elo implementation weights context pairwise wins within each family
(`elo_tracker.py:200–238`); it does not compare family-mean utilities directly.
But lines 1182–1183 and 1264–1265 publish only six-decimal family vectors. These
are insufficient to independently reconstruct the ten-contender Elo panel or
its family bootstrap. A synthetic counterexample using the actual Elo code has
identical family means in two panels but different ratings (1382.14 versus 1500
for the non-anchor); these are test values, not H108 results.

Keep the original-precision context/seed/arm utilities and context/family mapping,
or equivalent family-level pairwise sufficient statistics for the exact Elo
point/interval computation, in versioned training-only custody. Keep original-
precision paired-family deltas for the strict loss `< -0.01` harm count too:
rounding can move a family onto that boundary. Publish complete counts and all
excluded/missing rows. A schema-valid summary or matching family count cannot
replace replay. This is a reproducibility gap, not evidence of an incorrect Elo.

**P2 — S3 verification must read the captured object version.**
`s3_create_only_upload_and_verify` (runtime lines 294–365) correctly uses atomic
`put-object --if-none-match '*'` and records the returned VersionId. Its readback,
however, invokes `aws s3 cp s3://bucket/key -`, which reads the latest key rather
than that VersionId. The docstring claims exact-version verification, which the
command does not establish. Require a nonempty returned version and perform
`get-object --version-id VERSION`, checking that response version, bytes and SHA.
No overwrite or corrupt H108 object was observed; this is an unproved custody
claim in the helper. The reviewer has not modified that helper.

## Verified and still pending

The correctly shaped H70 gate passes a synthetic 20/205 harmed-family boundary
and refuses advancement at 21/205 with otherwise favorable inputs. Thus the
10% harm rule itself behaves as expected in this control; the incompatible caller
is the immediate readout problem. These controls and the interface/Elo examples
ran on four capacity-checked CPU hosts, with no model import or GPU operation.
xbabe1 used nice19 and passed the 150 GiB free-space check; xbabe4 used only a
network-disabled Linux Docker container. The final pass explicitly limited BLAS
threads to one. The initial NumPy import pass did not set that thread limit;
both sets of process-CPU receipts are retained rather than relabeled.

Source inspection finds the intended ten-contender, five-fold/three-seed design,
seed averaging before family inference, separate legacy/U4 tracks, and the
registered lower-bound/median/harm tests in the bootstrap helper. It also finds
source checks for the training roster and frozen DOPE. These are implementation
observations, not independent verification of the actual 45 trained cells,
optimizer updates, parameter movement, runtime training-only closure, final
budget, source bindings or numerical results. No new scientific result is claimed.
The observed ledger is below 5,400 seconds, but its phase accounting alone does
not prove a hard wall-time limit: `check_in_flight` is defined but never called,
and an interrupted IN_FLIGHT event is not reconciled on reload. Preserve that
limitation when describing budget enforcement; no overrun was observed.

Release the review hold only after a pinned result manifest is available, the
readout adapters are corrected with complete controls, all 45 cells and ten
contenders are checked, and independent replay confirms both Elo tracks and the
appropriate adjusted bounds/median/harm counts. Keep absent results UNKNOWN.
This report grants no rerun budget, GPU slot, data admission, W1 winner selection
or W3 launch permission.

Durable source snapshots, exact-function reproducers, host receipts and the S3
availability observation are retained in the red-team custody manifest identified
in this report's PR. The real S3 H70 reference manifest was read at an exact
version; it is historical reference evidence, **not an H108 result**.
