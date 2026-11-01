# H148: cross-fit checkpoint selection

**COMPLETE — RULE_NEUTRAL.** Both selectors retained r882 in all 15 cells
(30/30 step 0 decisions). XFIT minus VALSEL on the outer assessment panel is
0.000000, 95% family-bootstrap CI [0.000000,0.000000]; both have anchored Elo 1500
with CI [1500,1500]. The run executed all 600 planned updates; this result does
not mean training was skipped.

## Hypothesis and setup

Does selecting continuation checkpoints on a larger family panel generalize
better than selecting on 91 DL-Val contexts? The [preregistration](PREREG.md)
froze 1,219 real-S3 TRAIN contexts / 205 families into five groups of 41
(fold_seed 148140). For each outer A and seed 148101–148103, `B = (A + 1) % 5` selects
and the other three folds (123 families) fit 40 AdamW updates (lr 1e-4,
wd 1e-4) from an owned r882 copy. Snapshots 0/10/20/30/40 serve both rules, so 15 cells consume 600 updates.
Assessment families are excluded from fitting and selection in their own cell;
they can occur in other cells' fit/selection folds. Every cell resets r882 and
optimizer state. Only 47 adapter tensors / 29,211 parameters are trainable.

XFIT selects on B; VALSEL selects on 91 DL-Val contexts/19 families. Both require
best snapshot equal-family gain > 0.001 (threshold ties within 1e-12 rejected) and
paired 95% family-bootstrap lower bound > 0; otherwise 0. Both decisions and their
hash are written before scoring A. Native utility is minus original fitness,
or -2 for a failed gate. Original labels and evaluation utilities stay intact.

## Outcome and metrics

| Rule | Selected step 0 | Outer-A gain vs r882 (95% CI) | Outer-A Elo (95% CI) | DL-Val gain vs r882 (95% CI) |
| --- | --- | --- | --- | --- |
| XFIT | 15/15 | 0 [0,0] | 1500 [1500,1500] | 0 [0,0] |
| VALSEL | 15/15 | 0 [0,0] | 1500 [1500,1500] | 0 [0,0] |

All three seeds have identical selected-policy utility panels. Outer-A covers
1,219 contexts / 205 families exactly once per seed; equal-family utility is
1.385355433 for all three rules (r882/XFIT/VALSEL), with 17/1,219 invalid picks
per seed. DL-Val equal-family utility is 1.153615277, with 0/91 invalid picks.
The initial 91-context r882 regression check passed exactly: context mean
1.1377941898164525, maximum per-context error 0 and mean error 0. Context and
family means are different summaries.

Primary intervals use 10,000 shared family draws and the median seed contrast;
Elo uses the canonical 400-iteration BT MM, prior 0.01, tie margin 1e-4, r882=1500,
and 1,000 shared family draws. These zero-width intervals reflect identical
selected baseline policies, rather than uncertainty about a new generator.
All 75 saved snapshots were independently hash-checked; 15/15 final adapters
changed from their identical r882 starts (L2 drift 0.4169146–0.4592349). DOPE
`c51a56d0` and the complete non-trainable-state hash `8d1070fa` remained invariant.

## Verification and custody

The single xbabe2 CPU run used nice 19 / math 1. Worker wall time 8,000.790s;
supervisor wall 8,004.738s (exit 0, zero pauses), peak aggregate RSS 3.385 GB,
observed maximum 3 total OS threads, minimum free disk 623.832 GB and peak attempt scratch
27.382 MB. The audit records 600 updates, zero protected/refused reads and no CUDA
initialization. No live policy or serving promotion occurred.

Independent receipt checks verified actual family assignments, all 600 minibatch
boundaries, 15 selection locks, 75 snapshots, actual parameter differences and
resource bounds. A separate stdlib implementation rebuilt all primary/secondary
family means from raw cell receipts and matched the reported results. The
12 data-free controls passed on xbabe4; the constructor-only smoke scored 0
contexts and made 0 updates. Scientific source is unchanged since those checks.
Final closeout static checks are recorded with the PR; no GitHub CI was
dispatched under current master_plan §3.

Executed merged source: `fad90b9731be54fa9cad0aea2f83d1baab5823f7`.
The [manifest](manifest.json), folds, protocol and executable remain immutable
registration-time records; the manifest's registration status is historical.
[metrics.json](metrics.json) is a byte-identical copy of the generated result;
[result-receipt.json](result-receipt.json) binds its SHA and verification receipts.
Raw cell scores, adapter snapshots, batch traces and audit stay outside Git at
`/home/ubuntu/jope-data/h148/run/`; commands, resources and review receipts are at
`/home/ubuntu/jope_plan/evidence/H148/`.

Process deviation: central prereg merge #1286 (`db4e8516`) at 02:50:45Z preceded
Argon's exact-author-head PASS 5726174567 at 02:58:12Z. ENG acknowledged a
post-merge review hold at 02:56:30Z. Author/merged source bytes were verified
identical, and the scientific run began only after PASS at 03:03:15Z. Argon's
supplementary merged-blob attestation at 03:17:04Z is retained separately. All
times are 2026-11-01 UTC; review is not described as preceding merge.

## Limits and next step

The initial r882 historically saw these TRAIN families and reused DL-Val;
this is conditional continuation assessment, not fresh unseen-family testing.
The family bootstrap conditions on fitted policies and does not refit/reselect
models; overlapping fit folds are not 15 independent training replicates.
VALSEL's same-panel secondary is selection biased. The inherited roster excludes
known sealed identities/families; a universally complete alias roster was not
available. Rejected checkpoints were not scored on A.

No checkpoint was eligible under either registered rule. That is consistent
with a candidate-generation bottleneck for this fixed 40-update recipe; it does
not establish that selection methods are universally equivalent. Next:
preregister a new candidate-generation recipe while retaining this nested
separation of fitting, selection and assessment and these fixed admission
thresholds, then compare selectors on an eligible candidate set.
