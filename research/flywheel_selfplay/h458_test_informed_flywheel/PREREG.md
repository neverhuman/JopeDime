# H458 preregistration — Track D (evaluation-matched ICL specialists on dev91 family pools)

Wave: `h458_test_informed_flywheel` · Owner: Jepson · Executor: `cl_h458` (Claude, session 579ede16, works on xbabe2
over ssh from xbabe3) · Hosts: xbabe2 (GPU pilot, CPU probes), later xbabe1 by agreement · Source commit: the head of
`feat/h458-test-informed-flywheel` recorded in `launch.json` · PR: opened before scored inference, left unmerged ·
Claim: `~/jope_plan/ns/receipts/H458_claim.txt` · Coordination: `~/JopeDime/shared_CHAT.md`.

Written before any H458 checkpoint exists and before any audit label has been read for H458.

## Disclosure

Every number in this wave is **in-corpus**: `in_corpus_dev91_disclosure = true`,
`canonical_promotion_forbidden = true`. The family pools contain each context's SEARCH rows and AUDIT rows, and 60.4%
of all audit rows also sit in a sibling context's SEARCH rows. A specialist's utility on those contexts is therefore
mostly recall of labels it was trained on (closure class "recall"). It is reported beside the fair anchors
(`h448.APEX_FAIR_CHAMPION`, `h453.FAIR_OOS_0PCT_D91_NO_UGP`) and beside the leave-context-out "skill" arm below.

Owner rule (2026-11-14): training may use test labels; inference reads only the context's SEARCH rows (plus audit
features) and selects by CV on them. No sealed cohort (CONFIRM / HOLDOUT / SEALED / DL-VAL / FMGAP B,C) is opened.
The benchmark's exact search/audit partition is never used as a training episode (selftest gate
`benchmark_split_audit`).

## Hypothesis

The ICL tower is weak on dev91 because its training does not resemble evaluation (context size, duplicated rows,
synthetic columns, foreign targets, untrained rank-Gauss and CW/struct inputs, ~2% dev91 share, learning rate at its
floor). Fine-tuning an existing 23.7M checkpoint on one family's pool with episodes built by the evaluation's own
feature and target pipeline makes its columns beat the field maximum on that family's contexts.

## Inputs (immutable)

- Family pools: `/home/ubuntu/jope-data/h458-quarantine/family_pool_v1`, data sha256
  `33f29cac05f3e0416a588df7093b238b6a95ff763aa40ea84c50a2de976c527d` (19 families, 75,553 unique rows), built by
  `family_pool_h458.py` from `mmap_v1` cohort `D91-COMBINED-SHUFFLED` only. No cousin tables, no out-of-envelope S3
  rows (owner ruling "dev91 probe", session 7afc9e1c).
- Init: `fleet_M_x2_h455_maxent/ckpt_valley_002.pt`, key `ema` (valley 003 is compared by the probe before training).
- Evaluation code: the modules listed with their sha256 in `deps_h458.json` of the staged runtime copy.

## Treatment (fixed)

`train_h458.py`, one run per family: fresh stratified downsample and fresh context/query split per micro-batch,
unique rows; context size 70% evaluation-size (all, or 6/7, of 0.7·N of a member table), 20% at 384–768, 10% at
256–384; up to 1,024 query rows; the three direct target spaces (robust-ridge residual, Yeo-Johnson, rank-Gauss) from
one feature build; AdamW(0.9, 0.98), weight decay 0, 300-step warmup, hold to 60% of the exposure budget, cosine to
5e-6, clip 1.0, EMA 0.999; peak LR 5e-5; budget 2,000 query exposures per unique pool row with checkpoints at 250,
500, 1,000, 1,500 and 2,000.

An exposure is one query row in one forward pass, so a row scored in all three target spaces of a micro-batch counts
three: 2,000 exposures per row are about 667 three-mode appearances. Exposures and split commitments are booked only
when their gradients reach an optimizer step. A step with a non-finite gradient norm is dropped.

The GP-residual space is not trained in this cycle. The H458 block that enters the stack therefore carries the three
direct columns only; it does not emit `icl_e8_gp_resid_cv7` from a specialist.

Cross-fitting strata (skill arm, shared with Tracks S and G): `k = blake2b(canon(X_row), digest_size=8,
key=b"h458-k5-strata-v1") mod 5` on canonical feature bytes (float64 little-endian, -0.0 to +0.0, one NaN pattern,
no target).

Pilot families: 60b608db, 90a58676, f2893c38, 469b72b7. Pilot arms:

| Arm | Change from the treatment |
|---|---|
| `rand` | none (random feature-ID slots, the evaluation path as it is) |
| `canon` | canonical feature-ID slots in training and in the probe |
| `lr1e-5`, `lr1e-4` (60b608db only) | peak LR |
| `tiny` (60b608db only) | 48–100 context rows for the first 30% of the budget, then the treatment mix |
| `skill` | cross-fitted: five runs per family, each without one feature-hash stratum, warm-started from the fair h449 M v2; a row is scored by the run that never trained on its stratum |

## Measurement and gates

Probe (`probe_h458.py`): out-of-fold utility on the benchmark contexts' SEARCH rows, repeat 0 of the evaluation's
CV-7, per target mode and for the three-mode mean. It reads no audit label. Baseline = the init checkpoint.

| Gate | Pass |
|---|---|
| D-1, after 1,000 exposures | probe gain over init ≥ +0.15 on 60b608db and 90a58676 and ≥ +0.05 on f2893c38 |
| D-2 | probe ≥ 0.68 (60b608db), ≥ 1.10 (90a58676), ≥ 1.17 (f2893c38); 469b72b7 gains ≥ +0.5 on contexts whose SEARCH rows hold a target beyond 8 SD |
| stop rule | below +0.03 everywhere after 3,000 exposures at 5e-5 and 1e-4: Track D1 stops |

Official looks at audit labels: the campaign has four in total (shared ledger); Track D uses two — a 20-context pilot
look and one full-ladder look. Arms are named in the ledger before each look. Primary arm: legacy pool + the H458
recall block (tag `e8_cv7_hyb`, two checkpoints per family). Insurance arm: the H458 block alone. Skill arm: the same
with leave-context-out specialists. Statistics: the ladder's own Elo, family W-T-L against every entrant, and the
H335 sign-flip procedure.

## Resources

xbabe2 RTX 3090: at most 7 GiB, launched only with ≥ 9 GiB free; `MemoryMax` and `choom` on every process; cpus
64–79 untouched; host CPU ≤ 85%. Numeric threads: 1 per process; up to 16 single-thread data workers (owner-approved
departure from the template's total of 4). No job of another agent is stopped except the stuck
`fleet_M_x2_h455_maxent` (owner decision), announced in the shared chat first.

## Retention

Kept as scored-inference inputs that cannot be regenerated bit-exactly (bf16 on GPU): per family, the two
checkpoints that enter the stack, with sha256 in the evidence manifest. Pilot milestone checkpoints and probe caches
are listed in a retention receipt and removed only at wave close with the owner's go-ahead. Run configs, logs, probe
JSONs and per-context results are the durable record.

## Amendments (dated; everything above this line is the text frozen before the first checkpoint)

- 2026-11-14 22:10Z, after the first 60b608db probes and a reviewer challenge (`cl_fwriter`, shared chat): canonical
  slots are the scored path of the **recall** arm only. Whether they help on unseen rows is decided by the
  cross-fitted skill arm in both slot modes, not by the recall curve.
- 2026-11-14 22:50Z, skill arm on 60b608db (four of five strata read): random slots +0.01 to +0.02 u over the fair
  init at 1,000 exposures, fading by 2,000; canonical slots -0.09 to -0.15 u at 2,000. The skill arm is therefore
  not trained for the other families at this recipe. This is a departure from the pilot table above (which listed
  the skill arm per family) and is reported as such.
- 2026-11-14 22:50Z, looks: the joint final look reports two pre-named numbers from one ledger look, the full stack
  and the same stack without the recall columns (agreed with Track S). Recall columns are exempt from the peer
  guard by sidecar closure (rule S6), so one checkpoint per family enters the stack; the 1,500-exposure milestone
  of the same run may be delivered as a second block.
- 2026-11-14 22:50Z, operations: an out-of-memory error now trims the item (fewer query rows, then one target
  mode) and retries it; a queued item built under older caps is re-sized on consumption; only a failed backward
  pass drops the open gradient group. Data workers may run on another host (`--remote-workers`), which holds a
  verified copy of the staged code, the dependency stages and the family pool; no checkpoint, bank or audit label
  leaves the training host. The selftest pins numeric threads to one, because threaded BLAS changes the last bits
  of the ridge-base product with the shape of the query block.
