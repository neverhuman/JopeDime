# Q1 preregistration — draft for OWNER signature

Status: **OWNER-DELEGATED KEY INSTALLED / NO PLAN SIGNED / NOT AUTHORIZED FOR
QUALIFICATION**. This document proposes OD7; neither this commit nor a passing smoke
test is an owner signature over a plan or a Q1 pass.
Owner allowed-signers SHA-256: 6bc05cef59ee45376fe77fcaf27074edac2f7bd390faac3f193c466175313235

## Question and decision

Does the frozen regression bundle improve over heuristic guidance at equal rung-0
admission budgets? Q1 determines the regression `auto` default; it does not block
release. No binary or multiclass training, blended headline, or promotion is included.
The implementing agent must never sign on behalf of OWNER, except through the
owner-delegated Q1 key recorded below.

## Cohort and data boundary

The **sealed meta-test group is the only qualification holdout**, per
`lanes_0645.md`. Use at least 40 distinct authoritative regression families, exactly
one dataset per family. QLT-02 supplies and freezes a metadata-only cohort manifest
with dataset SHA-256, authoritative family key, row/feature counts, cut SHA-256 and
`holdout_group: sealed_meta_test`. Verify zero intersection with the frozen training
family closure from RES-01. Reject unknown/derived family keys, missing identities,
DL-holdout, training-validation replacements, mixed tasks, or fewer than 40 families.
Choose a dataset within a family by the smallest eligible dataset SHA-256, before
any outcome is observed. Freeze the complete list and its hash, not just these rules.
No optional stopping, post-outcome replacement, or cohort/subgroup retuning.

The authoring worker never opens sealed/DL-holdout payloads. The harness exchanges
metadata and aggregate receipts with a separately pinned, owner-authorized evaluator;
it does not resolve qualification dataset paths or read their rows. Only QLT-04 may
execute the later authorized qualification. Existing split constraints, FIT-only
preprocessing/calibration, and HOLDOUT/TEST isolation remain required by spec §5.4.
Do not use exposed historical test tracks as a replacement qualification holdout.

## Frozen execution identity and budgets

Freeze source commit, backend binary SHA-256, candidate bundle/DOPE SHA-256,
evaluator/vocabulary/protocol identities, objective `rmse`, all dataset/cut identities,
CPU device, thread count, limits and cohort before requesting approval. Both policies
use the identical dataset, folds, objective, budget, engine and seed; only guidance
changes. Seeds: **0, 1, 2**. Each policy gets **24 charged rung-0 admissions** per
(dataset, seed), including failed attempts. No free retries, substitution, additional
measured proposals or wall-time-based admission advantage. Freeze later-rung policy
and stopping rules in the backend protocol; require actual, nonzero evaluation counts
and report every attempt and failure. A run receipt must prove the protocol identity;
matching a JSON echo alone is not operational evaluator qualification.

Run both guidance policies for every dataset and seed in a deterministic schedule,
reversing policy order on odd seeds. Use CPU only, at most eight workers/threads and
fixed per-run limits (600 seconds per qualification run). Timeouts/failures consume their recorded budgets; interruption
leaves an incomplete panel and permanently records that the candidate was started.
The owner-signed plan pins the absolute persistent ledger root. The ledger refuses
a second qualification start for that bundle; changing the root requires a new owner signature. Resume or repeat
requires a separate owner-reviewed protocol, never an automatic budget reset.
The sealed cohort is consumed once per candidate bundle and never used for tuning.

## Estimand, normalization and uncertainty

Average the three final RMSE observations for each policy within a dataset first.
The dataset's normalized improvement is `(H - L) / max(H, 1e-12)`, where H and L are
heuristic and learned mean RMSE. Positive favors learned. The fixed floor handles
zero comparator loss; non-finite arithmetic is an invalid result, not zero.
The headline is the **median dataset-level normalized improvement**, never a pooled
row/seed mean. One dataset per family makes the dataset the independent resampling
unit. Draw **10,000 paired dataset-level resamples with replacement**, seed
**20261030**, and report the percentile 95% interval (2.5%, 97.5%, linearly interpolated).
The primary superiority criterion is a strictly positive lower endpoint.

Prespecified overlapping subgroups (membership from frozen metadata only):

- small/large: at most 1,000 rows / more than 1,000 rows;
- narrow/wide: at most 20 features / more than 20 features.

Each subgroup must contain at least five datasets, or qualification is INCOMPLETE.
**Proposed OD7 margin: 0.02 (2% relative normalized-loss regression).** Each subgroup's
median improvement and its Bonferroni one-sided 95% lower bound must be at least
-0.02. Use the same 10,000 paired resamples, subgroup seed derived by SHA-256 of its
name plus the primary seed, and lower quantile `0.05 / 4`. This conservative rule is
a proposal for OWNER approval, not an assertion that OD7 has already been ruled.

Q1 PASS requires a complete valid panel, primary superiority and every subgroup
non-inferiority check. Otherwise report FAIL or INCOMPLETE with the precise cause;
never hide failed/unknown observations or manufacture substitute scores. No pooled
binary results. Report regression validation, effect/interval, subgroup counts,
all failures, actual compute and separately measured Elo/percentiles when available;
never invent Elo from these loss calculations. Preserve receipts and input hashes.

## Owner signature and single-use enforcement

The digest above is the SHA-256 of the OpenSSH allowed-signers file for identity
`jepson`. It was originally to be an owner-supplied file. On 2026-10-31 it was
instead filled in for the owner-delegated key described next, replacing `PENDING`.

**Owner delegation, Q1 only.** Jepson approved on 2026-10-31 at 10:32 MDT that a
bot-held ed25519 key may sign Q1 as principal `jepson`. The committed file
`owner/allowed_signers` (public key only, namespace-restricted to `jopedime-q1`,
fingerprint `SHA256:2KNNiZS7c+YbwsAMt2x0Q9gIdRCd2Yi6CR3k/d1N6LY`, no expiry) is that
key. It is an **owner-delegated agent key, not a key Jepson holds personally**, and
authorizes nothing outside Q1. Its private half is a 0600 file on xbabe2 under the
shared `ubuntu` account, so **any agent process running as that user can produce a
valid Q1 signature as `jepson`**. The owner accepted this risk; a signature still
binds one exact document, plan and backend. To revoke the key, replace
`owner/allowed_signers` and the digest above in a reviewed PR. The canonical
signature path is
`/home/ubuntu/jope_plan/ns/qualification/q1_approval/<plan_digest>.sig`, where
`<plan_digest>` is `q1.digest(plan)`, the SHA-256 of the canonical plan JSON. The
canonical ledger root a signed plan must name is
`/home/ubuntu/jope_plan/ns/qualification/q1_ledger`. This amendment signs no plan:
the signature must still bind the final preregistration, full plan and qualified
backend hash, and every other gate below still applies. The signature is external to this
file. Sign the exact bytes emitted by `q1.py approval-payload` with namespace
`jopedime-q1`; these bind this preregistration, the entire frozen plan and the backend
binary hash. Verification checks the pinned allowed-signers digest and OpenSSH
signature before any qualification subprocess is launched, including its heuristic
arm. A changed document, cohort, plan, backend or signature invalidates authorization.
The harness has no signature-generation or approval-creation command. The actual
signature, cohort closure and selected bundle remain PENDING.

## Development smoke acceptance and limits

Three catalog-selected **non-sealed FIT blobs** are used only to exercise orchestration:
`00d3d3a9335ea29d`, `02683ae98254d34e`, `030629180644e2ca`. Their authoritative metadata
marks `has_meta_test=false`. Pin source hashes; use only train blobs, at most 128
rows, eight features, seeds 0/1 and four charged candidates per run. The reference
adapter calls real Rust `feat_math::evolve::api::evaluate_candidate` with two inner
CV folds and fixed starter order. It is CPU-only, heuristic-only and explicitly
refuses qualification/learned requests. No sealed payload, native test blob, alternate
Q1 holdout, learned model, DOPE weight or model-training campaign is accessed.

These runs demonstrate harness plumbing and actual numerical execution. They do
not qualify the production discovery loop, learned guidance, M3 parity or Q1.
The external engine adapter still needs DSC-08/RNK-10 integration and independent
receipts proving its real evaluation counts and data isolation. QLT-04 owns the
actual study/report/default decision. No owner signature is fabricated to finish
this task; unsigned acceptance remains an explicit outstanding requirement.

## Approved pre-outcome amendment — QLT-04 T0610 (2026-11-01)

ENG approved the R1/R2 amendment before any Q1 outcome. The wide threshold (`features > 20`) and minimum of five remain unchanged. The first catalog-order wide regression record in the pinned H150 `catalog_sealed_ids` set that is absent from the existing 153-member owner roster is `c4b1f6dfb2a8605f` (Allstate Claims Severity). `0380f8df655797ba` is not in that sealed-ID set. The selected candidate's raw source family identifier is `legacy:c4b1f6dfb2a8605f2b71db63d1bc79aa1f064019f1bc689ad64ccdf83bac42a2:openml-42571-233212-loss`; it does not satisfy the Q1 40/64-lowercase-hex family-key contract and no authoritative family alias binding was found. Therefore it is recorded as a pre-outcome sealed-ID candidate, but the official roster remains at four wide families until that identity is bound. No family key is synthesized and the subgroup minimum is not claimed met.

For each candidate FIT object, materialize three deterministic CV folds over all zero-based physical FIT row positions. Use fold seed `0` (the first preregistered evaluation seed), singleton row groups keyed by FIT row index, and the checked-in native `jope-service/src/discover/splits.rs::k_fold` algorithm with label `dsc03/rung0/groups`. The three validation partitions are emitted in FIT row order; each fold's training rows are their ascending complement; all folds have `repeat: 0`. No target, feature, test, holdout, or outcome values participate. The exact cut definitions and canonical `jope.q1-cut/v1` hashes for the existing 153 records plus the candidate are in [`cut_manifests/fdc243193d9436449cf74aa92b05ff7842c6c88bf4dce0b29ef7edf453b74987.json`](cut_manifests/fdc243193d9436449cf74aa92b05ff7842c6c88bf4dce0b29ef7edf453b74987.json) (SHA-256 `fdc243193d9436449cf74aa92b05ff7842c6c88bf4dce0b29ef7edf453b74987`). These are proposed source-bound cuts, not qualified evaluator inputs. The amendment does not authorize signing or a Q1 launch.

### T0640 roster-to-cut packet handoff

The amended 153-row owner roster is now separately projected in [`rosters/ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a.json`](rosters/ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a.json), SHA-256 `ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a`. It binds each roster dataset ID and dataset hash to `cut_sha256`, and each digest resolves to a canonical definition under [`cuts/`](cuts/). All 154 definitions (153 owner-roster entries plus the distinct R1 candidate) are derived from aggregate manifest SHA-256 `fdc243193d9436449cf74aa92b05ff7842c6c88bf4dce0b29ef7edf453b74987`; no dataset payload was read.

This packet does not join the QLT-02 candidate cohort: its 2,489 dataset IDs have zero overlap with the 153 owner-roster IDs. It also records 36 owner-roster rows without an authoritative family key. No IDs or family keys are inferred to bridge either gap. The R1 wide candidate remains outside the official roster, so the wide minimum remains 4/5. The packet gives QLT-05 a concrete roster-to-cut source contract; it does not make the cohort signable or authorize a Q1 run.

## Owner direction T1115 — 2026-11-01, before any Q1 outcome

[PREREG-AMENDMENT-T1115.md](PREREG-AMENDMENT-T1115.md) supersedes the unexecuted
R19 four-wide exception. Select one minimum-eligible-dataset-SHA representative
per frozen owner family (153 candidates -> 97 representatives), preserving the
original packet/cut identities. All subgroups require at least five distinct
families. The selected wide subgroup has three, so qualification is INCOMPLETE.
No override, family invention or duplicate count is authorized. All primary
statistics, thresholds, r782/DOPE pins, seeds, per-policy budgets and single-use
ledger rules above remain binding. No Q1 outcome has been observed.

## R20 pre-outcome wide census amendment

Owner ruling R20 supersedes the T1115 five-wide floor for this fixed census.
[PREREG-AMENDMENT-R20.md](PREREG-AMENDMENT-R20.md) and
[amendments/r20.json](amendments/r20.json) bind all three eligible distinct wide
families; all must appear with exact pinned identities. Other subgroups retain
five; every subgroup retains its existing non-inferiority check, margin and
0.05/4 correction. The 97-family selection, frozen pins/budgets and all runtime,
exposure, signing and single-use requirements remain unchanged.
