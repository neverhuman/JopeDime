# H76: infrastructure stop; optimizer hypothesis unresolved

H76 completed **2 of 12 outcome cells** before the unchanged host-load guard
stopped SAM/seed12. Nine cells never started. **Median gain, confidence interval,
Elo and three-seed train/val gaps are unassessed**, recorded as null in
[metrics.json](metrics.json). No arm is admitted. This is an incomplete
infrastructure result, with no evidence for or against the optimizer hypothesis.

The hypothesis was that sharper minima or too few adapter updates explain part
of r882's plateau. [PREREG.md](PREREG.md) fixes paired seeds 12–14 and CONTROL
(k2 AdamW), K8 (k8 AdamW), SAM (k2, rho .05) and ASAM (k2, rho .5), each for
20 rounds. All use batch 16, constant learning rate 1e-4 and weight decay 1e-4, the H59 CONTROL loss, the
1,219-context real-S3 DL-Train roster and terminal unrevealed-only evaluation on
91 dev contexts. These explicit H76 settings differ from historical H59; fresh
CONTROL is required. r882 and dev have prior exposure, so this is a conditional
exploratory comparison. The frozen checkpoint is copied, DOPE stays c51a56d0,
and only 29,211 non-anchor adapter parameters can train.

The primary requires three same-seed paired val_reg gains, their median and a
10,000-draw family-cluster bootstrap. Admission requires LCB95 strictly above
0.001546 and median >=0. Paired family win rate maps to Elo with fresh CONTROL
as reference 1500; the incomplete panel supplies no H76 Elo estimate. Individual
completed-cell scores were not used to select a setting or justify a retry.

## Execution record

CPU prereg/runner/tests PR [842](https://github.com/neverhuman/JopeDime/pull/842)
merged first; FIFO helper PR [832](https://github.com/neverhuman/JopeDime/pull/832)
was already merged before the detached waiter started.

| Attempt | Result | Charged cell wall |
|---|---|---:|
| First K8/seed12 timing | Infrastructure void: load 89.17/102 | 65.85s |
| Owner-authorized timing retry | Complete; feasibility passed | 84.14s |
| CONTROL/seed12 | Complete | 66.12s |
| K8/seed12 | Complete | 69.12s |
| SAM/seed12 | Infrastructure void: load 88.11/102 | 14.40s |
| ASAM/seed12; all seed13/14 cells | Unstarted | unspent |

[ADDENDUM-cpu-guard-t2335.md](ADDENDUM-cpu-guard-t2335.md) records the explicit
owner exception permitting one timing retry. Repair PR
[855](https://github.com/neverhuman/JopeDime/pull/855) added two load checks
<=86.7, 30 seconds apart, outside the GPU lock before every FIFO ticket/cell,
with up to 30 minutes backoff. The frozen preregistration and in-cell guards
remained unchanged. Original receipts, the void charge and lock wait survived;
retry files have distinct names. No outcome-cell retry is permitted.

The retry projected **1,433.11s** including the original void, below the 2,700s
cap. Actual charged cell wall across all five attempts is **299.630056s**;
remaining budget is not permission to replace a void outcome. SAM's launch load
was 72.35, then rose above the guard 14.40s later. Both coordinators and their
children exited and released qualification.lock. Other people's jobs were not
paused or killed. Maximum recorded RSS among successful children was 4,921,456
KiB (about 4.69 GiB); no RSS, memory or disk guard failed.

The two complete cells each changed adapter weights, verified the 29,211-parameter
inventory and c51 checksum before/after, and scored all 1,219 train and 91 dev
contexts. The void SAM cell has no final receipt; its completed update count and
post-cell integrity proof remain unknown. Missing cells are not scored as zero.
No sealed/holdout payloads, live learner, serving promotion or xbabe0/xbabe3 work
was involved.

## Context and next step

The owner's t2335 context is H61's
[closeout PR854](https://github.com/neverhuman/JopeDime/pull/854): 0/240 accepts
under its selection rules, pointing toward candidate generation. The [H59 index entry](../../../docs/research/studies/dope-neural-ranking-20261021/overnight-hwave-index.md) reports CLIP as least harmful (median Elo 1487.4
versus CONTROL 1479.4, both below r882). Those H59 Elo values use an r882
reference and are not H76's fresh-CONTROL comparison. This context changed no
H76 arms or analysis choices.

The optimizer hypothesis remains unresolved. A continuation needs a new
prospective owner protocol and a stable CPU window; this protocol authorizes
no replacement of the void outcome. Retain the completed cells as partial
receipts, without an arm ranking, primary estimate or admission claim.

All outcome cells used source `7c0dd66f58bff7eef718cef629afc6e9d8e17d8a`.
Raw receipts, adapter snapshots, the immutable first attempt, budget ledger and
status logs are under `~/jope-data/h76/h76-wave/`. The compact metrics generator
and input/output hashes are retained in
`~/jope_plan/evidence/H76/t2300/build_incomplete_metrics.py`; it inspects receipt
integrity and roster coverage without consuming utility/loss values. CPU gates
passed Ruff, py_compile and 37 unit tests. Publication adds JSON/link/index-row
and whitespace checks; no CI was dispatched.
