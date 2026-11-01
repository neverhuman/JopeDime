# H187: robust loss and equal-family fresh adapters

**NO_ADVANCE** in owner-authorized attempt-002 after 7,200 adapter updates. No model or live selector adoption.

Attempt-001 was INCOMPLETE after a transient host-load stop: 376 optimizer calls completed / 377 started (last completion unknown), 2 B cells, 0 candidate A/Val cells. Its artifacts are preserved and excluded from attempt-002. The owner authorized a fresh scratch run with unchanged seeds/arms/prereg; only hosting/resource guards changed ([relaunch](RELAUNCH.md)). Attempt-002 ran on xbabe1 with 80 GB free/output <1 GB limits; the xbabe2 fallback uses 3 consecutive high-load samples. Candidate outcomes from attempt-001 were not inspected before relaunch.

## Hypothesis and setup

Difficult contexts and large families might dominate the live loss. This study pairs H180 fresh-init control with RNORM (Huber2 of per-context live loss divided by detached anchor-only loss, floor .001) and FBAL (context probability proportional to inverse family size, with replacement). All arms share native initial weights; control/RNORM also share minibatches. Only the 29,211-parameter adapter trains.

Three seeds 187101–187103, five fresh cyclic folds 187140: FIT 123 / B 41 / A 41 families, actual final step **160** for every arm. Timing-only 80-step pilot is a retained prefix of the first control trajectory; fallback=False. All 45 trajectories completed. The imported H177 three-treatment Holm branch selects once on pooled B (205 families), then a durably fsynced, hash-bound lock precedes any candidate A/Val scoring. Native slot metadata 40 is recorded separately from the actual 80/160 budget. [Preregistration](PREREG.md) defines every treatment and gate.

Pooled A/B reuse the same 205 families in different roles and are correlated, not independent new-family replications. H178 validates the inherited four-arm mechanism; the owner-prescribed m3 branch was not separately empirically recalibrated. RNORM bounds influence in normalized-loss units, not the full parameter gradient. FBAL allows repeated contexts to make per-draw family probability exactly equal.

## Outcome

All gains/utility/Elo below are medians of three seeds with 10,000-draw paired family 95% intervals (seed reduction axis 0). B admission uses the imported seed-averaged-within-family functional and Holm m3; its mean/CI/p are reported separately. Ties reject.

| Arm | A gain vs R882, median [CI95] | Positive A seeds | B mean gain [CI95] | B Holm p | B admitted | ADVANCE |
|---|---:|---:|---:|---:|---|---|
| FRESH160 | +0.006986 [-0.022639, +0.039399] | 3/3 | +0.004899 [-0.027727, +0.039447] | 1.000000 | False | False |
| FRESH160_RNORM | +0.005898 [-0.023730, +0.038965] | 3/3 | +0.006669 [-0.023641, +0.039991] | 1.000000 | False | False |
| FRESH160_FBAL | +0.006033 [-0.026180, +0.041160] | 2/3 | +0.001696 [-0.031391, +0.036247] | 1.000000 | False | False |

ADVANCE requires B admission at gain > .001 + 1e-12, LCB95 > 0, Holm p < .05, followed by A median gain > 0, 3/3 positive seeds and A median-bootstrap LCB95 > 0. Candidate gains are shown even when B falls back to R882.

| Arm | A equal-family utility | A Elo (R882 = 1500) | A exact-2 count per seed (out of 1,219) |
|---|---:|---:|---|
| R882 | +1.385355 [+1.145946, +1.676323] | +1500.000000 [+1500.000000, +1500.000000] | [17, 17, 17] |
| FRESH160 | +1.392341 [+1.151463, +1.685712] | +1508.966026 [+1493.413290, +1524.465002] | [20, 22, 16] |
| FRESH160_RNORM | +1.391254 [+1.149831, +1.686005] | +1504.292775 [+1490.164365, +1521.619966] | [23, 21, 17] |
| FRESH160_FBAL | +1.391388 [+1.152437, +1.684202] | +1511.168933 [+1495.894331, +1524.791967] | [19, 20, 15] |

### Paired diagnostic contrasts on A

These contrasts do not change admission:

| Contrast | Median gain [CI95] | Per-seed gains |
|---|---:|---|
| FRESH160_RNORM minus FRESH160 | +0.000411 [-0.008865, +0.009401] | +0.000411, +0.005450, -0.010736 |
| FRESH160_FBAL minus FRESH160 | -0.000953 [-0.012288, +0.008452] | -0.000953, -0.004184, +0.002668 |

### Descriptive DL-Val

The 91 unique DL-Val contexts are never fitted or used for selection. The native qualification baseline uses a context mean (1.1377941898164525); these tables weight families equally, giving R882 utility 1.153615. Five fold models per seed give 455 model/context pairs; utility/Elo average over a uniformly random fold model, NOT a score-averaged ensemble. Exact-2 counts below are out of 455 pairs. Elo uses context-pairwise wins/ties before equal-family weighting (19 families), MM 400 iterations/prior .01.

| Arm | Equal-family utility | Elo | Exact-2 counts per seed /455 |
|---|---:|---:|---|
| R882 | +1.153615 [+0.740165, +1.630115] | +1500.000000 [+1500.000000, +1500.000000] | [0, 0, 0] |
| FRESH160 | +1.118919 [+0.702467, +1.594491] | +1518.101254 [+1476.588327, +1566.678214] | [8, 6, 5] |
| FRESH160_RNORM | +1.123732 [+0.709274, +1.596285] | +1520.477841 [+1482.775842, +1561.464312] | [10, 5, 5] |
| FRESH160_FBAL | +1.114043 [+0.697594, +1.593758] | +1516.273401 [+1482.449419, +1561.566149] | [8, 7, 5] |

## Verification and custody

Fresh native91 r882 baseline PASS before TRAIN decoding/updates: utility and complete candidate-cost bytes exact, mean 1.1377941898164525. DOPE c51a56d0 and all non-adapter tensors (8d1070fa...) frozen throughout. Actual optimizer/backward started/completed counts all equal 7,200. Fifteen native initial states pair across arms; 45 trained checkpoints differ from initialization and are retained. R882 is restored after the run for source-state closure; this does not mean the adapter was unchanged during training.

14 data-free tests, Ruff and all 5 jope-records gates PASS. Independent recount verifies 7,200 FIT minibatches/slates, 26,037 raw candidate choices against real pool labels, 15 initial/45 trained state hashes, complete B/A/Val panel reconstruction, all utility/contrast CIs/Holm and 10,000-draw Elo, B locks and ADVANCE gates. Separate native inference checks 9 trained adapters, 216 context predictions/248,832 candidate costs bit-exact against saved/cached costs, with zero optimizer/backward calls. This is targeted local/native research proof; full hosted/workspace CI is not claimed.

Independent rd_argon prereg review **PASS**, review 5728731368 at 69852b94aec9e624e4a6072ccf35226a26679eff, submitted 2026-11-01T15:02:20Z. The original preregistration, design/statistics and objective files are byte-identical to that review; hosting/resource guards were separately reviewed in #1582. Earlier pending readbacks were corrected once the review was retrieved; both records are retained. Exact-head author review and standing owner H-wave targeted-check/admin merge exception are recorded without claiming full hosted CI.

Resource guard PASS: 3353 samples, peak RSS 3.390 GB, max threads 3, max CPU 16.3%, min available 31.79 GiB, min free 107.6 GB (floor 80 GB on xbabe1), max output 164.6 MB (<1 GB). Nice 19 / math 1, CPU xbabe1, no GPU/protected/live writes.

[Compact metrics](metrics.json): 34,368 B (<100 KB), including per-seed gains, B/A/Val summaries and diagnostic contrasts. [S3 custody](custody.json) retains raw traces/states/costs/panels plus all 1,310 source pools, source/assets and audits: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H187/codex-t0730/e1259695e732c5cf/H187-evidence.zip` VersionId `QUew2tC7_HoHCj5MwIjvvA_9mjFu_0wb`. Actual exact-version restore verified all 1,804 member payload hashes, decoded 40,867 arrays and all 60 initial/trained adapter states. No large objects enter Git.

## Verdict and next step

NO_ADVANCE under the preregistered conjunction. Keep the live model/selector unchanged. Further training-signal work needs a separate preregistration; these outcomes do not establish equivalence or a representation ceiling.
