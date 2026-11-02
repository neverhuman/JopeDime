# H222 — Lipschitz smoothing of the adapter

**CLOSED INCOMPLETE.** The full attempt stopped on the load guard during
TRAIN-only tuning. There is no candidate validation result or efficacy verdict.
No scientific retry occurred.

The hypothesis was that constraining the adapter's Lipschitz constant would
reduce context-specific overfitting and improve family-grouped validation
utility. The registered comparison was unchanged r882 continuation (CTRL),
spectral normalization of every trainable adapter linear transform (SN), and
an input-Jacobian penalty (JAC), with lambda chosen from 0.001 and 0.01 using
five-fold TRAIN-family CV. DOPE stayed frozen; the initial checkpoint was an
offline copy. See [PREREG.md](PREREG.md), [manifest.json](manifest.json),
[fold_manifest.json](fold_manifest.json) and [metrics.json](metrics.json).

Seeds were 222101–222103, with fresh TRAIN-family fold seeds 222141–222143.
The protocol required all three lambda choices and nine final models to lock
before candidate DL-Val scoring. It used historical real-S3 TRAIN 1219/205
and DL-Val 91/19 development cohorts. It does not establish new-family,
fresh-native-CV or serving qualification. Utility uses native first-cost-argmin
selection: negative fitness, or -2 on a failed gate. Planned primary deltas
weight the 19 validation families equally. The separate 91-context r882
reference mean, 1.1377941898164525, matched bit-for-bit before training.

## Execution and outcome

[Preregistration PR #1863](https://github.com/neverhuman/JopeDime/pull/1863)
merged as `12da44ef5fb97d74084d652109690c49f1ace854` before model construction.
Ten data-free tests and Ruff, format, compilation, diff and inherited control
sampler/loss checks passed. Source main-guard succeeded; full/hosted CI was
not run and is not claimed passing.

One initial admission refused before a worker or checkpoint copy was created.
The subsequent single TRAIN smoke passed all three arms, with six optimizer
updates, changed parameters and forward outputs, finite JAC second derivatives,
15 SN transforms with exactly two power iterations each, unchanged DOPE and
no CUDA initialization.

The single full attempt stopped at 14:36:45Z on load5 > 60. The supervised
worker's actual exit was -15; it wrote INCOMPLETE with a load/disk guard error
just before termination. It completed 22 of 39 registered fits: all 22 were
TRAIN-CV JAC fits. Those fits contain 880 updates, with one further completed
update in the unfinished 23rd fit. All 881 update intents and completions pair
exactly, without unmatched intents. Including smoke, 887 optimizer updates
are known. The 22 completed state-file hashes and their gradient, parameter,
forward-output and DOPE-before/after proofs were verified.

There is no selection lock, no final model panel and zero candidate validation
readouts. Primary gains, bootstrap intervals, per-seed signs, harm rates and
Elo remain null. The unfinished fit's final tensors and DOPE-after observation,
and the full terminal input audit, are unknown. Completed-fit checks do not
supply those missing observations. This is INCOMPLETE, with no CANDIDATE or
NO_ADVANCE efficacy judgment.

## Load ruling A1

The 14:00Z ruling requested pause above load5 = 60, recheck every 60 seconds,
and resume at load5 <= 55. Its explicit safe-change exception was recorded at
14:00:28Z before candidate outcomes. Both active immutable worker and supervisor
already held in-memory abort guards, with no reload or cooperative handoff.
Pausing only the worker would leave the supervisor abort active. Replacing
an in-flight fit could not preserve optimizer/RNG state. The active attempt
was therefore left unchanged under that exception: its actual policy remained
ABORT above 60. No worker/supervisor was signalled for the amendment, and no
source or controller was injected. Requested and actual policies are retained
separately in the manifest.

The new metadata-only custody unit safely adopted the requested pause/resume
policy. RSS, MemAvailable, free-disk and deadline guards remain hard stops.
Original CPU-only xbabe2, nice 19, four-core, 4 GiB RSS and 300 GB free-disk
limits remain recorded. Compute stopped no later than 17:00Z; closeout was due
17:30Z on 2026-11-02, without a deadline or budget reset.

Raw roots remain `/home/ubuntu/jope-data/h222/attempt-001` and
`/home/ubuntu/jope-data/h222/smoke-12da44ef5fb9`. Source, smoke, resource,
optimizer, state-hash and partial-reconciliation receipts remain under
`/home/ubuntu/jope_plan/evidence/H222/res08-t1300/`. No sealed, holdout,
meta-test, live-loop, GPU, xbabe0 or foreign process/source action occurred.

## Next step

Retain the partial evidence and review the recorded resource boundary before
any separately authorized execution. This closeout authorizes no retry.


## Versioned output custody

The output/source capsule and terminal manifest were uploaded create-only and
downloaded at their exact returned VersionIds. All 160
archive member sizes and SHA-256s matched. The capsule preserves the partial
execution; it does not convert INCOMPLETE into an efficacy result.

Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H222/res08-t1300/terminal-12da44ef5fb9-810a474cfc52d17a/H222-terminal-manifest.json`.
VersionId: `WZDnKYyACarLJ3JPNZ_vHG6AvDkkypQY`.
SHA-256: `27cd34d4e9c995c23ab02d308f3a84570868d666543a62655b0fe60f2174c660`.
The archive's exact version and hash are also bound in [manifest.json](manifest.json).
Original raw roots remain retained. External initial model/dataset payloads and
library temporaries are excluded from this output capsule and remain local.
The new metadata-only custody unit safely used the A1 pause/resume policy;
the already-running scientific worker used the recorded safe-change exception.
The unfinished fit's final tensors/DOPE observation and full terminal input audit
are unknown. The partial ledger records 887 known updates including smoke,
with no unmatched intents and no candidate validation outcome.
