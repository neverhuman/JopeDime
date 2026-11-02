# H215 — ensemble-disagreement pessimism

**PREREGISTERED, NOT RUN.** Test whether adding fresh-ensemble disagreement to
native r882 costs reduces harmful picks. PESS_NEST is the sole primary;
PESS_FIX, shuffled disagreement and optimistic disagreement are diagnostics.

[PREREG.md](PREREG.md) fixes ten fresh adapter members, 40 updates each,
three inner FIT-family folds, three outer seeds and five family folds.
Nested selection plus refits totals 600 member fits / 24,000 updates within
one 90-minute xbabe2 GPU0 attempt. No fallback or second attempt is allowed.
Native91 bit-exact parity precedes candidate forwards. The exact H178-calibrated
four-arm B selector and all fitted members lock before A evaluation.

Primary acceptance requires calibrated B admission, positive median A gain
and lower confidence bound, Holm4 p<.05 and harm<=10%. Elo anchors r882 at 1500;
no promotion, speed claim or U4 measurement. Source, input and fold identities
are pinned in [manifest.json](manifest.json), [provenance.json](provenance.json)
and [fold_manifest.json](fold_manifest.json).

Authority: hw_t0230_astra.md. Preregistration deadline 10:15Z; finish 12:00Z
on 2026-11-02. Local tests/Ruff/records and exact-head review precede launch.
