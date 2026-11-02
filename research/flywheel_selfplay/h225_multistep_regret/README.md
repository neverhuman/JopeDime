# H225 — multi-step proposals with continuous-regret selection

Status: **INCOMPLETE / INCOMPLETE / None**.

The hypothesis was that larger adapter proposals escape the r882 top-1 plateau,
with continuous TRAIN regret selecting them more reliably than discrete utility.
The frozen design used MSCR, MSTOP1 and within-family MSCR_SHUF, eight proposals
per seed/outer cell, three seeds, and fresh five-fold family partitions. The
120 outer proposals plus240 shared inner fits have a logical budget of8280 updates, plus
the two-update TRAIN smoke. Selection is locked before candidate A/DL-Val scores.
Full protocol and input pins are in [PREREG.md](PREREG.md) and [manifest.json](manifest.json).
Those prereg source hashes refer to immutable execution source `bd028747af73f0e72f66c9ddb64b1c486a382b13`.

No complete A/DL-Val cohort was measured; gain, CI, harm and Elo are unknown.

Failure: The sole admitted CPU continuation was interrupted by the host reboot initiated19:27Z (final shutdownSIGTERM19:29:09Z). Both controller and worker are absent; only120/360 unique completed-fit proofs exist. Preregistered ONE continuation is consumed; no scientific restart. Original scheduled19:00 stop/admission facts and all costs remain preserved.. Ledger and fit counts are in metrics.json; no partial result was promoted.

Raw artifacts, state hashes, optimizer ledger, frozen-DOPE proofs and exact input
audit are preserved in the version-replayed S3 objects in [metrics.json](metrics.json).
S3 custody: REMOTE_VERIFIED. The local run root remains `~/jope-data/h225`.
Reference r882 Elo is1500. Missing observations remain unknown; no protected,
holdout or live-run qualification is inferred. The fixed compute deadline was
22:30Z; actual terminal timestamps remain in raw receipts. Closeout authority is
the original H225 instruction.

Next step: Retain r882. Review the recorded failure and obtain a new prospective attempt before any scientific retry.
