# H82 operator continuation after a pre-model staging fault

This is a bounded operational repair. The frozen PREREG and all numerical worker
files stay unchanged at `037f914a59aabd34279000c40288315dc00500d8`.

SAM/seed24 attempt1 aborted on load86.40 before scoring (90.175953s charged).
Attempt2 then failed before loading the model: `shutil.copy2` attempted to
overwrite the read-only `scratch/r882_adapter_copy.pt` left by attempt1. Its
5.621730s remains charged and it remains recorded as a fatal setup error.
No optimizer or scoring operation began in attempt2. The five-attempt prefix
charges330.761267s, with CONTROL/K8 seed24 complete and unmodified.

The continuation consumes only the remaining second retry of the original
load-aborted logical cell: SAM's next/final physical attempt is3. It does not
add a fourth attempt, reclassify the setup error as a load abort, repeat a scored
outcome, change seeds/settings, or reset a budget. It is sealed to this exact
stopped ledger and traceback. All future logical cells retain the original
pre-scoring-load-only retry rule and three-attempt maximum.

Before each attempt, preserve/readback-hash and unlink only the stale staging
copy if its full SHA equals the immutable r882 input. This is input housekeeping;
it never removes learned snapshots, phase files, receipts or the source checkpoint.
Unknown bytes, symlinks, extra outcome/phase/snapshot files, changed receipts,
computed/uncertain outcomes or a reused recovery prefix are refused.

The launcher is reviewed/merged separately and copied byte-for-byte from that
Git blob into external custody for execution. It verifies its own merged bytes,
then loads the unchanged worker from the original pinned checkout. Both worker
and launcher revisions/hashes are recorded in the same ledger. This preserves
the original scientific source identity across all cells. Operator downtime is
conservatively charged to the separate3h wait allowance; the2700s GPU-cell cap,
all prior charges/waits, CPU/disk/RSS guards and per-attempt FIFO remain intact.

A new exclusive coordinator lock protects the continuation. The launcher restores
only the two already-complete receipts by hash and finishes the unstarted original
panel. It invokes the unchanged worker readout only after all12 cells complete.
Original setup diagnostics remain retained separately. No H76 outcome is read.

CPU regression proof reproduces the actual permission failure with a synthetic
read-only checkpoint, proves successful preserved cleanup, and rejects changed,
unknown, scored, integrity-failed or extra-artifact prefixes. No GPU is used by
these tests. This repair is reviewed before any resumed GPU attempt.
