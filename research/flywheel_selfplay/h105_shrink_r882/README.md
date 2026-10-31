# H105 — r882 shrinkage readiness and guarded queue

The fixed [preregistration](PREREG.md) and numerical worker first merged in
PR #1017 (`0471b241bf800a7ea4fc5b5e85f278c4767f6c75`). The later owner directive
`unpark_t0855_sol_v1_dsc08.md` authorizes real DL-Train CPU validation and queueing
behind H99/H100. It supersedes the earlier preparation-only STOP; it does not
change the original alpha grid, seeds 42–44, fold 105042, anchor strength, 40 updates,
Elo/family-bootstrap/Holm rules, strict .001 margin, <=10% harm, or 2700 s GPU cap.
No H105 timing, outcome, Elo or admission has been measured yet.

## Real training-data CPU validation

[CPU_VALIDATION.json](CPU_VALIDATION.json) records the source-bound forward/backward
smoke at commit `e649c53b3989383f560508c9577ebb30a3121e8d`. The fixed metadata partition has 1006 training
contexts/164 families and 213 assessment contexts/41 families. Sixteen training
payloads were opened; the H97 recipe's first 16-slot batch (sampling with replacement)
produced finite loss 1.806297302 and gradient norm 0.427285017.
Exactly 29,211 non-anchor adapter parameters were trainable. The test exercised
linear interpolation and raw L2 anchoring on gradient-direction copies only;
no optimizer was constructed and no parameter was updated. The adapter digest
and frozen DOPE composite `c51a56d0...` were unchanged. Assessment/DL-Val/outer/
protected payload opens, CUDA initialization and FIFO tickets were all 0.
Elo is null and admission is unassessed; this is a CPU smoke, not a wave outcome.

The CPU command is `cpu_validate.py --source-commit FULL_SHA
--shared-source-root ROOT --out UNIQUE_RECEIPT`. It checks pinned H50 identities,
r882 COPY, selected train-only JTT1/pool bytes, an input-open allowlist, gradients,
source identity before/after, interpolation and the fixed anchor penalty.
Checkpoint copies and all receipts are retained; no shared cache is cleaned.

## Read-only source libraries without another lane

The host already has more than 8 managed lanes with none safely removable. This
change creates no checkout and edits no other lane. Execution binds the actual
five owned code/protocol files and all 13 shared library sources to an explicit
Git commit. The shared files also retain the original SHA256 pins. A different
physical repository HEAD is recorded separately and is never represented as the
executed commit. Byte mismatches fail before payload access and are rechecked at
cell completion; irrelevant HEAD moves do not become mixed numerical source.
Python bytecode writes are disabled. The read-only shared source root used by the
CPU smoke is `/home/ubuntu/JopeDime/.agent/worktrees/agy_h61`; its branch/index/files remain untouched.
No extra code archive is used as a working tree.

After this readiness PR merges, launch from the verified merged bytes:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 python3 research/flywheel_selfplay/h105_shrink_r882/run_wave.py \
  --execute --cpu-pr READY_PR --source-commit MERGE_SHA \
  --shared-source-root READ_ONLY_ROOT
```

The dependency check requires PR #832, original H105 PR #1017 and READY_PR to be
merged ancestors of MERGE_SHA, which must itself be on origin/main. The source
bytes must exactly match that commit; unmerged source cannot create a ticket.
The canonical checkout HEAD/index and foreign source directories are preserved.

## H99 native ledger and FIFO

H99 publishes `jope.h99.gpu-budget/v1`, without the `wave`/`command` keys used by
H100. The old H105 gate missed valid H99 GPU-start evidence while H99 held the
lock and its ticket was absent. Accept the native completed-cell receipt only
with the expected schema/cap, positive finite charged wall, GPU memory/preflight
identity and a hashed output under H99's outcome root. CPU plans, empty and failed
records remain insufficient. No full-wave closeout is required. Existing older
live tickets and H100's recorded-start path remain supported.

The two new regressions failed before the readiness fix and passed afterward;
all 41 synthetic CPU controls passed. No GPU was accessed by preparation.
The real guarded service still needs an inherited qualification.lock lease for
any CUDA work. It joins the FIFO tail behind H99/H100 and all older tickets.
All previous disk/load/memory/RSS, unique scratch, retry/scoring-boundary,
non-resettable 2700 s wall and cumulative wait guards remain. Failed guards wait;
no other job or checkpoint/dataset/receipt is deleted or paused. Current queue
state and the 2700s ledger live outside Git in master/status/H105.md and
`~/jope-data/h105/h105-wave/` after admission.
