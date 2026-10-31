# H96 GPU queue entry — `r882_top4` continuity arm

**Status: queued, not started. No GPU taken and no lock ticket held.**

## Position

Under `~/jope_plan/qualification.lock`, this entry comes **after H57 → Fused D → W1**, the owner's lock order (t2320,
t0300). It must not start before W1 has released the lock. It is an entry in this record, not a live FIFO ticket. A
waiting ticket process would take the GPU on its turn, which is the queue owner's decision. When W1 is done, the queue
owner, or rd_opus on the owner's explicit go, starts the launcher. The launcher enqueues through the merged
`research/flywheel_selfplay/qualification_lock_fifo.py` with `wave="H96"` and waits its turn like every other wave.

## What runs on the GPU

- **Inputs, all produced by the CPU `run` stage first:**
  - the 16 new proposals per roster context-seed (`rows_w4.jsonl` plus the proposal DSLs in `fits.jsonl`)
  - the four hash-ordered pool anchors with their admitted utilities
- **Model:** the r882 checkpoint copy (sha256 `0084479a…`, the copy used by H82/H93), with DOPE `c51a56d0` frozen and
  hash-checked before and after the run. Inference only: no parameter update.
- **Output:** for each context-seed, r882's top 4 of the 16 proposals. Fits reuse the CPU `FitStore`: proposals
  already fitted by a CPU arm cost nothing, and new ones are fitted on CPU under the same 1,800-fit ledger.
- **Estimated GPU wall:** under 15 minutes, for 150 context-seeds × 16 proposals of inference. Per H93's host
  guards, cap at 900 s cell wall, RSS < 6 GiB and MemAvailable ≥ 16 GiB, with the disk guard at < 94.5%.
- **Reporting:** `r882_top4` joins the Elo panel as a descriptive contender, labelled "history-mismatched: sees four
  anchors". It never replaces `prog_top4` as the comparator.

## Not yet written

The r882 inference adapter (scoring new proposals given anchors) is not part of this PR. It needs the r882 scoring
API that H82/H93 use. It is to be written and reviewed as a small follow-up before W1 releases the lock, so the
entry is ready when its turn comes. Until then this entry blocks nothing: the CPU arms and the primary contrast do
not depend on it.
