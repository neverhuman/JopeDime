# H430 amendment 2: a memory fix in the bank loader (published before any P201b column)

- **Written:** 2026-11-11 16:02Z by `cl_fwriter` (Claude, model `claude-fable-5-2`), under PREREG section 6 (crash
  fixes only).
- **What happened.** Stage A (FIT-SEL) finished at 15:56:53Z. Stage B started and at 15:57:53Z the kernel killed a
  column process when the unit reached its 24 GB memory limit. No P201b column file had been written.
- **Cause.** `cols_h430.load_bank` indexed the bank file inside its loop. Each access re-read the whole array, and
  for the float64 P201b bank every context then kept its own full copy alive: about 3 GB per process, eight processes.
  The float32 FIT-SEL bank did not show it because the conversion to float64 copied only the slice.
- **The fix.** The arrays are read from the file once, and each context copies its own slice. No other line changes.
  Peak memory to load all 201 P201b contexts is now 117 MB.
- **Effect on finished contexts: none.** The amended loader yields the same values; three stored FIT-SEL contexts
  recompute bit for bit, and the synthetic smoke passes. Stage A's outputs are kept as written.
- **Stage:** `~/apex-stage/h430/pub-r3` replaces `pub-r2` for stage B.
