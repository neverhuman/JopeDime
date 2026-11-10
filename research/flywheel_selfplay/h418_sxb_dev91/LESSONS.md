# H418 lessons — the one GE replay divergence, and whether the screener cache needs a fix

Written 2026-11-10 (after the H418 closeout, v1/next `7cd5b86f6f`) by `cl_fwriter` (Claude Opus 5.5), on ENG:watcher's
4:20 PM MST request. No dev91 byte was read for this note: the probe below uses synthetic inputs, and the rest reads
code on `origin/main` (`aeef69e198`; screener bytes `2e3f2fa04c88`, the same bytes the H418 LOCK pins) and stored H418
harness records (program shas, generation counts, failure counts; no utility value).

## 1. What happened
- **The one failure.** In the deploy D2 replay of the GE search for (context `43022ae6a8bc`, seed 335102), one of the 12
  `AGG.g50.CV3@12` stock members differed: `35f192f2…`, which only GL had evaluated, replaced `261e8073…`. The value Δ
  was 3.43e-5.
- **Everything else matched.** The other 363 GE/GL replays were identical, and the GS replays were 182/182.
- **Disposition.** Owner Amendment 2 scored that cell at X0b; the descriptive deploy value moves the Elo by +0.9.

## 2. The published "likely cause" is not supported
`INCOMPLETE_1.md`, `AMENDMENT_2.md` §2 and the H418 README name a likely cause: the screener's per-(context, program)
score cache was shared by the two seeds of a context in one GE arm process, while each deploy ran one process per seed.
The checks below argue against that mechanism; none of them supports it.

| Check | Result |
| --- | --- |
| Is a cached score bitwise independent of the batch it was computed in? `~/jope_plan/evidence/H418/lessons/batch_invariance_probe.py` (xbabe2 evidence, not in git): the GE screener's fold-0 models (`H381-E2-S3fz`, scale `S3`, two training seeds), called exactly as `rapid_hypothesis_lab.Screener.score` calls them, on synthetic inputs | **Yes.** Bitwise identical (max abs Δ 0.0) for batch prefixes of 1–399 of 400, for a permuted batch, and for repeated calls, at 1 and 4 torch threads, with 2 and 6 views (`~/jope_plan/evidence/H418/lessons/probe_*.json`). On this build the cache returns exactly what a fresh call would (xbabe2 torch 2.7.0; the xbabe0 docker build is unchecked, §3) |
| Does the cache apply to GL? | No: `H415-L` has `set_blocks` (scale L), so `cache_scores` is False. GL's 182/182 exact replays therefore say nothing for or against the cache |
| Can two threads get different values for one program from the engine memo? | No. `gp_engine_h317.EvalEngine._evaluate` dedupes in-flight requests (`_inflight`) and writes the memo before the future resolves |
| Is selection order-dependent? | No. `_Pop.add` appends in request order, and `rank_key` breaks ties by sha |
| Did the cell have a timeout or failed evaluation? | No. GE `n_failed` = 0 for both seeds of `43022ae6`. GE's 7 failures sit in two other units (`9edb5a73` seed 335102: 3; `e9d0481c` seed 335101: 4) |

- **Conclusion: the mechanism is unknown.** The structural differences that remain between the arm and the replay:
  1. **Threading.** The arm ran the work list `(context, seed)` for both seeds through `L.run_parallel(..., jobs=12)`:
     12 threads in one process, sharing one runtime (`d91_h418.py`, the `gp` step). Each deploy ran one seed.
  2. **Runtime builders.** The arm used `h335_lab.build_rt`; the deploy used `deploy_h410.build_rt_v`.
  3. **Memo.** The deploy's memo was the concatenated GS + GE + GL memo, and the swapped-in member came from GL.
- **The stored records cannot localise it.** They hold no per-generation pick trace: the GE record has the result
  shas and generation of each program, and the D2 record has only the replay's stock output. Finding the first
  diverging generation would mean replaying the unit on dev91 SEARCH rows, which is closed. The cause can only be
  studied on TRAIN.
- **This file is the correction.** The "likely cause" sentences in `INCOMPLETE_1.md`, `AMENDMENT_2.md` §2 and
  `README.md` are left unchanged, because this publication is create-only. Read them together with this note.

## 3. Does `screener_h317.NeuralScreenerAdapter`'s cache need a fix on main?
- **No correctness fix found.** The cache is exact for the GE screener's scale-S3 fold-0 models on xbabe2's torch build
  (2.7.0, CPU). Not probed: the other folds, GS's scale-S models (same code path), and the torch build inside the
  xbabe0 docker image `h317-empty:v1` where the arms ran; the §4.3 test should repeat the probe there. The
  adapter never caches batch-coupled scale-L models.
- **Optional hardening, low priority:**
  - **(a)** A unit test asserting bitwise batch invariance for every cached scale. The guarantee is numerical, so a
    torch or BLAS change could silently break it.
  - **(b)** Scope the cache to one search unit, via a `reset_cache()` called by `unit_lf`, or a scope token in the key.
    A unit's scores then never depend on another unit's history, even if (a) ever fails.
  - Neither changes any stored result. Both belong in a later PR (work order queued in `~/ci_logs/agy_wo_ledger.md`).

## 4. Lessons for later waves
1. **One process per (context, seed) for GP arms whose results a deploy replays per seed**, or at least one seed per
   process, and no threads sharing a runtime across seeds. This removes difference 1 at no cost in compute. It is a
   precaution: the specific mechanism is unproven.
2. **Store a per-generation pick trace** (chosen shas, plus the margin between the k-th and (k+1)-th pick score) in
   each unit record. A replay mismatch can then be localised from records alone, without rerunning anything on a holdout.
3. **Before any holdout run, run a TRAIN-only arm-vs-replay identity test** with the real layout: 2 seeds, the real
   thread count and the real runtime builders. H416's TRAIN replays were single-seed, so they never exercised the
   two-seed, 12-thread arm.
4. **Pre-register how the replay gate fails.** For example: "a listed or unlisted single-cell divergence scores X0b,
   with the deploy value descriptive." One cell then needs no post-hoc amendment.
5. **Never publish a "likely cause" that hasn't been checked.** H418's was stated from structure alone and did not hold
   up once tested.

## 5. Open items (no PR now; ENG:watcher 4:20 PM MST: no new PR while 3 or more are open)
- **Work order** (`~/ci_logs/agy_wo_ledger.md`, queued 2026-11-10 4:29 PM MST): pick trace (§4.2), a
  one-process-per-seed driver option (§4.1), the TRAIN-only determinism test (§4.3), and cache hardening (§3 a/b).
- **Record correction:** this file, published create-only to v1/next on ENG:watcher's 4:55 PM MST instruction. No
  other H418 file was changed.

## Files (xbabe2 evidence, `~/jope_plan/evidence/H418/lessons/`; not in git)
| File | sha256 prefix | What |
| --- | --- | --- |
| `batch_invariance_probe.py` | `c170f4f9c74158c7` | The probe (torch; synthetic inputs only) |
| `code/multiscale_axial_bridge.py` | `f99726e42cfec848` | main's bytes at `aeef69e198` (`code/main_commit`) |
| `probe_fold0_t1.json` | `2ff22344cbc718a8` | 1 thread, 2 views, batch 64 |
| `probe_fold0_t1_v2_m400.json` | `da54cb697d90783f` | 1 thread, 2 views, batch 400 |
| `probe_fold0_t1_v6_m400.json` | `9f4bf0bcb6d41e62` | 1 thread, 6 views, batch 400 |
| `probe_fold0_t4_v2_m400.json` | `99df5ded76e3f5fa` | 4 threads, 2 views, batch 400 |
