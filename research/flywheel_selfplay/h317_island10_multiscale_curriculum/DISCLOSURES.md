# H317 disclosures: reads made before the preregistration commit

This file lists everything any H317 process or sub-agent opened, before the prereg commit, beyond the
TRAIN-only boundary of PREREG §1, as each agent reported it. No DL-Val outcome or label value was printed or used, and
none informed any design choice. The guard v2 refusal log of production runs starts empty.

## Parent session (cl_fwriter)
- One infrastructure probe ran the champion DSL on one non-pilot TRAIN context, `002f7ead…`, on its native, SEARCH and
  AUDIT envelopes.
- From res-01 `selected.json`, `source_manifest.json` and `champions.json`/`winners.json` it read structure and counts
  only. For dev rows it read identifiers only (cid, family, dataset), to build the guard and the redacted view.
- From the H258 `inputs/r882/best_dev_checkpoint.json` (DL-Val R882 reference) it read the top-level keys, the
  `evaluation_contract`/`utility` key names and `len(contexts) = 91`. No utility values were read.
- From the h58 `cache_manifest.json` it read the full file (metadata), plus zip member listings (`unzip -l`) of the
  mixed npz files. No member data was read.
- It ran read-only ssh checks of the xbabe1 flywheel writer, its custody files and the GPU state.

## Leakage red team
- It ran one tree-wide `git grep -l` on `origin/v1/next`. That regex-scanned every blob, including the 91 dev table
  and pool blobs committed under `research/flywheel_selfplay/custody/files/`. Only file names were printed.
- A provenance sub-agent of that red team may have run `git grep` on `origin/main`/`origin/v1/next` with unknown
  pathspecs.
- It took a path-name inventory of the home directory (`all_paths.txt`, names only) and searched dev identifiers in
  those names.
- Its worker transduction tests used TRAIN contexts only. Its guard probes used synthetic paths.
- It did not open `full-packet.tar` or anything under h256/h257.

## Neural builder
- Before the warning it ran tree-wide `git grep -l 'context-feature-cache'` and `git log --all -S…` on `origin/main`
  and `origin/v1/next`. Paths and commit subjects were printed only.
- It read the TRAIN prefix of the mixed h58 npz through the H308 loader. A byte audit shows exact TRAIN-prefix ranges;
  the `cids`/`offsets` identifier members were read in full, by design.
- It copied the H189 TRAIN R882 per-context results (5 files; 0 DL-Val cids).
- It ran timing on permuted targets only.

## GP builder (catalog search for a pool sha → DSL map, since dropped)
- **res-01 inputs:**
  - `champions.json` and `winners.json` were parsed.
  - All 1,219 TRAIN pools were read for program_sha256/fitness/gate. fitness/gate were used only to count catalog
    coverage of the original 24-pilot top-32 rows; no GP decision used them.
  - One non-pilot pool, `002f7ead`, was printed in full.
- **Other files opened:**
  - `hwv-m3-full-t0150/proposer/native-admission-candidates.jsonl` was parsed fully (44,544 rows, all TRAIN).
  - `h202/inputs/real-catalog.jsonl`: first 700 B only.
  - h58 `journal_candidates.json`: first 600 B, plus `grep -o` of cids and key names.
- **Searches for one pool sha (`grep -rl -F`, 100 s timeout, no match, no content printed)** over `res-01/runtime`,
  `jope-flywheel/manifests`, `jope-flywheel/ledger`, `~/flywheel/data` (later denylisted; holds raw datasets, possibly
  including DL-Val datasets) and `~/flywheel/out`. A `grep -rl` over `~/jope_plan/evidence/RES-01` printed matching
  filenames only.
- **Git:** `git grep -l "res-01" origin/main -- '*.py' '*.md' '*.json'` and `git grep` for function names, both without
  the custody exclusion; filenames and .py lines printed. It also ran `git show` of the custody `INVENTORY.md` and the
  first 3,000 B of `MANIFEST.json`.
- **Before the denylist:** it read the evaluator's Rust source under `~/scratch/jopedime-dl-training-v1/build/` (source
  code only) and ran `strings` on the worker binary.
- **Pilot tables:** an early helper read the original 24 pilot TRAIN tables in full for row/column counts. Later
  checks read 17-byte headers only.

## Method brainstorm, stats red team, evidence miner, Gemini audit, ops red team
- Their real-data reads were TRAIN pool and champion summary statistics, TRAIN table headers, git-published wave
  documents and code, and committed JSON in git.
- The Gemini audit read the H256/H257 code and committed telemetry only, never their caches.

## Assessment
- Several processes read bytes of DL-Val-bearing files without printing them: regex scans of git blobs, and a grep
  over `~/flywheel/data`. None of these feeds any H317 artifact. Every H317 input is TRAIN-only and was rebuilt or
  verified inside the sandbox (tables/pools hard-linked from res-01; views re-encoded from TRAIN tables; R882 from
  TRAIN-only H189 files).
- DL-Val is in any case a reuse cohort for Stage V (PREREG §1).

## Added before the GP-block launch (amendment A2)
- **P205 timing context.** P205 (all families) made the GP builder's earlier timing context `8f4cde4314f5…` (6 columns)
  a pilot context. Before launch, that context had one full real-worker H0.4→H0.6 timing job, with the test-only
  fallback screener and the numpy coder; its memo is separate, under `gp/timing_root`. No decision or design choice
  used its values, and the production run uses a different memo root, so no cache is reused. The context is kept in
  P205 (dropping it would be a post-hoc pilot change) and is flagged in `LAUNCH_GP.json`.
- **Fake-worker fixtures.** Two test fixture contexts that are P205 pilots (`64f250c8…`, `8463e453…`) were used only
  with the fake worker; no real evaluation was run on them.

## Learner-build agent (superset worker; reported 14:43Z)
- **res-01 table headers, outside the guard (breach).** Before the data rules arrived, it read the 17-byte headers
  (row count, width) of the first 400 res-01 tables without the guard installed. 29 of those are DL-Val tables. No
  labels or features were read; only aggregate percentiles were printed, and nothing fed any H317 choice.
- **`~/scratch`.** It read the pinned worker's source snapshot `s1-src-6bf65b12` (copied), dependency/fingerprint
  files, `s1-prod/ledger.header.json` and `s1-control/receipt.json` (which names 3 control contexts by dataset hash,
  cut id and size), the source manifest and `steps.jsonl`. A `grep -rl` over s1 receipts timed out and printed
  nothing.
- **JopeDimeMicro.** A `grep -rl` over `results/**` for the binary sha printed one path; a `.rs` grep timed out. It also
  read the bin manifest, `rust_evaluator.py`, `rank_prepare.py` and `assets/projections/program-feature-names.json`.
- **Git and sub-agent.** One tree-wide `git grep -l s1main origin/main -- .` without the custody exclusion printed
  file names only. Its sub-agent hashed the scratch snapshot and read git objects; whether its greps excluded custody
  is unverified.
- **Empirical checks** used TRAIN non-pilot contexts only, from P96 at the time. Under P205 every family is a pilot
  family, so some of those contexts are P205 pilots. The checks were standalone learner probes with no H317 arm, memo
  or decision involvement.

## Superset worker and seed scheme F
- The superset worker (`bootstrap.s1main.superset-2a848c8f8085`) passed parity and its unit tests at 14:43Z. That was
  after the GP-block launch at 14:23:58Z, so under PREREG §2.5 it was not admitted and seed scheme F is absent from
  H317's decisions.
- Any run of the new heads (xtrees / rforest / oblivious) on the pilot is EXPLORATORY and report-only. Such a run is
  not eligible for LOCK and is labelled as such.
