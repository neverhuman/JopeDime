# Numbers trace: the CLOVER write-up and the README

This file records where every number in [`TECHNICAL_WRITEUP.md`](TECHNICAL_WRITEUP.md) and in the
repository [`README.md`](../../README.md) comes from, how it was checked, and what the check found.
The audit ran on 2026-10-16 on xbabe2, at `origin/main` `827881ef7` (version 0.3.0, untagged), and
its corrections are in the same change as this file.

Three kinds of check were made:

1. **Source pins.** Every claim in [`claims.json`](claims.json) names its source files and pins one
   SHA-256. Each source was resolved on xbabe2 and re-hashed.
2. **Values.** Each claim's value was looked up at the cited line or JSON path. Where the value is
   derived (a sum, a ratio, a rounding), it was recomputed from the source.
3. **Printed numbers.** Every number printed in the write-up was matched to a claim that the
   write-up cites in the same line or section.

The claim sources are host-local files on xbabe2, so only the third check can run from a clone.

## How to re-check

From a clone, in CI's hygiene lane (`deploy/ci/check.sh`) and in the `jope-records` unit test
`paper_trace::tests::the_committed_documents_agree`:

```sh
cargo run -q --locked -p jope-records -- paper-trace
```

It fails when:

- a number printed in the write-up is not a number of a cited claim's statement or value, or
  that number at the precision printed;
- a claim's `tex` prints a number that is not one of its own statement or value numbers at the
  precision the `tex` prints, so a stale LaTeX display fails instead of widening what passes;
- the Appendix G totals or the Section 10.4 pending table disagree with the ledger;
- a cited claim id is missing, or a ledger claim is never cited;
- this file's claim table or exemption table is out of step with the ledger or the write-up;
- the README's version or an "At a glance" count disagrees with the file its footnote names.

What it does not check: a printed number passes if it equals, at the printed precision, any number of
any claim cited in its line or section, so a value swapped for another cited claim's value passes.
Single-digit integers and numbers inside inline code are not checked. Claim values are compared with
their sources only by the record in this file; `--sources` re-hashes each source file, it does not
re-read the value.

On xbabe2, `paper-trace --sources` also re-hashes every claim source under the roots that
`claims.json` declares. On the initial 2026-10-16 audit of `C01`–`C183` (177 claims) it reported
175 claims re-hashed to their pin, and with the `C184`–`C190` addendum (184 claims total) it
reports 182 claims re-hashed to their pin, plus 2 that name no single file (C24 is derived from C10
and C23; C115 is a pending result that names a directory).

## Findings and corrections

Each correction below was checked against the source named. None changes a conclusion of the
write-up. No pending claim was filled, and no sealed or held-out data was opened.

| # | Where | Printed before | What the source says | Change |
| --- | --- | --- | --- | --- |
| 1 | Section 10.3, C140 | last evaluated epochs range "from 0.8964 to 0.9022" | the lowest is sweep 6's L6 + 0.2·KL arm at its incomplete epoch 8, 0.8963494, the value the next sentence names | 0.8963 to 0.9022 |
| 2 | Section 10.3 table, C137 | sweep 4a, L_PI 0.01: 0.012–0.020 | probe KL 0.011465 at epoch 1 to 0.019903 at epoch 4 | 0.011–0.020 |
| 3 | Section 10.3 table, C137 | sweep 6, L6 only: 0.031–0.032 | 0.031133 and 0.031480, which are 0.031 at both epochs | 0.031 at both epochs |
| 4 | Section 10.3.1 | "three further scratch training runs and three judge looks" | four training receipts (the matched control, R25, R26, R27) and five judge records dated 2026-10-13 | four runs and five judge records, new claim C182 |
| 5 | Sections 9.7 and 10.3.1, C173, C177 | the 885 contexts were held out "of every fine-tuning arm" | sweeps 3 and 4a had no holdout and trained on them | scoped to sweeps 4b–6 and every later run |
| 6 | Section 10.3.1 | the consistency check "passed for all 6,888 terms" | 6,888 checks per checkpoint, passing under an absolute-error escape registered at 11:20:58Z, before the look; under the original relative rule, 4 feasibility terms failed (absolute differences about 1.3e-8) | stated, new claim C183 |
| 7 | Sections 9.7 and 10.3.1 | "found the anchored fine-tune worse than the frozen v1 policy"; "The fine-tune forgot lineages it did not revisit" | the verdict is a non-qualifying REGRESSION ALERT for three checkpoints, one of them R25 (which adds a neighbour term), and both strata are marked not decisive | "raised a regression alert"; the strata reading marked directional |
| 8 | Section 10.3 | the other four arms "have no paired verdict" | sweep 3's two arms were judged against their own L6 control at epochs 1–3 | "no paired verdict against frozen v1" |
| 9 | Section 10.3, C139 | held-out L6 "did not improve in any arm" | true at each arm's last evaluated epoch; two sweep 4b arms were marginally below the start at epoch 1 | qualified |
| 10 | Section 12, C141 | the two frozen v1 L6 values "are not reconciled here" | the epoch-40 record evaluates step 17,416; the re-evaluation is the terminal step 17,773, 357 updates later | reconciled, new claim C181 |
| 11 | Section 10.1.2, C78 | LightGBM "0.785" with no pool named | 0.785 is equal-lineage R² on the corrected 7,120-view pool; row R² there is 0.752, beat10 AUC 0.903; the clean-pool reference is also 0.785 but a different pool; on the original 7,087-view pool the yardstick was R² 0.718 and AUC 0.903 at repeat 0 | pool and metrics stated, new claims C178 and C179 |
| 12 | Section 8 | the sealed families' role was implicit | the corpus authority accepts only the one-holdout role map (2026-10-07 ruling) | stated, new claim C180 |
| 13 | `claims.json` roots | CAPTURE, ARENA, HOST at `/home/ubuntu/jopedime_*` | moved to `/home/ubuntu/jope-evidence/` on 2026-10-15 | paths updated; REPO written as a path; PROBE and CANON added |
| 14 | Header, Section 10.3.1 | EI-1 "planned only" with no date | true at the 2026-10-13 revision | the scope date is stated; EI-1 marked "as of this revision" |
| 15 | Section 6.4, C93 | "About 98% of proposals" | about 98.5% of outbox rows (line 43); the "about 98% native" at line 39 counts verified outcomes, not proposals | 98.5% |
| 16 | `claims.json` `tex`, C140, C93, C168 | C140 $[0.8964, 0.9022]$; C93 ≈98%; C168 "17 of 24", which its statement and value did not hold | the corrected record; for C168, the paired summary covers 17 of the 24 planned contexts (`GP_EXPANSION_STUDY.md:89-91,222`) | C140 and C93 `tex` print their record: $[0.8963, 0.9022]$, ≈98.5%; C168's statement and value now hold the 17 of 24 |

Ledger totals move from 171 claims (112 established, 20 preliminary, 16 non-qualifying,
23 pending) to 177 (118, 20, 16, 23), with the six new established claims C178–C183.

The Section 10 values were re-derived in a separate pass: 182 printed values in Sections 10.2, 10.3
and 10.3.1, recomputed from the run receipts and judge records. 177 matched at the printed
precision. The rest are findings 1 to 4 above.

## Known facts checked

The audit brief named four facts to check the documents against.

| Fact | Where it stands |
| --- | --- |
| The wave-1/2 "X on residuals" rows score foundation's own 0.476 / 0.4668; they are not a residual model | Neither the write-up nor the README reports them, and no tracked document presents them as encoder lift. The source records them relabelled "foundation + X on residuals" (`ENCODER_RESEARCH_MASTER_SUMMARY.md` §6, item 2, under the PROBE root). |
| The starlight regressor is about vanilla TabPFN-3: research-only, not shipped or a headline | The write-up's policy reads only Kumo-Tabular Small (C03; `deploy/dl-flywheel/docs/model.md`). The model bundle marks `starlight_core_regressor` restricted, with redistribution prohibited and no policy eligibility. The README's design diagram listed "foundation" without saying so; a note now does. |
| LightGBM outcome-predictor R² 0.718; beat10 AUC 0.903 | That is repeat 0 of the yardstick on the original 7,087-view pool (`figs/dl_waves/confirm_report.json` `.lgb`). The write-up's comparator (C78) is a different pool and weighting, so both are now stated (C178, C179). |
| A single sealed meta-test holdout (2026-10-07 ruling); never opened | Stated in Section 8 (C180). No evaluation in the write-up uses the sealed families. This audit did not open them. |

## README

### "At a glance" and the counts in prose

Recomputed on 2026-10-16. `paper-trace` re-derives every value below from committed files except
the `--help` count, which needs a build; the fresh-clone run printed it.

| README value | How it was recomputed | Result |
| --- | --- | --- |
| `jope 0.3.0` | `cargo run --locked --bin jope` (fresh clone) | `jope 0.3.0` |
| 26 members; 19 at 0.3.0, 7 at 8.0.0; 8 default | `cargo metadata --locked --no-deps` | 26; 19 + 7; 8 |
| 17 subcommands plus hidden `motifs` | `jope --help` and `jope motifs --help` from the fresh clone of `73f9327fc` (logs in the audit directory, `fresh-clone-73f9327fc-help/`) | 17 listed, plus clap's own `help`; `motifs` is hidden and its help works |
| 6 emit targets | the single-language arms of `fn languages` in `crates/jope-cli/src/kernel.rs` | 6 |
| Rust 4,993 listed, 26 packages, 72 ignored; Python 182 | `docs/parity/test-inventory.json`, refreshed by `deploy/ci/test-inventory.sh --update` (it recorded 4,189 and 66, and missed 785 tests that `main` lists) | 4,993; 26; 72; 182 |
| 9 PR jobs; 11 `full-` families | `.github/workflows/ci.yml` jobs; `deploy/ci/coverage.json` lanes | 9; 11 |
| 678 coverage items, 24 lanes | `deploy/ci/coverage.json` | 678; 24 |
| Capability matrix 824 / 353 / 61 / 1 / 402 / 7 | the total row of `docs/parity/capability-matrix.md` | as printed |
| 40 synthetic fixtures, 10 test datasets | `datasets/synthetic/catalog.json`, `datasets/manifest.json` | 40; 10 |
| 1,184 `.py` files, 159 under `python/`; 6 ADRs | `git ls-files` | 1,184; 159; 6 |

The generated blocks (arena, same-cut and emit latency) are checked by
`jope-records readme-benchmarks --check`. Both `jope-arena elo --check` commands in the arena block
reproduced their committed ladders byte for byte from a fresh clone.

### README corrections

| Where | Problem | Change |
| --- | --- | --- |
| Arena block (generated) | cited `python/src/jope/research/index_combined_regression_evolution.py:183`, a file the Python removal archived | the renderer writes archived references as `8a55c4fef:<path>:<line>`, which `git show` reads; line 183 is the `best_rmse` minimum |
| At a glance, Rust tests | `docs/parity/test-inventory.json` recorded 4,189 tests (66 ignored); its own guard adds tests without failing, and `cargo nextest list` on this tree lists 4,993 (72 ignored): 785 that `main` already had and the inventory missed, and 19 added by this change | inventory refreshed with `deploy/ci/test-inventory.sh --update`; README row updated |
| Install and check | `pipeline run --emit rust,python,cpp17,cpp20,js` emitted 2 artifacts in the fresh-clone run (the selected head has no C++ or JavaScript emitter) and the README did not say this can happen | a comment says unsupported targets are reported as not emitted |
| CLOVER quick start, steps 3–5 | their inputs (`coverage.json`, `protocol.json`, the roster and its pack, `GP_RESULTS_INDEX.json`) are not in Git, so they cannot run from a clone; `ingest-coverage` left out the held-out context list that the pre-registered J9 run excluded | a host-inputs table; `--exclude-contexts`; the protocol identity read from the run receipt instead of hashing a `protocol.json` |
| CLOVER quick start, steps 3–5 | `cargo run` without `--release`: in the fresh-clone run, `evaluate` took 767 s and `flywheel-distill` 8,125 s (2 h 15 min) in the debug profile, against 44 s and 570 s in release | the commands use `--release`, and a sentence gives both timings |
| README length | 395 lines at `73f9327fc`, against 368 on `main`; the owner asked for a short README, and a review of `73f9327fc` flagged it | steps 3–5, which need host inputs and cannot run from a clone, moved unchanged to `docs/clover-host-steps.md`; the encoder note shortened; 331 lines |
| Intended pipeline | listed the foundation and target encoders without their status | a note on their research-only status |
| S3 row, layout table, CI paragraph, further reading | no mention of the offline CLOVER tools, the numbers trace or the paper | added |



The quick start was run from fresh clones; see [Fresh-clone run](#fresh-clone-run).

## Figure 1

Figure 1 (`figures/why-it-matters.svg`, under the abstract) is drawn only from established claims.
It shows why the work matters: what the frozen first generation could not do.

| Panel | Data | How it was checked |
| --- | --- | --- |
| (a) full validation loss by closed epoch, the plateau shaded | `figures/v1_full_validation.dat`, SHA-256 `8155223d…35e3`: C27's pin | `extract_v1_curves.sh` (scaffold, `tex/figures/scripts/`) re-run on 2026-10-16 against the CAPTURE root's `copy/training/`. It reproduced the file byte for byte, and its 40 input hashes match the scaffold's `SOURCES.sha256`. |
| (b) the outcome head's pairwise ranking accuracy on fresh programs, chance marked | C76's value and lineage count, read from `claims.json` | C76's pin re-hashed on xbabe2 (Claim trace below) |

`paper-trace` re-renders the SVG from those two inputs and fails when:

- the data file is not C27's pin;
- the plateau recomputed from the data is not C27's value (`[0.8968201, 0.8978393] over 31 epochs` from epoch 10, with epoch 9 outside the band);
- the first, best and last losses are not C26's value;
- C76 cannot be read;
- the committed SVG differs from the rendering;
- the write-up no longer shows the figure.

`jope-records paper-trace --write-figure` rewrites the SVG. The planned three-panel teaser is still
only a specification, because its panels need qualified receipts that do not exist yet (C114).

## Other documents

- [`CHANGES.md`](CHANGES.md) records each revision; this audit adds its own section there.
- The LaTeX scaffold (`tex/`) and the paper tools live in a lane directory on xbabe2, not in this
  repository. They still mirror the 2026-10-10 draft, as `CHANGES.md` says, and were not audited.
- `docs/asm-maintainer-manual.pdf` and `docs/asm-evidence-reference.pdf` are the private ASM
  manual, not the write-up. Their SHA-256 values match `docs/internal/asm-manual/render-receipt.json`,
  and all 197 render inputs it pins are unchanged. Their content was not re-audited here.
## Fresh-clone run

On 2026-10-16 the README's shell blocks were run on xbabe2 from fresh clones in a scratch
directory under `/tmp`. The audit brief asked for that, but it departs from the repository rule
(AGENTS.md: never clone into `/tmp`); the clones held no unique work. Each command ran
from the clone's root, in README order, and its exit status, wall time and output were recorded.
Copies of the logs are in `/home/ubuntu/jope-evidence/paper-readme-audit-20261016/` on xbabe2.

| Clone | Commit | Source |
| --- | --- | --- |
| `main` | `827881ef7` | `git clone https://github.com/neverhuman/JopeDime.git` |
| first audit commit | `9fdff904c` | `git clone --no-local` of this branch |
| README `--release` change | `73f9327fc` | `git clone --no-local` of this branch |

| README block | `main` `827881ef7` | this branch |
| --- | --- | --- |
| Arena ladders: 2 × `jope-arena elo --check` | exit 0; both ladders byte-identical to the committed files | at `9fdff904c`: the same |
| Same-cut reproduction: 3 commands | not run | at `9fdff904c`: all exit 0. The 10-dataset run took 273 s, and `elo --check` found the deep ladder byte-identical to `assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv` |
| Install and check: 11 commands | all exit 0. `jope 0.3.0`; `Verified 40 synthetic datasets, 828029 bytes, offline.`; `pipeline run` wrote 2 artifacts and reported `cpp17`, `cpp20` and `javascript` as not emitted for the selected head; `verify-receipt` ok (7 stages, 2 artifacts) | at `9fdff904c`: the same |
| Emitter proof: `JOPE_PARITY_STRICT=1 cargo test --locked -p feat-math` | exit 0; 1,168 passed, 12 ignored, 0 failed; 1,854 s | at `9fdff904c`: the same counts; 1,720 s |
| CLOVER steps 1–2: 2 × `cargo test` | exit 0; 46 passed | at `9fdff904c`: the same |
| CLOVER steps 3–5: 7 commands | all 7 fail (exit 1 or 2): their inputs are not in Git, and the README did not list them | at `73f9327fc`, exactly as written (`cargo run --release`; then in the README, now byte for byte in `docs/clover-host-steps.md`): all 7 exit 0 in 820 s, build included. At `9fdff904c`, in the debug profile: all 7 exit 0, `flywheel-distill` after 8,125 s |
| Release verification | not run: it needs a published tag, and v0.3.0 is not tagged | not run, for the same reason |

At `73f9327fc` only the CLOVER steps 3–5 block was run again. The commits after `9fdff904c` change `jope-records`, the paper and README prose, plus the `--release` flag of those seven commands; the other blocks run none of them. Moving steps 3–5 to `docs/clover-host-steps.md` afterwards left the seven commands byte-identical.

The CLOVER host inputs were copied from xbabe2's lane records into the clone. They are R22
scratch data: scratch-only and non-qualifying. The outputs below show only that the commands
run; they are not results.

| README path | Copied from | SHA-256 |
| --- | --- | --- |
| `target/pools/coverage.json` | `LANE:rlfw-takeover/lpi-scratch/receipts/runs/coverage.json` | `73c15fbb6a70de91…` |
| `target/pools/heldout.ids` (885 ids) | `LANE:rlfw-takeover/lpi-scratch/work/heldout.ids` | `74442da022b58cc4…` |
| `target/pools/roster.jsonl` | `COORD:lanes/CX-MET/ingest/joined-roster-v2/train.jsonl` | `104849d13aebb4f0…` |
| `target/pools/roster-vectors.f32` | `COORD:lanes/CX-MET/ingest/joined-roster-v2/train-vectors.f32` | `99959f86866260aa…` |
| `target/retention/GP_RESULTS_INDEX.json` | `COORD:lanes/GP-INDEX/GP_RESULTS_INDEX.json` | `9c6e6cb3f99ca9c9…` |
| `PROTOCOL_SHA256` | `.inputs.snapshot_spec_sha256` of `LANE:rlfw-takeover/lpi-scratch/receipts/runs/claude-rlfw-lpi-scratch-20261010T174612Z/receipt.json`, the run that wrote the coverage document | `05dc9558e3646106…` (the value itself) |

What the seven commands printed:

1. `ingest-coverage`: 2,267 train contexts, 885 left out by the list, 1,382 emitted.
2. `freeze`: 1,222 pools (160 excluded) from 21,199 rows.
3. `ingest-roster-features`: 1,222 context vectors.
4. `evaluate`: strongest baseline `ridge_program`, `ranking_learnability NOT_ESTABLISHED`. The
   receipt (`b9f6dd48…`) was the same from the debug and the release build.
5. `distill-gp-index`: 8,908 rows.
6. `flywheel-distill`: 5 rounds, `flywheel_established false`, `ranking_learnability NOT_ESTABLISHED`;
   the same receipt id (`bed5ad79…`) from the debug run and both release runs.
7. `inspect-parquet`: the shard's column and page summary.

Timings:

| Command | debug profile, `9fdff904c` | release binary, `9fdff904c` | `cargo run --release`, `73f9327fc` |
| --- | --- | --- | --- |
| `evaluate` | 767 s | 44 s | 51 s |
| `flywheel-distill` | 8,125 s (2 h 15 min) | 570 s | 558 s |

That gap is why the commands now use `--release`.

## Claim trace

One row per claim in `claims.json`, with its status, its value as the ledger states it, the first
of its sources, the source-pin result on xbabe2 on 2026-10-16, and how the value was checked.
"Value found at the cited locator" means every number of the value occurs at the cited line or
JSON path. Rows with a note were derived or corrected as the note says.

| Id | Status | Value | Source (first part) | Pin | Value check |
| --- | --- | --- | --- | --- | --- |
| C01 | established | 893,957 | `LANE:w10/operator/evidence/r17-final-capture.json#.checkpoint.manifest.body.parameters` | matches | value found at the cited locator |
| C02 | established | d=128, L=2, H=4, d_ff=512, p=0.1, V=282, T_max=640 | `LANE:w10/operator/evidence/r17-final-capture.json#.checkpoint.manifest.body.model_config` | matches | value found at the cited locator |
| C03 | established | 2,051 = 2,048 + 3 | `REPO:deploy/dl-flywheel/docs/model.md:27` | matches | value found at the cited locator |
| C04 | established | clip [-10, 10] | `REPO:deploy/dl-flywheel/docs/model.md:31` | matches | value found at the cited locator |
| C05 | established | 6,640 | `CAPTURE:copy/training/progress.json#.proposal_catalog` | matches | value found at the cited locator |
| C06 | established | lr 3e-4; wd 0.01; clip 1; x0.5, patience 5, floor 1e-6 | `REPO:deploy/dl-flywheel/docs/model.md:35` | matches | 3e-4 is printed 0.0003 at model.md:35 |
| C07 | established | 16 contexts; 8,192 tokens; <=8 candidates | `REPO:deploy/dl-flywheel/docs/model.md:39` | matches | value found at the cited locator |
| C08 | established | 1.00 / 0.25 / 0.10 / 0.25 / 0.10 / 0.05 | `REPO:deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/objective.py:30` | matches | value found at the cited locator |
| C09 | established | 10,751,892 B; 48 + 144 tensors; 2 RNG states | `LANE:w10/operator/evidence/r17-final-capture.json#.checkpoint` | matches | value found at the cited locator |
| C10 | established | 2 per update | `REPO:deploy/dl-flywheel/docs/model.md:41` | matches | value found at the cited locator |
| C11 | established | 1.4826; 1.349; 1e-12; 1e-6; log 1000; 1e6 | `REPO:deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/labels.py:9-14,46,81-98,181-` | matches | log 1000 is log(1e3) at labels.py:46 |
| C12 | established | 90 s | `REPO:deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/labels.py:34` | matches | value found at the cited locator |
| C13 | established | cap 1; boost 2 at 0.9; >=3; delta 1; tau 0.25 | `REPO:deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/labels.py:53-57` | matches | value found at the cited locator |
| C14 | established | 14,987 / 3,003 / 12,581 / 7,120 (6,857 + 263) / 5,461 | `REPO:deploy/dl-flywheel/docs/model.md:17` | matches | value found at the cited locator |
| C15 | established | 1,068 (1,014 + 54); ~5% | `REPO:deploy/dl-flywheel/docs/model.md:23` | matches | value found at the cited locator |
| C16 | established | 3,859; 3,755 unknown; 104 recorded | `REPO:docs/datasets.md:59-61,111-112` | matches | value found at the cited locator |
| C17 | established | ruling | `REPO:docs/datasets.md:116-134` | matches | value found at the cited locator |
| C18 | established | 99 / 263 | `CAPTURE:copy/training/validation-original-epoch-000002.json#.denominators` | matches | value found at the cited locator |
| C19 | established | 6,859+264 -> 6,982+266 | `REPO:deploy/dl-flywheel/docs/continuous-runtime-pins-20261009.json#.interpretation` | matches | value found at the cited locator |
| C20 | established | 429 updates; 6,857; 263 | `REPO:deploy/dl-flywheel/docs/model.md:7` | matches | value found at the cited locator |
| C21 | established | 844.3 s; 309.4 / 252.6 / 29.0 s | `REPO:deploy/dl-flywheel/docs/model.md:93` | matches | value found at the cited locator |
| C22 | established | 17,344 / 17,773 / 40 / 284,006 / 6,982 | `LANE:w10/operator/evidence/r17-final-capture.json#.ancestry` | matches | value found at the cited locator |
| C23 | established | 3.550 s per update (0.2817 updates/s) | `CAPTURE:copy/training/optimizer-history.jsonl (first and last rows, field at)` | matches | optimizer-history: (t_last - t_first) / 17,343 intervals = 3.5502 s; 1/3.550 = 0.2817 |
| C24 | established | ~2,028 proposals/h | `derived from C10 and C23` | derived (no file) | derived: 2 x 0.2817 x 3600 = 2,028 |
| C25 | established | 9 halvings; floor from step 5,677; 12,096 updates | `CAPTURE:copy/training/loss-history.jsonl (rows with plateau.lr_reduced=true)` | matches | loss-history lr_reduced at 2,477 ... 5,677 (9 rows); 17,773 - 5,677 = 12,096 |
| C26 | established | 1.0055211870 -> 0.8968201270 (ep 28) -> 0.8974925443 (ep 40) | `CAPTURE:copy/training/validation-original-epoch-000002.json#.loss` | matches | value found at the cited locator |
| C27 | established | [0.8968201, 0.8978393] over 31 epochs | `CAPTURE:copy/training/validation-original-epoch-0000{10..40}.json#.loss` | matches | value found at the cited locator |
| C28 | established | CE 0.6376->0.5477; q 0.1804->0.1809; BT 0.4719->0.4136; list 1.9985->1.9736; rt 0.1001->0.0758 | `CAPTURE:copy/training/validation-original-epoch-000002.json#.terms` | matches | value found at the cited locator |
| C29 | established | val 1.3813796 -> 0.8705179 (step 16,813) -> 0.8712129; train 1.4274187 -> 0.8807968 | `CAPTURE:copy/training/loss-history.jsonl (rows batch=429, 16813, 17773)` | matches | loss-history rows 429 / 16,813 / 17,773 give 1.3813796 / 0.8705179 / 0.8712129 (validation), 1.4274187 -> 0.8807968 (training) |
| C30 | established | 35,546; 2,734; 2; 43; 15; 32,752 | `LANE:w10/operator/evidence/r17-final-capture.json#.outbox` | matches | outbox parent + v1 states: 838+1,896 = 2,734 published; 12+31 = 43 failure; 8+7 = 15 unknown; 858+34,688 = 35,546 written |
| C31 | established | runtime_limit_stopped_idle; reserve 180 s | `LANE:w10/operator/evidence/r17-final-capture.json#.terminal` | matches | value found at the cited locator |
| C32 | established | 52,329 entries; 3,272,631,982 B; 0 writes | `LANE:w10/operator/evidence/r17-operator-receipt.json:20-47` | matches | value found at the cited locator |
| C33 | established | 0 mismatches | `LANE:w10/operator/evidence/cuda-parity-014611.json:13-38` | matches | value found at the cited locator |
| C34 | established | identical sha256 ad23c53c... and 083ec20b... | `LANE:v2build/real/BUILD-QUALIFICATION-v1.json:9-28` | matches | value found at the cited locator |
| C35 | established | 516.36 s | `LANE:v2build/real/BUILD-QUALIFICATION-v1.json:65` | matches | value found at the cited locator |
| C36 | established | 4.12 s; 1.16 s | `LANE:v2build/real/BUILD-QUALIFICATION-v1.json:64,66` | matches | value found at the cited locator |
| C37 | established | 0.4355 s | `LANE:v2build/real/BUILD-QUALIFICATION-v1.json:40` | matches | value found at the cited locator |
| C38 | preliminary | 0.7399 updates/s (n=3) | `LANE:v2build/real/BUILD-QUALIFICATION-GPU-v1.json:4,24,30` | matches | value found at the cited locator |
| C39 | established | 1e-9 | `REPO:deploy/dl-flywheel/docs/model.md:71` | matches | value found at the cited locator |
| C40 | established | 3 results; 486 predictions | `REPO:deploy/dl-flywheel/docs/evidence.md:121` | matches | value found at the cited locator |
| C41 | established | 41 / 1,554 (2.6%) | `LANE:c7/rootcause_design.md:46,88` | matches | 41 / 1,554 = 2.64% |
| C42 | established | 41/41; 342/1,554; 1,636/6,640 | `LANE:c7/rootcause_design.md:81,86-88` | matches | value found at the cited locator |
| C43 | established | 12.0% [8.7, 15.9]; 0/1,212 (<=0.25%) | `LANE:c7/rootcause_design.md:99-100` | matches | value found at the cited locator |
| C44 | established | 114 observations | `LANE:c7/rootcause_design.md:63-65` | matches | value found at the cited locator |
| C45 | established | 253/253; 2,557/2,558 | `LANE:c7/rootcause_design.md:74-77` | matches | value found at the cited locator |
| C46 | established | 29/29; 200/212; p=3.4e-28 | `LANE:c7/rootcause_design.md:209-220` | matches | value found at the cited locator |
| C47 | established | 554/800; bin 3 vs 0; 0.0986 | `LANE:c7/rootcause_design.md:175-186` | matches | max error 0.0986 is on rootcause_design.md:187, one line past the cited range |
| C48 | established | 3.3e-8 .. 0.344; median 6.7e-3; 39/41 | `LANE:c7/rootcause_design.md:66-71` | matches | value found at the cited locator |
| C49 | established | 1.82.0; #128254; 1.97.1 | `LANE:c7/rootcause_design.md:20-27` | matches | value found at the cited locator |
| C50 | established | 3,927 keys; 0 executable | `LANE:registry_v2/DELIVERY-r2.json:1245-1250` | matches | value found at the cited locator |
| C51 | established | 2,774 / 2,774 | `LANE:registry_v2/DELIVERY-r2.json:1257-1258` | matches | value found at the cited locator |
| C52 | established | 2,185 calls; 0 rejected; 0 evaluated; 31.1 s | `LANE:registry_v2/DELIVERY-r2.json:395-411` | matches | value found at the cited locator |
| C53 | preliminary | 37.58 ms <= 50 ms | `LANE:w4/DELIVERY.json:233-234` | matches | value found at the cited locator |
| C54 | preliminary | 124,343 verified/h | `LANE:w4/DELIVERY.json:197` | matches | 124,342.79 rounds to 124,343 |
| C55 | preliminary | 344.9 -> 6.5 ms; 21.8 -> 2.3 ms | `LANE:w11/DELIVERY.json:310-324` | matches | value found at the cited locator |
| C56 | established | $0.023; $0.004; $0.005/1k; $0.004/10k | `LANE:w6/pricing/s3-us-east-2-price-receipt-v1.json:41,50,104,131,151` | matches | value found at the cited locator |
| C57 | preliminary | 4.69 MB; 4.45 MB; 7 PUT | `LANE:w6/DELIVERY.json:501-508` | matches | value found at the cited locator |
| C58 | preliminary | 225.12 GB/day; +$155.33/month per month | `LANE:w6/DELIVERY.json:430` | matches | 225.12 at DELIVERY.json:425, 155.3328 at :430 |
| C59 | preliminary | $54.43/month; ~$7.8/month bundled | `LANE:w6/DELIVERY.json:415,427` | matches | value found at the cited locator |
| C60 | preliminary | +$0.09 .. $0.89/month per month | `LANE:w6/DELIVERY.json:415` | matches | value found at the cited locator |
| C61 | preliminary | ~4.4 days | `LANE:w6/DELIVERY.json:416` | matches | value found at the cited locator |
| C62 | established | 1% | `LANE:prereg/keeper-audit-salt-v1.json (sha256 168bf9fd...)` | matches | value found at the cited locator |
| C63 | established | 23 + 36; 10/10 | `LANE:c10/DELIVERY-r7.json:253,268 and .fault_injection` | matches | value found at the cited locator |
| C64 | established | 22 + 30; 8/8 | `LANE:c10/DELIVERY-r6.json:242,257 and .fault_injection` | matches | value found at the cited locator |
| C65 | established | 30; 17/17 | `LANE:w3/DELIVERY-r2.json:137 and .fault_injection` | matches | 17 injected faults, all caught |
| C66 | established | 29; 6 + 5 guards | `LANE:w4/DELIVERY-r2.json:367` | matches | value found at the cited locator |
| C67 | established | 27 | `LANE:w6/DELIVERY.json:211` | matches | value found at the cited locator |
| C68 | established | 17; 11/11 | `LANE:w8/DELIVERY-r2.json:87 and .fault_injection` | matches | 11 injected faults, all caught |
| C69 | established | 32; 31/31 | `LANE:w10/DELIVERY-r2.json:461 and .fault_injection` | matches | 31 of 31 injected faults caught |
| C70 | established | 22; 8/8 | `LANE:w11/DELIVERY.json:208 and .fault_injection` | matches | value found at the cited locator |
| C71 | established | 53; 32 kept; 21 excluded; B=8; tau=0.01; rerank finished 2 min 21 s after the split | `ARENA:runs/d5b/analysis-completed-1791577670/analysis.json#.coverage.per_game.rerank_T1_D5` | matches | rerank completion 20:27:00.77Z vs split sealed 20:24:39.70Z = 2 min 21 s |
| C72 | established | 0.109375 [0.03125, 0.203125]; 0.078125 [0.015625, 0.15625]; 0.140625 [0.0625, 0.234375] | `ARENA:runs/d5b/analysis-completed-1791577670/analysis.json#.games.baseline_T1_D5b.metrics[` | matches | analysis.json metrics[0.01] means and CIs match at the printed precision |
| C73 | established | 2.1337 / 1.9939 / 2.0127 / 1.5424 | `ARENA:runs/d5b/analysis-completed-1791577670/analysis.json#.games[*].metrics["0.01"][*].po` | matches | analysis.json best_q@8 means 2.1337 / 1.9939 / 2.0127 / 1.5424 |
| C74 | established | 0.1398 / 0.1211 / 0.5913 | `ARENA:runs/d5b/analysis-completed-1791577670/analysis.json#.games[*].metrics["0.01"][*].po` | matches | analysis.json regret means 0.1398 / 0.1211 / 0.5913 |
| C75 | established | dScore 0.03125 [-0.0625, 0.125]; dBestQ 0.01873 [-0.03204, 0.08107]; INCONCLUSIVE | `ARENA:runs/d5b/analysis-completed-1791577670/analysis.json#.paired` | matches | value found at the cited locator |
| C76 | established | 0.5056656 [0.4412160, 0.5684545] | `ARENA:runs/d5b/analysis-completed-1791577670/analysis.json#.games.rerank_T1_D5b.reviewed_s` | matches | value found at the cited locator |
| C77 | established | percentile, 2,000 draws, seed 7 | `REPO:deploy/dl-flywheel/pipelines/rlfw/arena/kpi.py:83-110` | matches | value found at the cited locator |
| C78 | established | 0.690 vs 0.785; [-0.133, -0.056] | `REPO:deploy/dl-flywheel/docs/evidence.md:7` | matches | matches evidence.md:7 and DL_wave4_full_comparison.json; pool now named (C178) |
| C79 | established | 26 / 27 | `LANE:r6/r6-dev-split-preregistration-v1.json:121-131` | matches | value found at the cited locator |
| C80 | established | 10,000; G=16; B=8; alpha 0.05; ~4,208; 5% | `LANE:prereg/e1-e2-preregistration-v1.json:13-44,63` | matches | value found at the cited locator |
| C81 | established | 3.75e-5 | `LANE:rulings/owner-askuserquestion-receipt-R7-R16-v1.json:7` | matches | value found at the cited locator |
| C82 | established | 72 h; 80,000 | `LANE:rulings/owner-askuserquestion-receipt-R7-R16-v1.json:4` | matches | value found at the cited locator |
| C83 | established | 1e-6; 0.8705178834; 2; 9 | `CAPTURE:copy/training/optimizer-control.json` | matches | optimizer-control.json: group rates 1e-06, best 0.8705178834, 9 reductions |
| C84 | preliminary | 2 x (128 cores + 1 RTX 3090) | `PLAN:CLAUDE_PLAN_v2_0.md:51` | matches | value found at the cited locator |
| C85 | preliminary | ~1,024/h vs ~2,000-2,200/h | `PLAN:CLAUDE_PLAN_v2_0.md:43` | matches | value found at the cited locator |
| C86 | preliminary | ~94% | `PLAN:CLAUDE_PLAN_v2_0.md:45` | matches | value found at the cited locator |
| C87 | preliminary | 32,768; 100,000; 256 MiB | `PLAN:CLAUDE_PLAN_v2_0.md:42` | matches | value found at the cited locator |
| C88 | preliminary | 8 CPU/8 GiB; 1 CPU/1 GiB; 15%; 16 GiB | `PLAN:CLAUDE_PLAN_v2_0.md:33,206,607` | matches | value found at the cited locator |
| C89 | preliminary | 60 s; 8 rows; 2,048 rows; 1,024 contexts; ~435 updates/epoch | `PLAN:CLAUDE_PLAN_v2_0.md:112,176,196-205` | matches | value found at the cited locator |
| C90 | preliminary | 5 s; 30 s; 8 keys | `PLAN:CLAUDE_PLAN_v2_0.md:206-214` | matches | value found at the cited locator |
| C91 | preliminary | 14 d; 7 d; 25 GB/day; 1 TB; 1.5M PUT/day; 80%; $25/month | `PLAN:CLAUDE_PLAN_v2_0.md:244-279` | matches | value found at the cited locator |
| C93 | preliminary | ~98.5% native; ~1,248 rows | `PLAN:CLAUDE_PLAN_v2_0.md:39,43,112` | matches | value found at the cited locator |
| C92 | preliminary | 2%; 3 epochs | `PLAN:CLAUDE_PLAN_v2_0.md:115` | matches | value found at the cited locator |
| C100 | pending | pending | `LANE:prereg/e1-e2-preregistration-v1.json:13-14` | matches | pending: no value exists |
| C101 | pending | pending | `LANE:prereg/e1-e2-preregistration-v1.json:16-20` | matches | pending: no value exists |
| C102 | pending | pending | `LANE:prereg/e1-e2-preregistration-v1.json:42` | matches | pending: no value exists |
| C103 | pending | pending | `LANE:prereg/e1-e2-preregistration-v1.json:33-42` | matches | pending: no value exists |
| C104 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:76 (definition V2)` | matches | pending: no value exists |
| C105 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:78,622 (definitions V4, L7)` | matches | pending: no value exists |
| C106 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:427` | matches | pending: no value exists |
| C107 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:428` | matches | pending: no value exists |
| C108 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:124,649` | matches | pending: no value exists |
| C109 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:256-260` | matches | pending: no value exists |
| C110 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:279,652` | matches | pending: no value exists |
| C111 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:116` | matches | pending: no value exists |
| C112 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:115` | matches | pending: no value exists |
| C113 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:229,623` | matches | pending: no value exists |
| C114 | pending | pending | `PLAN:CLAUDE_PLAN_v2_0.md:389` | matches | pending: no value exists |
| C115 | pending | pending | `ARENA:runs/d5b/analysis-completed-1791577670/inputs (baseline_T1 outbox and spec)` | no file (directory) | pending: no value exists |
| C116 | pending | pending | `LANE:r6/r6-dev-split-preregistration-v1.json:122-127` | matches | pending: no value exists |
| C117 | established | #292, 9e5baa28f, 2026-10-12 | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/runner_v6.py:1-56` | matches | value found at the cited locator |
| C118 | established | 230 = 83 + 55 + 36 + 32 + 24 | `MAIN:deploy/dl-flywheel/run_v2_offline.py:156-159` | matches | value found at the cited locator |
| C119 | established | L6 + 0.2 KL; 0.03 / 0.05 | `LANE:rulings/owner-askuserquestion-receipt-R23-v1.json` | matches | value found at the cited locator |
| C120 | established | lambda 0.2; 1 context/step; 16,384 tokens; T = 1.0 | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/anchored_step.py:1-30,45-47` | matches | value found at the cited locator |
| C121 | established | 1c0c050f...; bf02f873... | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/runner_v6.py:48-56` | matches | value found at the cited locator |
| C122 | established | KL = 0; \|g\| <= ~3e-17; bitwise | `LANE:v2build/candidate-r3/NOTES.md:97-105` | matches | value found at the cited locator |
| C123 | established | 1.02; 3 epochs; 0.8705178834 | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/runner_v6.py:31-36,168-176` | matches | value found at the cited locator |
| C124 | established | 64 probes; 0.03; 0.05; 0.15 | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/kl_guard.py:1-24,48-49,111-112` | matches | value found at the cited locator |
| C125 | established | per-epoch watermark; full prefix re-hash | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/custody_v6.py:1-22` | matches | value found at the cited locator |
| C126 | established | intent -> verification -> result | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/runner_v6.py:19-29` | matches | value found at the cited locator |
| C127 | established | envelope + candidate + witness | `MAIN:deploy/dl-flywheel/pipelines/native_result_overlay_v1/README.md:5,43-68` | matches | value found at the cited locator |
| C128 | established | at most 1 adoption per closed epoch | `MAIN:deploy/dl-flywheel/pipelines/boundary_admission_v2/boundary.py:1-30` | matches | value found at the cited locator |
| C129 | established | 5,712 / 1,270 | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/genesis.py:28-32,67-80` | matches | value found at the cited locator |
| C130 | established | 3 gates; 1e-6 -> 3.75e-5 | `HOST:v2.0.0-campaign-r3-20261011T011225Z/training/control-event-r14.json#.parity_gate` | matches | control-event-r14.json: rates 1e-6 -> 3.75e-5, three gates passed |
| C131 | established | 8/8 bitwise; loss 0.9540845 | `HOST:parity-20261011T011225Z/PARITY-CUDA.json` | matches | re-derived from the receipts (Section 10 pass) |
| C132 | established | 17,775 / 17,776; 0.47 s / 0.46 s | `LANE:v2build/real-r3/BUILD-QUALIFICATION-r3.json#.dry_runs,.not_claimed` | matches | re-derived from the receipts (Section 10 pass) |
| C133 | established | 19 of 21 | `LANE:v2build/real-r3/BUILD-QUALIFICATION-r3.json#.candidate_runtime_files` | matches | re-derived from the receipts (Section 10 pass) |
| C134 | established | 1 or 2; 1 in the build root | `MAIN:deploy/dl-flywheel/pipelines/continuous_v6/runner_v6.py:12-15` | matches | value found at the cited locator |
| C135 | established | non-qualifying | `LANE:rulings/owner-askuserquestion-receipt-R22-v1.json` | matches | re-derived from the receipts (Section 10 pass) |
| C136 | non-qualifying | 0.031 -> 0.053; 0.004-0.011 | `LANE:rlfw-takeover/lpi-scratch/x2/run/claude-rlfw-lpi-scratch-sweep4a-20261010T225200Z/rec` | matches | re-derived from the receipts (Section 10 pass) |
| C137 | non-qualifying | L6+0.2KL <= 0.014; L6 alone >= 0.052 by epoch 2 at 3.75e-5 | `COORD:lanes/CX-MET/daily-20261011/final/20261011T195459Z-000/inputs/ccdc9e175b57f647ee5bf6` | matches | corrected: sweep 4a L_PI 0.01 is 0.011-0.020 (0.011465 at epoch 1); sweep 6 L6 only is 0.031 at both epochs (0.031133, 0.031480) |
| C138 | non-qualifying | 7.080 -> 6.59-6.87 (L_PI) vs 7.07-7.21 | `COORD:lanes/CX-MET/daily-20261011/final/20261011T195459Z-000/inputs/ad010b1e4bef57c99ae26d` | matches | re-derived from the receipts (Section 10 pass) |
| C139 | non-qualifying | 0.8610 -> 0.8621-0.8660 | `COORD:lanes/CX-MET/daily-20261011/final/20261011T195459Z-000/inputs/ad010b1e4bef57c99ae26d` | matches | qualified: true at each arm's last evaluated epoch; two sweep 4b arms dipped below the start at epoch 1 |
| C140 | non-qualifying | [0.8963, 0.9022]; panel 0.8712 -> 0.8649; full 0.8973 -> 0.8977 | `COORD:lanes/CX-MET/daily-20261011/final/20261011T195459Z-000/inputs/ccdc9e175b57f647ee5bf6` | matches | corrected: lower bound 0.8963 (sweep 6 L6 + 0.2 KL, 0.8963494 at incomplete epoch 8), not 0.8964 |
| C141 | established | 0.8973272 | `LANE:rlfw-takeover/lpi-scratch/x2/run/claude-rlfw-lpi-scratch-sweep4a-20261010T225200Z/rec` | matches | 0.8973272291 in the sweep 4a receipt; epoch-40 record reconciled by C181 |
| C142 | established | all INCONCLUSIVE; 0 GREATER | `COORDSNAP:lanes/G-JUDGE/scratch-verdicts.md:28-31,122-146,1126-1446` | matches | re-derived from the receipts (Section 10 pass) |
| C143 | established | unjudged | `COORDSNAP:lanes/G-JUDGE/scratch-verdicts.md` | matches | re-derived from the receipts (Section 10 pass) |
| C144 | established | not a sample size; #302 | `COORDSNAP:lanes/G-JUDGE/scratch-verdicts.md:706-716` | matches | value found at the cited locator |
| C145 | established | not taken | `COORDSNAP:GAP_BOARD_20261012.md:13,62` | matches | value found at the cited locator |
| C146 | established | Discover foundation; Learn gated | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:6,23-28,1257` | matches | value found at the cited locator |
| C147 | established | H1-H4 | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:40-50` | matches | value found at the cited locator |
| C148 | established | claim rule | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_SUMMARY.md:23-27` | matches | value found at the cited locator |
| C149 | established | phases 0-5; #296 | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:780-938,1163-1180` | matches | value found at the cited locator |
| C150 | established | threshold does not carry over | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:521` | matches | value found at the cited locator |
| C151 | established | candidate | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:1079,1257` | matches | value found at the cited locator |
| C152 | established | L6 not a success metric | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_SUMMARY.md:66-68,108-109` | matches | value found at the cited locator |
| C153 | established | none established | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:1261` | matches | value found at the cited locator |
| C154 | pending | pending | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:808-814` | matches | pending: no value exists |
| C155 | pending | pending | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:836-842` | matches | pending: no value exists |
| C156 | pending | pending | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:864-870` | matches | pending: no value exists |
| C157 | pending | pending | `MAINGIT:docs/internal/clover/CLOVER_VNEXT_FINAL.md:892-898` | matches | pending: no value exists |
| C158 | pending | pending | `COORDSNAP:GAP_BOARD_20261012.md:62` | matches | pending: no value exists |
| C159 | established | not authorized | `COORDSNAP:GAP_BOARD_20261012.md:13` | matches | value found at the cited locator |
| C160 | pending | pending | `LANE:v2build/real-r3/BUILD-QUALIFICATION-r3.json#.not_claimed` | matches | pending: no value exists |
| C161 | established | 213/213, 0 skipped, two torch versions | `LANE:v2build/candidate-r3.1/DELIVERY.json#.tests` | matches | value found at the cited locator |
| C162 | established | not run | `LANE:v2pr-r3.1/PR_BODY.md:393,582` | matches | re-derived from the receipts (Section 10 pass) |
| C163 | non-qualifying | 6 arms below by 0.0000285-0.0009778; lowest 0.8963494 | `LANE:rlfw-takeover/lpi-scratch/x2/run/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/rece` | matches | re-derived from the receipts (Section 10 pass) |
| C164 | established | merged source + CPU fixtures | `MAIN307:deploy/dl-flywheel/pipelines/rlfw/onpolicy/INTERFACE.md:78-117` | matches | re-derived from the receipts (Section 10 pass) |
| C165 | established | merged source; one pre-registered arm | `MAIN307:deploy/dl-flywheel/pipelines/rlfw/r25_knnce/README.md:1-30` | matches | re-derived from the receipts (Section 10 pass) |
| C166 | non-qualifying | 10/2/5; 16/1/0; 8, 10, 11 of 17 | `MAIN307:docs/internal/clover/GP_EXPANSION_STUDY.md:222-236` | matches | re-derived from the receipts (Section 10 pass) |
| C167 | non-qualifying | 40/51 outside catalog; 48/51 bitwise parity | `MAIN307:docs/internal/clover/GP_EXPANSION_STUDY.md:239-245` | matches | re-derived from the receipts (Section 10 pass) |
| C168 | non-qualifying | 53 completed, 18 interrupted, 6 contexts incomplete; coordinator-only paired summary 17/24 contexts (complete-case) | `MAIN307:docs/internal/clover/GP_EXPANSION_STUDY.md:89-91,213-224,251-254` | matches | arm totals re-derived; paired coverage from pinned protocol/readout (complete-case, non-qualifying) |
| C169 | non-qualifying | d = +0.000124 [-0.00331, 0.00303]; KL 0.1079 | `COORD:lanes/G-JUDGE/r25-e1-verdict-de66cb6a.json` | matches | re-derived from the receipts (Section 10 pass) |
| C170 | non-qualifying | 6 arms INCONCLUSIVE; d from -0.00321 to +0.00399 | `COORD:lanes/G-JUDGE/j21-paired-verdict-98fa563d.json` | matches | j21 verdict: epochs at 352 / 704 / 1,056 / 1,408 updates added |
| C171 | non-qualifying | E4 0.8951989 (telemetry) | `COORD:lanes/CX-OP/r26-seedrep/TRAIN_EXEC_EXIT.json#.epochs` | matches | re-derived from the receipts (Section 10 pass) |
| C172 | non-qualifying | E4 0.8951011 (telemetry); KL 0.0377 | `LANE:gb11/sources/r27/receipt.json#.epochs` | matches | re-derived from the receipts (Section 10 pass) |
| C173 | established | n = 1,148; 221 clusters; K = 3 | `COORD:lanes/G-JUDGE/l6-1148-prereg.json` | matches | corrected scope: sweeps 3 and 4a had no holdout |
| C174 | non-qualifying | +0.00511, +0.00310, +0.00333; REGRESSION ALERT x3 | `COORDSNAP:lanes/G-JUDGE/l6-1148-verdict.json#.arms` | matches | re-derived from the receipts (Section 10 pass) |
| C175 | preliminary | 48-context roster, disjoint = true | `LANE:ei-1/EI1_DISJOINTNESS.json` | matches | re-derived from the receipts (Section 10 pass) |
| C176 | non-qualifying | -0.000978 [-0.00446, 0.00250]; -0.000315 [-0.00365, 0.00302] | `COORD:lanes/G-JUDGE/sweep6-verdict-8b7aaead.json#.arms` | matches | re-derived from the receipts (Section 10 pass) |
| C177 | non-qualifying | validation about 0; held-out lineages +0.0042 to +0.0063 | `COORDSNAP:lanes/G-JUDGE/l6-1148-verdict.json#.arms[].stratum_a,.arms[].stratum_h` | matches | scope: the three judged fine-tunes excluded the 885 |
| C178 | established | row R2 0.652 vs 0.752; equal-lineage R2 0.690 vs 0.785; beat10 AUC 0.901 vs 0.903 | `PROBE:ofm-20261008/dashboard_codex/DL_wave4_full_comparison.json#.ensemble_delta_vs_lgb` | matches | new: DL_wave4_full_comparison.json .ensemble_delta_vs_lgb |
| C179 | established | 0.718 / 0.760 / 0.745; AUC 0.903 / 0.904 / 0.903 | `PROBE:ofm-20261008/figs/dl_waves/confirm_report.json#.lgb` | matches | new: confirm_report.json .lgb 0.7176 / 0.7603 / 0.7450, AUC 0.9031 / 0.9038 / 0.9031 |
| C180 | established | one holdout: sealed families | `REPO:deploy/dl-flywheel/pipelines/continuous/native_corpus_authority.py:25,34-41` | matches | new: native_corpus_authority.py:25 requires jope.dltrain-one-holdout-role-map/v1 |
| C181 | established | 17,416 vs 17,773; 357 updates | `CAPTURE:copy/training/optimizer-history.jsonl (rows with epoch 40 and 41)` | matches | new: epoch 40 = batches 16,980-17,416; epoch 41 = 17,417-17,773 (357); continuous_runner.py:164-168 validates on epoch completion |
| C182 | established | 4 runs; 5 judge records | `LANE:rlfw-takeover/lpi-scratch/x2/run/claude-rlfw-lpi-scratch-r25control-20261013T083136Z/` | matches | new: receipts r25control, R25, R26, R27; five G-JUDGE verdict files dated 2026-10-13 |
| C183 | established | 6,888 per checkpoint; 4 relative-only failures (feas, 1.3e-8) | `COORD:lanes/G-JUDGE/l6-1148-prereg-amendment-abs.json` | matches | new: amendment rule A at 11:20:58Z; selfcheck relative_failure_count 4, all feas |
| C184 | established | PR #343 (`c402b21183670d3e4df73b835292a385a04db30f`); 11 tests; 4 rounds, B=8, 2,000 draws, alpha <= 0.025, KL <= 0.20, mass <= 0.15; UNPINNED, 0 GPU s | `MAIN343:docs/internal/clover/selfplay-round1-prereg.v1.json` | matches | new: SPQ-1 prereg and round_qual.rs / selfplay_round_qual.rs |
| C185 | non-qualifying | 24 runs, 96 slots (0 unresolved); P_EI0 +0.000004 [-0.000362, +0.000389] INCONCLUSIVE; P_EI0r Stratum H +0.000596 [+0.000115, +0.001087]; dose 0.01 s30-31 H +0.000338, pooled +0.000110; descriptive 0.03/0.10 s28-29 pooled -0.000416 / -0.000377 | `EVIDENCE:rlfw-g-abl/ei0-fate-96-20261014.json` | matches | new: ei0-fate-96-20261014.json + l6-1148-pei0.json, l6-1148-pei0r.json, l6-1148-ei0-dose-descriptive.json |
| C186 | non-qualifying | ei1_arm frozen at ep 3 (1,056 updates, KL 0.079634 > 0.05); PRIMARY_NOT_COMPUTED (`do_not_open_paired_primary = true`, r69e3/r70e3 unopened); in-run ep-3 receipt delta +0.004297 (H +0.003565, A +0.006758) | `EVIDENCE:jopedime-flywheel-ei1/v8-run/EI1_RUN_RECEIPT.json` | matches | new: EI1_RUN_RECEIPT.json, claude-ei1-v8-armctrl-20261014T191054Z/receipt.json and G-JUDGE klstop rulings |
| C187 | non-qualifying | 3 hosts x 1,408 updates, 0 KL stops (xbabe2 max probe KL 0.022950 at ep 4); 3-seed L6-1148 mean +0.000378 [+0.000021, +0.000744] (H +0.000407, A +0.000281) | `EVIDENCE:ei3-study-20261015/EI2_EI2X_3HOST_SUMMARY.json` | matches | new: EI2_EI2X_3HOST_SUMMARY.json and xbabe2 EI2_RUN_RECEIPT.json / receipt.json |
| C188 | non-qualifying | xbabe2 frozen at ep 3 (1,056 updates, kl_stop; KL 0.039816 ep2, 0.080342 ep3); 3-host paired_k_star_e3 L6-1148 mean +0.003962 [+0.003222, +0.004744] (H +0.003969, A +0.003940) | `EVIDENCE:ei3-study-20261015/EI3_3HOST_SUMMARY.json` | matches | new: EI3_3HOST_SUMMARY.json and xbabe2 receipt.json |
| C189 | non-qualifying | 1,408 updates, 0 KL alarms; in-run receipt delta +0.000720 at ep 3 (H +0.000378, A +0.001870), +0.000082 at ep 4 (H +0.000051, A +0.000187); CPU row-file delta +0.000518 ep 3, +0.000106 ep 4 | `EVIDENCE:ei4-study-20261015/xbabe2-run/ei4-xbabe2-armctrl-20261015/receipt.json` | matches | new: ei4-xbabe2-armctrl-20261015/receipt.json (49.5 KiB) |
| C190 | non-qualifying | 8,908 S3 rows; shadow-pool-v2 (1,136 contexts, 255 clusters) R0->R4 pairwise 0.634318 -> 0.680342 (+0.046025 [+0.039928, +0.051692] GREATER), B=1 gain 0.785168 -> 0.919919 (+0.134751 [+0.067497, +0.215110] GREATER), ESTABLISHED_SHADOW, flywheel_established = false | `EVIDENCE:selfplay-s3-sos-debiased-flywheel-20261016/gp-results-index-s3-receipt.json` | matches | new: gp-results-index-s3-receipt.json and bounded PR #339 r3 review extracts (`scratch-r3-extract.txt`, `scratch-extract-rounds.txt`, `VERDICT-source.md`) |

## Exempt numbers

Numbers the write-up prints that are illustrative arithmetic or a definition rather than a
measurement. `paper-trace` accepts the listed numbers on any line containing the text, and fails if
the text no longer occurs in the write-up.

| Text | Numbers | Why |
| --- | --- | --- |
| `$q = \log 10$ means ten times lower` | 10 | reading guide to the definition of $q$ |
| `\log 91` | 91 | $\log(1 + 90)$: the 90 s timeout of C12 on the $\log(1 + t)$ scale |
| `0.5 by construction` | 0.5 | the teacher's tie-adjusted score against itself is 1/2 under the scoring rule of C72 |
| `If $r = (0.5, 0.5)$ and $b = (1, 1)$` | 0.5, 0.693 | Appendix A worked numbers, illustrative arithmetic |
| `A program ten times better than the mean predictor on both folds` | 10, 2.303 | Appendix A worked numbers, illustrative arithmetic |
| `The clip at $\log 1000 \approx 6.908$` | 6.908 | Appendix A worked numbers, illustrative arithmetic |
