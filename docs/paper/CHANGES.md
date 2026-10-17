# Changes to the CLOVER write-up: 2026-10-10 draft to 2026-10-13 revision (gap GB-11)

## Inputs and outputs

The previous draft, the LaTeX scaffold (`tex/`), the paper tools (`tools/`) and the internal companion are lane files in `.agent/proposals/claude-rlfw-20261009/paper/` on xbabe2. They are not in this repository.

| | Previous draft | This revision |
|---|---|---|
| Write-up | `paper/TECHNICAL_WRITEUP.md`, sha256 `f41abdffb9502a977421d7d5abf394f79d25ae00da740dc19b571ceb97fdedf4` | `docs/paper/TECHNICAL_WRITEUP.md` |
| Claim ledger | `paper/claims.json`, sha256 `429f72ff7ff4a0f0e3c5e48ed320108c3aadc15597492f7802bba75f2ee6fa09`, 110 claims | `docs/paper/claims.json`, 157 claims |

## Why the draft changed

Gap GB-11 records that the paper and the model description were stale. Its acceptance test requires that the documents reflect the merged v2 learner, the R23 objective and the CLOVER direction.

The 2026-10-10 draft was written before four events:

- **R22, 2026-10-10.** The owner ruling permitted scratch-only training. It produced sweeps 3 to 6, all labelled non-qualifying.
- **R23, 2026-10-11.** The v2 training objective became L6 + 0.2·KL to the frozen v1 policy, with a KL guard.
- **Merge of the v2 learner, PR #292, 2026-10-12.** The merge carried the caller-completion work of PR #293 in the same squash.
- **The CLOVER direction, PR #295, 2026-10-12.** It splits the system into CLOVER-Discover and CLOVER-Learn and states hypotheses H1 to H4.

The old draft described v2 as an L6-only loop whose feedback effect was pending. It also treated that loop as the whole of CLOVER.

## Framing and claims, repositioned

- **CLOVER is now two capabilities.**
  - CLOVER-Discover, a qualified program-discovery engine, is the foundation.
  - CLOVER-Learn, a learned controller, is accepted only on controlled evidence of added value.
  - The v1/v2 policy and its loop are presented as the first CLOVER-Learn candidate, not as the system. [C146, C151]
- **Title.** Option 1 now follows the direction document: "CLOVER: Qualified Program Discovery with Learned Evaluation Allocation". The old acronym expansion, "Continual Learning from Outcome-VERified execution", no longer appears in the title. Choosing a final title is left to the owner.
- **Headline claims.**
  - Old: "the loop exists, and its feedback effect is pre-registered and pending".
  - New:
    - the learning candidate and its integrity contracts exist as merged source, with bitwise build qualification;
    - scratch evidence is non-qualifying;
    - H1 to H4 are pending;
    - no objective has been shown to beat frozen v1 under qualified evaluation.
  - A new Section 1.4 lists what the report does not claim. It quotes the CLOVER rule that no self-improving claim is made until fresh feedback causally improves independently confirmed selection beyond the strongest non-learning baseline. [C142, C148, C153–C158]
- **Contributions.** These are new:
  - the anchored objective and the separation of its two identities;
  - the two sticky guards;
  - caller-written overlay acceptance with a return witness;
  - verified-watermark custody with an ordered closure;
  - an evidence ledger that keeps qualified, non-qualifying and pending results apart.

  Three old contributions are no longer listed as contributions:
  - **"Bounded closure, measured"** is folded into the custody contribution.
  - **"A tiered custody policy"** is now explicitly a proposal, because no deletion is authorized.
  - **"A pre-registered evaluation protocol"** is folded into the evidence ledger.

## Section by section

| Section | Change |
|---|---|
| Header | Adds the `non-qualifying` status to the vocabulary. Notes that `tex/` has not been regenerated. |
| Abstract | Rewritten around the CLOVER split, the merged learner, the build qualification, the non-qualifying scratch results and the pending hypotheses. |
| Teaser | The panels are tied to H3/E1, H2/E2 and cost. They are filled only from qualified receipts. |
| 1 | New 1.1 (CLOVER in two parts), 1.2 (the first CLOVER-Learn candidate) and 1.4 (what is not claimed). Contributions revised as above. |
| 2.2, 2.6 | Expansion beyond the catalog is assigned to CLOVER-Discover and H1. States that L6 is not a success metric [C152]. |
| 3.3 | Notes that the prefix tree makes the catalog KL exact. |
| 4 | Retitled "Objectives". 4.3 now lists the three R14 gates [C130]. 4.4 (the R23 anchored objective) and 4.5 (training identity versus label identity) are new. The old sentence placing the guard at "2% above v1's final value" is corrected and moved to 6.6. |
| 6.2 | v2 emits 1 or 2 proposals per update, and 1 in the build-qualification root [C134]. |
| 6.3 | Adds the operator acceptance envelope for prepared candidates and the recheck of the authority proof inside the compare-and-swap. Lane fixture counts move to 10.1.6. |
| 6.4 | Adds caller-written acceptance, the publication candidate and the return witness [C127]. |
| 6.6 | New: the regression guard (corrected reference [C92] → [C123]) and the KL guard [C124], including the CLOVER note that the threshold does not carry over to expanded support [C150]. |
| 6.7 | Extends the old 6.6 with the verified watermark, prefix integrity and the ordered terminal records [C125, C126]. |
| 6.8 | Adds the eight next-update parity receipts [C131], the CPU gap [C162] and the fresh v2 schedule [C129]. Removes the old sentence saying both E1 arms see the same batch order, which no longer has a merged source. |
| 6.9 | Restores the broker and learner-side timing figures [C54, C55]. Adds that the anchored step's GPU cost is unmeasured [C160]. |
| 7.2 | The retention table is now labelled a proposal. "Deleted with receipts" is replaced by "would require separate owner authorization" [C159]. |
| 9.2, 9.3 | Label E1 as an H3 test and E2 as an H2-adjacent test. Note that the shadow control learner is lane source and not merged [C68]. |
| 9.6, 9.7 | New: the CLOVER gate table (Phases 0 to 5) [C149], and the scratch judge's sample-size rule with its power-unit caveat [C142, C144]. |
| 10.1.6 | Reports the main fixture roster (230) [C118] and the 213/213 result under two torch versions [C161]. Lane fault-injection figures are labelled lane receipts, and unmerged lane packages are named. |
| 10.2 | New: v2 build qualification, including partial coverage of the merged bytes [C133]. |
| 10.3 | New: R22 sweeps 3 to 6, non-qualifying, with three directional readings [C135–C143]. |
| 10.4 | Adds C154–C158 and C160 to the pending table. |
| 11 | AutoML is reframed around Discover and Learn. A new paragraph on KL-regularized updates cites Schulman et al. (2015) and Ziegler et al. (2019). The continual-learning paragraph contrasts parameter anchoring with output anchoring. |
| 12 | Adds limitations: no hypothesis result; non-qualifying scratch; no qualified sample size; two values for frozen v1 L6; catalog-bound anchor; observed-only at-anchor equivalence; partial build coverage; retention is only a proposal. |
| 13 | Reorganized by CLOVER phase. Expansion beyond the catalog becomes Phase 1. The measured-tilt challenger competes only under Phase 2 or 3 rules. |
| Appendices | B adds the v2 anchor, guards and emission rows. C adds 11 merged schema names. E adds the scratch judge. F lists the newly pinned sources. G has new counts. H adds 9 glossary terms. |

## Claim ledger (`claims.json`)

- **New status.** `non-qualifying`, for real measurements from owner-labelled scratch runs.
- **New source roots:**
  - `MAIN`: a byte copy of `deploy/dl-flywheel` at `origin/main` `e466d1a1`;
  - `MAINGIT`: byte copies of the three CLOVER documents at the same commit, in the lane directory `gb11/sources/`;
  - `MAIN302`: a byte copy of the power table corrected by PR #302 (`a2f7e6ecd`, merged 2026-10-13 while this revision was being written), cited by C144;
  - `COORD`: the coordination board and the scratch-judge record;
  - `HOST`: the v2 build-qualification roots and parity receipts.
- **47 new claims, C117–C163:**
  - 35 established;
  - 6 non-qualifying (C136–C140, C163);
  - 6 pending (C154–C158, C160).
- **Updated claims:**
  - C10: v1 only, with v2 pointing to C134.
  - C68: lane-only note added.
  - C81: merged source added.
  - C91: proposal, no deletion authorized.
  - C92: corrected reference, `superseded_by` C123.
- **Unchanged.** No claim was removed. No pending claim was filled or estimated. Totals: 157 claims (109 established, 19 preliminary, 6 non-qualifying, 23 pending).
- **Check.** `paper/tools/check_claims.sh` passes on the revised write-up and ledger, together with the unchanged `tex/`: 157 cited in the write-up, 155 hashes checked, 2 unresolvable.

## Not changed in this revision

- **`tex/`** still mirrors the 2026-10-10 draft. It still contains the old guard wording, "deleted with receipts", and no R23 or CLOVER material. It must be regenerated with `tools/gen_claim_macros.sh` and edited before any PDF build. `refs.bib` needs entries for Schulman et al. (2015) and Ziegler et al. (2019).
- **`IP_SCRUB_REPORT.md`** is superseded by the scrub result recorded in this change's pull request.
- **`internal/INTERNAL_COMPANION.md`** is unchanged.
- **Wording.** The public wording rule (R15) is kept. CLOVER-Discover is described as "a conventional offline search" and H1/H4 are paraphrased accordingly; the internal CLOVER documents use the restricted method name.

## Update of 2026-10-13, 11:30Z (still gap GB-11)

The owner asked for the write-up and the model description to be refreshed to R23, R25, R26, R27 and the CLOVER direction, with non-qualifying wording throughout.

**Write-up changes:**
- **New Section 10.3.1**, covering:
  - the merges #304, #306 and #307;
  - the paired looks against frozen v1 for sweep 6, the matched control (four epochs) and R25;
  - the R26 and R27 telemetry, which has no verdict;
  - the 1,148-context evaluation set, which is pre-registered and not yet scored;
  - the closed H1 pilot;
  - the planned first expert-iteration round.

  Every number is non-qualifying, and every look is INCONCLUSIVE.
- **Section 9.7** points to the 1,148-context set.
- **Section 10.3** no longer says sweep 6 is unjudged.
- **Section 10.4** gains C174.
- **Wording** follows R15. The pilot arms are described as start conditions for the offline search: nothing, frozen-policy draws, or the context's own verified best program.

**Ledger changes:**
- **13 new claims, C164–C176:**
  - 3 established (C164, C165, C173);
  - 8 non-qualifying (C166–C172, C176);
  - 1 pending (C174);
  - 1 preliminary (C175, a plan).
- **Updated claims:**
  - C143 now says sweep 6 was judged at 04:47Z.
  - C170 records both interval types.
- **Re-pinned claims:** C142–C145, C158 and C159. Their live coordination sources changed after drafting, so they now cite pinned copies under the new root `COORDSNAP`:
  - the judge record's first 1,446 lines, an append-only prefix whose hash is the original pin;
  - the gap board as edited at 10:17:15Z, with the cited line 59 now at line 62.
- **New root `MAIN307`:** byte copies from `origin/main` `0ed1a381`.
- **Totals:** 170 claims (112 established, 20 preliminary, 14 non-qualifying, 24 pending).

**Model description changes** (`deploy/dl-flywheel/docs/model.md`):
- a new section, "Evidence status: 2026-10-13 scratch results (NON-QUALIFYING)";
- an H1 pilot and EI-1 paragraph in "CLOVER direction";
- the same sweep 6 correction.

`SOURCE_MANIFEST.json` re-pins its row.

## Update of 2026-10-13, 12:00Z: the L6-1148 verdict

The judge scored the pre-registered 1,148-context set at 11:38:25Z.

- **C174** was pending and is now non-qualifying, with its named experiment's result: REGRESSION ALERT for all three checkpoints, with pooled differences of +0.00511, +0.00310 and +0.00333.
- **New C177** gives the strata, which are reported but do not decide. Validation is near zero; the held-out lineages show +0.0042 to +0.0063.
- **Write-up sections changed:**
  - Section 9.7 states the outcome.
  - Section 10.3.1 adds the verdict table and the forgetting reading. The EI-1 paragraph gains the proposed matched-control primary.
  - Section 10.4 drops C174.
- **model.md** gains the verdict table and the EI-1 amendment note.
- **Branch:** rebased onto `origin/main` `2140b4bd6` (#309). That base changed other SOURCE_MANIFEST rows; only the model.md row is re-pinned here.
- **Totals:** 171 claims (112 established, 20 preliminary, 16 non-qualifying, 23 pending).

## Update of 2026-10-13, 14:55Z: review corrections (jopedime_clover CHANGES on a5ce00af)

- **Appendix G** now gives the current ledger totals: 171 claims (112 established, 20 preliminary, 16 non-qualifying, 23 pending). It previously said 157 / 109 / 19 / 6 / 23. The earlier sections of this file keep the totals each revision had at the time.
- **C163, Section 10.3 and model.md.** The six-arm statement had said none of the six arms below frozen v1 has a paired verdict. Now it says:
  - the two sweep 6 arms at epoch 8 were judged against frozen v1 at 04:47Z, at exactly these checkpoints, and both are INCONCLUSIVE (C176);
  - the other four have no paired verdict.

  The values and the conclusion (no qualified improvement) are unchanged.

## Update of 2026-10-16: source audit (screen jopedime_paper)

Every claim was traced back to its source on xbabe2. [`NUMBERS_TRACE.md`](NUMBERS_TRACE.md) records the method, each claim's pin and value check, and the corrections. In summary:

- **Source pins.** 175 of 177 claims re-hash to their pin. C24 is derived, and C115 is pending and names a directory. Three roots (CAPTURE, ARENA, HOST) had moved under `/home/ubuntu/jope-evidence/` on 2026-10-15; `claims.json` now records the new paths. The REPO root is written as a path, and the new roots PROBE and CANON are added.
- **Corrected values:**
  - C140: the range of last evaluated epochs is 0.8963 to 0.9022, not 0.8964;
  - C137 and the Section 10.3 table: sweep 4a L_PI 0.01 is 0.011–0.020, and sweep 6 L6 only is 0.031 at both epochs;
  - C93 and Section 6.4: about 98.5% of proposals are on native contexts, the recorded value, not 98%;
  - the `tex` of C140 and C93 now prints its own record, and C168's statement and value now hold the 17 of 24 planned contexts its `tex` prints.
- **Corrected scope or wording:**
  - C139, C141, C170, C173 and C177;
  - Sections 9.7, 10.3 and 10.3.1: the 885 held-out contexts were not held out of sweeps 3 and 4a; the L6-1148 look is a regression alert, not a finding that "the fine-tune forgot"; and "no paired verdict" means against frozen v1.
- **New established claims, C178–C183:**
  - C178 and C179: the LightGBM comparator's pool and metrics (row R² 0.752 and beat10 AUC 0.903 on the corrected pool; R² 0.718 and AUC 0.903 at repeat 0 on the original pool);
  - C180: the one-holdout ruling;
  - C181: the two frozen v1 L6 values come from steps 17,416 and 17,773;
  - C182: four post-sweep runs and five judge records;
  - C183: the consistency check's pre-look amendment.
- **Scope.** The header states that the report covers events up to 2026-10-13T12:00Z. EI-1 is marked as planned "as of this revision".
- **Totals.** 177 claims (118 established, 20 preliminary, 16 non-qualifying, 23 pending).
- **Check.** `jope-records paper-trace` now runs in CI. It checks every printed number against the statements and values of the claims cited in its line or section, each claim's `tex` against its own statement and value, the Appendix G totals, the Section 10.4 pending table, and this file's claim trace and exemption tables.
- **Figure 1 on page 1.** It shows why the work matters, using only established measurements of the frozen first generation: (a) the full validation plateau [C26, C27], (b) the outcome head's ranking accuracy on fresh programs at chance [C76]. The planned teaser stays a specification: its panels need qualified receipts that do not exist [C114]. `paper-trace` renders the figure from `figures/v1_full_validation.dat`, the file C27 pins, and from C76's value, and fails when the committed SVG, its data or those claims disagree.

`tex/` is still not regenerated.

## Update of 2026-10-16 (addendum): post-C183 studies C184–C190 and Figure 1 panel (b)

- **New source roots `EVIDENCE` and `MAIN343`:** `/home/ubuntu/jope-evidence` on xbabe2 and `/home/ubuntu/jope-evidence/pr347-repair-20261017/main-c402b211` (commit-pinned `git show c402b21183670d3e4df73b835292a385a04db30f:<path>` byte copies), declared in `claims.json` (`source_conventions.roots`) so `jope-records paper-trace --sources` re-hashes the post-C183 host receipts and PR #343 files without mutable worktree dependence (182 of 184 claims re-hashed to their pin, plus 2 that name no single file).
- **7 new claims, C184–C190:**
  - 1 established (`C184`: `SPQ-1` offline round-qualification verifier `jope-pools qualify-rounds` and unpinned `selfplay-round1-prereg.v1.json` in PR #343, `c402b21183670d3e4df73b835292a385a04db30f`);
  - 6 non-qualifying (`C185`: `EI-0` 24-run / 96-slot continuation dose probes under $\lambda_{\mathrm{KL}} = 3.0$; `C186`: `EI-1` v8 archive-CE `kl_stop` at epoch 3, `PRIMARY_NOT_COMPUTED` with `do_not_open_paired_primary = true`, and bounded in-run epoch-3 receipt differences; `C187`: `EI-2`/`EI-2X` 3-host $\lambda_{\mathrm{ar}} = 0.03$ panel; `C188`: `EI-3` 3-host $\lambda_{\pi} = 0.05, \lambda_{\mathrm{ar}} = 0.10, m = 8$ `kl_stop` panel; `C189`: `EI-4` $\lambda_{\mathrm{KL}} = 5.0, \lambda_{\mathrm{ar}} = 0.025, m = 8$ run with `config.body` template-field disclosure and bounded in-run receipt differences; `C190`: `PR #339` S3 SoS-debiased contextual surrogate 5-round shadow-pool flywheel traced via `gp-results-index-s3-receipt.json` and bounded `< 21 KiB` `r3` review extracts).
- **Write-up changes:**
  - **New Section 10.3.2** summarizes `C184`–`C190` with an explicit sign convention (`arm minus matched control` on L6 loss vs. positive surrogate gain/ranking lift), non-qualifying and shadow-only scope preserved, and `< 512 KiB` source provenance noted.
  - **Figure 1 panel (b)** and `crates/jope-records/src/paper_figure.rs` plot both the v1 outcome head on freshly executed programs (`0.506` with 95% lineage level interval `[0.441, 0.568]`, 32 lineages, established) [C76] and, across a dashed vertical separator, the S3 SoS-debiased contextual surrogate on frozen `shadow-pool-v2` candidate pools (`R0 0.634 -> R4 0.680`, paired lift `+0.046` with 95% parent-cluster bootstrap lift interval `[+0.040, +0.052]`, 255 parent clusters, non-qualifying shadow validation) [C190].
  - **Appendix G** updates the claim ledger totals to **184 claims: 119 established, 20 preliminary, 22 non-qualifying and 23 pending**.
- **Experiment ledger (`docs/experiments/experiments.jsonl` and `docs/experiments/README.md`):** binds `EXP-012`–`EXP-018` to `C184`–`C190`, matches exact floating-point metrics in `EXP-013`–`EXP-017` to their source receipts, records verified `< 512 KiB` receipt/extract digests on `EXP-014`–`EXP-018` (with `EXP-018`'s 2.4–2.5 MB outer flywheel receipt file hashes explicitly marked owner-reported/unverified under the 512 KiB cap), and reconciles `EXP-017` and `EXP-018` in `docs/experiments/README.md` to the bounded receipts and `r3` review extracts.

## Update of 2026-10-17: C191–C192 (`EXP-NEXT-04` parity replay and `SPQ-1` `--prereg` extension)

- **New source root `MAIN351`:** `/home/ubuntu/jope-evidence/pr351-exp008-parity-replay-20261017/main-5be5ac4d` (commit-pinned `git show 5be5ac4dc9d5c0494aeb2964e9a755ff3e022dff:<path>` byte copies), declared in `claims.json` (`source_conventions.roots`) so `jope-records paper-trace --sources` re-hashes the PR #349 and PR #351 files without mutable worktree dependence (184 of 186 claims re-hashed to their pin, plus 2 that name no single file).
- **2 new established claims, C191–C192:**
  - `C191`: opt-in `EmitOptions::partition_point_quantile_bounds` emitter (PR #346, `c509ba0304f4fbdf61da3b97831fc5877b7421fc`, verified across Rust, Python, `C++17`, JavaScript and bound Python on a synthetic 24x3 1-ULP tie-plateau fixture) and offline evidence-only pilot champion parity replay (PR #351, `5be5ac4dc9d5c0494aeb2964e9a755ff3e022dff`, 5 tests in `crates/feat-math/tests/clover_quantile.rs`, receipt `docs/internal/clover/exp-008-quantile-parity-replay.20261017.json`, raising emitted Rust inner-cut parity on the 3 failing pilot champion programs from 3/6 to 6/6 cuts at 0.0 maximum absolute prediction delta while keeping `EmitOptions::default()` and R5 parity charging unchanged).
  - `C192`: `SPQ-1` prereg-loading and secondary-endpoint verifier extension (PR #349, `797f2863a10e9e4252b1fc3000f07fd75f055e69`, 13 dedicated SPQ-1 unit and integration tests: 2 unit in `round_qual.rs` and 11 integration in `selfplay_round_qual.rs`, plus 12 unit tests in `expansion.rs`), adding `SelfPlayRoundReceipt::ordinary_champion_genome` (`run_selfplay_ratchet`), `SelfPlayPrereg`, `UnpinnedOr<T>`, `qualify_rounds_with_prereg`, `jope_pools qualify-rounds --prereg <PATH>`, secondary endpoint `ratchet_vs_ordinary_final_round` (`Endpoint::RatchetVsOrdinaryFinalRound`) alongside `final_vs_round0_incumbent`, and fail-closed unpinned-sentinel cross-checks while keeping `qualified_claim`, `promotion` and `l6_claim` `false` with 0 GPU seconds authorized.
- **Write-up changes:**
  - **Sections 5.5, 10.3.1, 10.3.2 and 12** cite `C191` and `C192`.
  - **Appendix C** adds `jope.selfplay-round-qualification/v1` and `jope.selfplay-round1-prereg/v1`.
  - **Appendix G** updates the claim ledger totals to **186 claims: 121 established, 20 preliminary, 22 non-qualifying and 23 pending**.
- **Experiment ledger (`docs/experiments/experiments.jsonl`, `docs/experiments/README.md`, `docs/experiments/SYNTHESIS_AND_NEXT_EXPERIMENTS.md`):** binds `C191` to `EXP-004` and `EXP-008`, and updates `EXP-012` for PR #349 (`C192`, 13 dedicated SPQ-1 tests + 12 unit tests in `expansion.rs`, `--prereg`, `ordinary_champion_genome`, `ratchet_vs_ordinary_final_round`).

## Update of 2026-10-17 (figures): README overview and Section 10.3.2 Figures 2 and 3

- **Three deterministic SVG figures in `docs/paper/figures/`** (rendered from committed repository inputs with zero hardcoded coordinates; custody receipt `/home/ubuntu/jope-evidence/pr353-why-it-matters-plots-20261017/why_it_matters_plots_receipt.json`):
  - `docs/paper/figures/readme-clover-overview.svg` (SHA-256 `9128edc49717eb138f2b49f45ecc25ec00c61fe91b5358ff1b631bfa057367f7`): embedded in `README.md` under `## Measured results` (outside the generated `<!-- arena/samecut/bench -->` markers), summarizing (a) the same-cut `test-10` deep ladder (`assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv`) and (b) `v1` fresh-program `q`-hat pairwise accuracy (`0.506 [0.441, 0.568]`, established `C76`) vs. the offline S3 SoS-debiased contextual surrogate on `shadow-pool-v2` (`R0 0.634 -> R4 0.680` pairwise, `0.785 -> 0.920` `B=1` gain, non-qualifying `C190`).
  - `docs/paper/figures/clover-why-it-matters-3panel.svg` (SHA-256 `04590ecc014a7856f1a97751cb6d3e7115b4efc6886bcc45c7f886503f42669d`): embedded as **Figure 2** in `TECHNICAL_WRITEUP.md` Section 10.3.2, plotting (a) the `v1` 31-epoch validation plateau `[0.8968, 0.8978]` (`C26, C27`), (b) continuation probe KL vs. paired `L6-1148` loss difference across `R25`, `R26`, and `EI-1`–`EI-4` (`C174, C186–C189`), and (c) `v1` fresh `q`-hat (`C76`) vs. the S3 SoS-debiased surrogate (`C190`).
  - `docs/paper/figures/parity-census-and-exp008-replay.svg` (SHA-256 `d5195b56e8b03e3718355e2f58e472c037cc8bbe2b7c988c1dcb33cad6c04f84`): embedded as **Figure 3** in `TECHNICAL_WRITEUP.md` Section 10.3.2, plotting (a) the 1,554-pair quantile-binning parity census (`41/342 = 12.0% [8.7, 15.9%]` with binning vs. `0/1,212` without binning, `C41–C43, C49`, and `48/51` `EXP-008` default-emit pass rate, `C167`) and (b) the offline `EXP-008` 3-champion replay (`3/6 -> 6/6` inner cuts passing at `0.0` maximum absolute delta under opt-in `EmitOptions::partition_point_quantile_bounds`, `C191`).
- **Header and snapshot sync:** updates `TECHNICAL_WRITEUP.md` line 3 (`2026-10-17 post-C183 addendum covering C184–C192`) and syncs `catalog_snapshot_commit` across all 18 records in `docs/experiments/experiments.jsonl` and `docs/experiments/README.md` to `d14c06a9123d723256f4f0784a1cb5c58ebc2182`.
