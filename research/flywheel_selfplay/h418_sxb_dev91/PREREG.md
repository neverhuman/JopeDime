# H418 — the one `dev91-audit-v1` score of the locked H416 entrant `h416.SXB`

- **Owner:** `cl_fwriter`, Claude Opus 5.5. Claim: `ns/receipts/H418_claim.txt`.
- **Authority.** Jepson typed in session 624cd2df at 2026-11-10T09:26Z: "confirm: score h416.SXB once on
  dev91-audit-v1". This confirms the relayed approval `~/nudges/H416_SXB_dev91_score_APPROVED.md` (ENG:watcher,
  2026-11-09 5:31 PM MST): one score, no rerun, no other entrant rescored.
- **Gate.** Jepson's approval at 4:09 AM MST (2026-11-10T11:09Z): run the one score after PRs 2180 and 2181 have both
  merged, not before. `run_h418.sh unlock` refuses unless every PR in the gate file
  (`~/jope_plan/evidence/H418/gate_prs.txt`) is merged. The file is append-only and lists 2178, 2179, 2180 and 2181
  (earlier messages named 2178–2180), so it can only get stricter. At 2026-11-10T12:18Z, PRs 2178, 2179 and 2180 were
  merged and PR 2181 was open.
- **Commit order.** This PREREG and every H418 file are committed to `origin/v1/next` before the LOCK. The run stages
  the published bytes (`stage_h418.sh pub`), and every step re-checks them against `origin/v1/next`. Nothing is
  chosen on dev91.
- **Pre-check.** No `h416.*` dev91 score exists: none on `main`, `v1/next` (31 entrants), the 307 remote branches,
  the local board root or the receipts (2026-11-10T06:53Z). `run_h418.sh precheck` repeats it, and the readout refuses
  a board that already holds an `h416.*` file.

## 1. Entrant (frozen; nothing is changed for dev91)
- **Board id** `h416.SXB`: entrant wave H416 (closeout `d9ee05954e`, `SXB_CONFIRMED` on TRAIN P201b). The design is
  `BCAR.mix` from H416 PREREG part B (`25d8c2d329`, `DESIGN.json`), with H416 amendment 1 (`23474c1c55`).
- **Population.** Per (context, search seed), three GP runs of the locked H335 AGG search: seed F + Evo A (K = 6 live,
  P = 48 screened), 50 generations, stock `AGG.g50.{single, CV3@12}`, superset worker `2a848c8f`, 60 s evaluation
  limit. They differ only in the screener:
  - **GS** (`H0.1-A`): the stored H335 Stage V run (stage `V_AGG`), replayed from its memo. No new search.
  - **GE** (`H381-E2-S3fz`) and **GL** (`H415-L`): new runs. The unit is `gp_h381.unit_lf` at `gp_h416`'s settings
    (`lf_rows` 0, checkpoint 50, stage `E416_AGG`).
  - **Screener form on dev91:** each screener's fold-0 pair (both training seeds), as in H335 Stage V and the H381 and
    H410 dev91 runs. No screener fold saw a DL-Val family. On TRAIN the same screeners were used out of fold.
- **Deploy.** `deploy_h416.h416_unit`, mode confirm, at its published bytes:
  - `deploy_h415.deploy_unit`: the replays, pools P2 and P3, the library, H (at most 64) and its 5-fold out-of-fold
    SEARCH matrix Q;
  - the `akf` (fold models → AUDIT) and `pa` (full-SEARCH refit → AUDIT) columns of every usable H program;
  - BCAR: 32 row-bootstrap Caruana bags fitted on Q (SEARCH out-of-fold data only, inside a selection scope), deployed
    as `mix` = 0.5 fold bag + 0.5 refit.
  - Deploy evaluation limit 300 s, as in H416. The deploy memo per context is `cat GS GE GL` (the H416 order).
- **Runtime.** Phase V: `h335_lab.build_rt` for the arms and `deploy_h410.build_rt_v` for the deploy, with the H335
  Stage V SEARCH views. Docker image `h317-empty:v1` on xbabe0: no network, read-only code, a TRAIN-only view until
  the unlock, `$DATA` the only writable path.

## 2. Seeds and the board value
- **Board value per context** = the mean over the two dev91 search seeds, 335101 and 335102, of the per-seed SXB
  value. Every GP entrant of ours on this board uses this convention (`h335.*` via `tools/import_h335.py`;
  `h364.XSTACK_2L`, H369), so SXB is read on the same footing. `per_seed` is recorded.
- **Disclosure.** On TRAIN P201b, H416 used one seed (335101), because that board's GP entrants are single-seed. The
  recipe per seed is unchanged; only the seed set follows the board. This was fixed before any dev91 read.
- **One process and one run dir per seed:** D1 = 335101, D2 = 335102. The H415 and H416 unit stores and the capture
  dictionary are keyed by context, so two seeds of one context never share a process.
- **Per (context, seed) value:**
  - the deploy's `P3+LIB3.SXB` value;
  - X0b if the unit has no usable H program (the H416 rule);
  - X0b on an amendment-1 exemption (§4.4).
- **X0b per seed** = the stored `h335.H335_AGG` `per_seed` value (= the stored Stage V `AGG.g50.CV3@12` output).

## 3. Steps (`run_h418.sh`, on xbabe2)
1. `stage TAG`: a create-once, read-only code root on xbabe2 and xbabe0 from the published bytes. Then `selftest`:
   `analyze_h418.py` on synthetic fixtures only (`selftest_h418.py`, no dev91 byte and no board file).
   - The clean case and the amendment-1 case must score, and the real `tools/ladder.py` must reproduce their Elo
     (≤ 1e-9).
   - A second analyze on the same outputs must be refused (create-once).
   - Each single harness fault must yield `INCOMPLETE` on its own gate, with no rows and no entrant value in the
     output. The faults are: GS replay, board file, LOCK manifest, ladder, X0b mean, config, missing record, AUDIT
     envelope, error record, SXB re-derivation, and an `h416.*` file already on the board.
   - Any failed case stops the run before `inputs`.
2. `gate`: every gate-file PR is merged. `precheck`: no `h416.*` dev91 score and no H418 LOCK receipt yet.
3. `inputs`: create-once copies of the H335 prep manifests, plus sha-only manifests of:
   - the stored GS files (91 memos + 182 `V_AGG` records);
   - the board files at the PREREG commit (`BOARD.json`, `ladder.json`, 31 score files).

   No value is read.
4. `lock`: `LOCK.json` (create-once, TRAIN-only sandbox).
   - `code_sha256` holds code, the H317 pins, the worker and the six fold-0 screener models. The code includes the
     readout (`analyze_h418.py`, `selftest_h418.py`) and the orchestration (`run_h418.sh`, `stage_h418.sh`), so the
     analysis is fixed before the unlock. It has no dev91 path
     (H369 attempt-1 lesson). The H317 guard refuses any DL-Val token in a path before the unlock.
   - The four manifests are in `inputs_sha256`.
   - The receipt `ns/receipts/H418_LOCK_sha256.txt` is create-once (noclobber), on xbabe0 and on xbabe2.
5. **xbabe0 cores:** posted on `AGENT_CHAT.md` before use (cl_apex's H417 units yield).
6. `unlock` (gate re-checked): the one-shot `C.unlock_stage_v()`. Then `select`: the 91 contexts / 19 families, each
   table and SEARCH view sha in the H335 prep manifests.
7. `gp GE` and `gp GL`: 91 × 2 units each.
8. `deploy-prep` and `memo-conflicts`. Then `deploy` D1 and D2.
9. `pull` (sha-compared copies on xbabe2). Then `analyze` (gates, then the board rows), then `board` (ladder).

## 4. Harness gates (any failure ⇒ `INCOMPLETE`; no value is read, printed or written)
1. **Board.**
   - **Identity:** `BOARD.json` is `dev91-audit-v1` with 91 contexts / 19 families and a full `board_sha256`.
   - **Files:** every file equals the board manifest, whose sha is the LOCK's `board_audit.sha256` input.
   - **Ladder:** the committed ladder is reproduced from the stored entrants (≤ 1e-6), fitted in `tools/ladder.py`'s
     entrant order.
   - **Entrants:** 31 complete entrants and no `h416.*` file.
   - **X0b:** every `h335.H335_AGG` record has 2 `per_seed` values whose mean is its utility (≤ 1e-12).
2. **Contexts.** `cids91.json` = the board's contexts = every deploy and arm config (91 contexts, 19 families, same
   families).
3. **LOCK and configs.**
   - **LOCK:** it names this entrant, the seeds 335101 and 335102, and `BCAR.mix`. Its `design_sha256` equals the
     staged `DESIGN.json`, and the staged `DESIGN.json` is that design.
   - **Configs:** tags and seeds (D1 = 335101, D2 = 335102); mode confirm. Every deploy config carries the staged
     design. Every arm config names its arm, both seeds and stage `E416_AGG`, and every config names this LOCK.
4. **Replays (≤ 1e-9, same members).**
   - `GS.stock` = the stored `V_AGG` output = the board's `h335.H335_AGG` `per_seed` value, for every (context, seed)
     except amendment-1 exemptions.
   - `GE.stock` / `GL.stock` = the arm records, for all 182.
   - **Amendment-1 rule, carried from H416 §10.** The deploy memo concatenates GS, GE and GL, and the loader keeps the
     last entry per key. A GS replay failure is admitted only on a context whose memo-conflict scan shows a
     GS-involving conflict. That (context, seed) scores X0b.
5. **AUDIT envelope** = the board envelope sha, 182 of 182.
6. **Re-derivations (≤ 1e-6):**
   - SXB recomputed from the npz columns;
   - `KCAR100.bag` = the unit's KFB;
   - the captured reference predictions = the H415 record values.

   H416 used 1e-9, and its P201b maximum was 8.4e-10. These checks test the harness, not the entrant, and 1e-6 is
   100× below the board's tie tolerance (1e-4). The maxima are reported.
7. **Records.** One confirm record per (context, seed). The only allowed error is "no usable H program". Wherever a
   usable H exists, SXB is completed and finite.

**On a failure.** Diagnosis reads harness fields only. A repair of the readout is a published amendment, made before
any SXB value is read. No completed unit is rerun. A unit with no record (an infrastructure crash) is resumed from its
memo under the same LOCK and code.

**Host reboot or crash.** xbabe0 rebooted twice on 2026-11-10, so this is written down in advance.
- Re-run the same `gp` or `deploy` step with the same stage and arguments. The run config must match exactly.
- A unit with a stored record is returned as stored, not rerun.
- A deploy unit with an H415 record but no H416 record is recomputed under `<tag>r`. Every evaluation there is a memo
  hit, and the readout uses that record.
- A unit that never finished is rerun from its memo.
- The Stage V unlock is never repeated: later steps resume Stage V.
- Every restart is logged, with its time, in `logs/units.log`. If a gate cannot be met, the result is `INCOMPLETE` and is reported; there is no
second score.

## 5. Output
- `scores/h416.SXB.jsonl`, written create-once in the board's record format (H369's fields; `harness_version`
  `h418-sxb-v1`).
- The ladder is regenerated by `tools/ladder.py` from stored scores only: anchor `h335.H257_REPLICA` = 1500, 10,000
  family bootstrap draws, seed 335160. Plus `LADDER_focus_SXB.md`.
- **Report:**
  - SXB's AUDIT-Elo and 95% CI, and its rank out of N (32);
  - the Elo gap to `h364.XSTACK_2L` and to the top entrant, with bootstrap ΔElo CIs;
  - paired against `h335.H335_AGG`, `h364.XSTACK_2L` and the top entrant: families W/T/L, gain, % RMSE, clipped mean,
    sign-flip p, LOFO.
- **Descriptive only (never a headline):**
  - seed 335101 alone and seed 335102 alone;
  - the board without the five H381/H410-exposed contexts;
  - the deploy's own values on any amendment-1 exemption.
- **No adoption claim.** dev91 is a reuse cohort: H257 tuned on it, and five contexts were exposed by H381 and H410.
  SXB's design was chosen on TRAIN P201e and confirmed on TRAIN P201b.
- **Recorded in:**
  - the board README (sections 1 and 3, and the rule-4 exposure log);
  - `SCORE_LOCATIONS.json`;
  - a create-only S3 archive, verified by GET;
  - `origin/v1/next`.

## 6. Closed
- No retraining or tuning on dev91. No rerun if the result disappoints. No other entrant is rescored (the ladder
  refit reads stored scores).
- The deploy unit's other outputs (KFB, BAG, HFR, …) are not read or reported. They enter only the harness identities
  of §4, as pass/fail plus maximum deltas.
- Untouched:
  - DL-Val outside `dev91-audit-v1`, DL-Test and sealed;
  - DOPE `c51a56d0`;
  - `provenance_bind.py`;
  - the flywheel writer, which stays stopped.

## 7. Compute
- xbabe0 cpus 16-63,80-127 (96 cores), nice 19, after the AGENT_CHAT post. CPU only; no GPU is used.
- GE and GL run in parallel (48 cores each), then D1 and D2 (48 cores each).
- **Launch settings.** The density matches H416 on P201b.
  - **Arms:** `gp ARM CPUS 12 32g`. H416 ran 4 shards × 11 workers on 44 cores per arm.
  - **Deploys:** `deploy SEED CPUS 8 48g` with `--inner-jobs 3`. H416 ran `--jobs 16` on 100 cores.
  - These settings change only timing: each unit is deterministic given its seed and memo.
- **Estimate from H416 on P201b (xbabe0 journal + status files):**
  - **Arms (GE ∥ GL).** H416: GE 1.48 h and GL 1.87 h, each on 44 cores for 201 units. Scaled to dev91 (182 units on
    48 cores): GE 1.48 × 182/201 × 44/48 ≈ 1.2 h; GL 1.87 × 182/201 × 44/48 ≈ 1.6 h.
  - **Cross-check.** The H335 dev91 run of the same search (364 units in 1 h 26 min on 96 workers) gives about 1.4 h
    per arm.
  - **Deploys (D1 ∥ D2).** H416: 4.57 h on 100 cores for 201 units; per unit, median 18 min, p90 28 min, max 56 min.
    Scaled: 4.57 × 182/201 × 100/96 ≈ 4.3 h.
  - **Steps around them.** Unlock, select, deploy-prep, memo-conflicts, pull, analyze and board total ≤ 1 h.
  - **Total.** About 7 h of wall clock, or about 550 core-hours. The range is 6.5–8.5 h; a long unit at the end, or
    other load on xbabe0, moves it toward the upper end.
