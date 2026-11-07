# H414: the frontier regressor `kimik3p1_max` on every stored board

**Status:** complete, 2026-11-07. Owner: `cl_fwriter` (Claude Opus 5.5). Pre-registration: [`PREREG.md`](PREREG.md)
(published on `v1/next` `cce696b8ee` before the LOCK).
**Owner order (Jepson, 2026-11-07):** "We have a new model we added: xbabe2:~/JopeDime/kimik3p1_max.rs please add it to
the right location in our code base, and please score it on ALL data and include the elo ranking with this new item.
Don't rush this, double check it and make sure you are really doing it right, in the right location, etc.."

## The entrant
- **Source:** the owner's `kimik3p1_max.rs`, 47,156 bytes, sha256 `0c211694…5d6ceb3`. It is used byte-for-byte and
  never edited.
- **Code location:** `crates/jope-automl/src/kimik3p1_max.rs`, registered in `jope_automl::REGRESSORS` next to the
  other frontier regressors, with `crates/jope-automl/submissions/kimik3p1_max/`. That change and the AutoML Elo v3
  result (`assets/automl-elo/v3/`) go to `main` through their own pull request (branch
  `feat/kimik3p1-max-automl`). This directory holds only the board scoring.
- **Authorship:** the file came with no authoring manifest, so its authoring conditions are recorded as unknown.

## Result: Kimi on all four panels
| Panel | Rows | Kimi rank | Elo (95% CI) | Kimi family-mean u |
| --- | --- | ---: | --- | ---: |
| AutoML Elo v3 (TRAIN; 64 families × 3 folds × 3 seeds) | its own cut | **3 of 6** | 1510.7 (1469–1554) | — |
| [`p201b-train-audit-v1`](../../elo/reference-board/boards/p201b-train-audit-v1/LADDER.md) (TRAIN; 201 contexts) | AUDIT | **12 of 19** | 1351.8 (1311–1391) | 1.3188 |
| [`dev91-audit-v1`](../../elo/reference-board/boards/dev91-audit-v1/LADDER.md) (DL-Val91; 91 contexts / 19 families) | AUDIT | **24 of 31** | 1411.0 (1305–1509) | 1.1509 |
| [`dev91-s1-two-fold-v1`](../../elo/reference-board/boards/dev91-s1-two-fold-v1/LADDER.md) (DL-Val91) | H257 two-fold | **13 of 14** | 1370.0 (1277–1447) | 0.9867 |

Elo depends on the field and anchor of each panel (AutoML centre 1500; AUDIT boards `h335.H257_REPLICA` = 1500;
two-fold `h257.Opus5p5_Extra` = 1500). Compare entrants within one row only.

**Frontier-only view** (the 6 LLM regressors; rank agreement only, the panels are not comparable with each other):

| Entrant | AutoML v3 | P201b TRAIN | dev91 AUDIT | dev91 two-fold |
| --- | ---: | ---: | ---: | ---: |
| Opus5p5_Extra | 1 (1690.8) | 1 (1574.4) | 1 (1564.1) | 1 (1500.0) |
| Gemini4_argon_high | 2 (1603.3) | 2 (1470.4) | 2 (1519.4) | 2 (1499.2) |
| GPT6p1_Sol_Max | 4 (1497.5) | 3 (1438.8) | 3 (1503.2) | 3 (1476.4) |
| **kimik3p1_max** | **3 (1510.7)** | **4 (1351.8)** | **4 (1411.0)** | **4 (1370.0)** |
| GPT6_Astra_Max | 5 (1389.2) | 5 (1212.2) | 5 (1312.9) | 5 (1284.8) |
| grok4p7_xhigh | 6 (1308.4) | 6 (1093.5) | 6 (1120.7) | — (no two-fold score) |

- The order is the same on all three board panels: Opus > Gemini > Sol > **Kimi** > Astra > grok.
- On AutoML v3 Kimi is 13 points above Sol, inside both intervals; it wins 55% of families against Sol there
  (`assets/automl-elo/v3/SOURCES.md`).
- Kimi beats grok significantly wherever grok is scored (AutoML v3 Holm p 0.010; P201b and dev91 AUDIT pass the H335
  primary test). Against Astra it passes on P201b only. On both dev91 boards it is ahead of Astra without passing,
  and on AutoML v3 the two are level.

**Head to head on `dev91-audit-v1`** (`tools/ladder.py --focus h414.kimik3p1_max`; gain = family-equal mean of
u(Kimi) − u(other); RMSE cut = 1 − exp(−gain), positive = Kimi has lower RMSE; pass = the H335 primary procedure,
sign-flip p < 0.025 and LOFO min > 0):

| vs | Gain | RMSE cut | Families W/T/L | Sign-flip p | Passes |
| --- | ---: | ---: | --- | ---: | --- |
| `h412.grok4p7_xhigh` | +0.2870 | +24.9% | 18/0/1 | 0.001368 | yes |
| `h335.GPT6_Astra_Max` | +0.1294 | +12.1% | 14/0/5 | 0.05259 | no |
| `catboost.v1` | −0.0377 | −3.8% | 8/0/11 | 0.6619 | no |
| `autogluon_300s.v1` | −0.1015 | −10.7% | 9/0/10 | 0.7519 | no |
| `h335.Opus5p5_Extra` | −0.1434 | −15.4% | 5/0/14 | 0.9761 | no |
| `h335.GPT6p1_Sol_Max` | −0.2035 | −22.6% | 4/0/15 | 0.983 | no |
| `h335.Gemini4_argon_high` | −0.2078 | −23.1% | 5/0/14 | 0.9802 | no |
| `h364.XSTACK_2L` (ours) | −0.4099 | −50.7% | 1/0/18 | 0.9995 | no |
| `kumo_tabular_large_e8.v1` | −0.7847 | −119.2% | 0/0/19 | 1 | no |

Our `h364.XSTACK_2L` beats Kimi in 18 of 19 dev91 families (Kimi's RMSE is about 51% higher). The full views against
every entrant of each board are in [`focus/`](focus/).

## How it was run
- **Harnesses** (frozen in `PREREG.md` §3 and in the LOCK):
  - AUDIT boards: `frontier_audit_h414` = the H412 harness plus a sixth entrant; binary sha256 `c34025df…a627162`.
  - Two-fold: `twofold_eval_h414` = H257's two-fold frontier evaluator plus a fifth entrant; binary sha256
    `97bf8653…6cd871`.
- **Pre-LOCK checks (TRAIN and synthetic only):**
  - both binaries rebuild bit-for-bit;
  - on the 201 TRAIN P201b envelopes, the four H335 frontier entrants reproduce all 804 stored H335 TRAIN records, and
    grok reproduces its 201 H412 records;
  - on the 16 TRAIN two-fold envelopes, the four H257 frontier entrants give the stored H257 binary's records in every
    field except `wall_ms` (64 of 64);
  - Kimi completes 201/201 and 16/16, and a second run of each is identical.

  So the stored rows of every earlier frontier entrant stay comparable with Kimi's.
- **LOCK** `73c7b415aef902986f3530a2e17d305294709be125a7375eba021aded44d1858`: create-once receipt
  `~/jope_plan/ns/receipts/H414_LOCK_sha256.txt` (0444), one unlock (`STAGE_V_STARTED`, 18:01:50Z), 0 guard refusals.
- **dev91 AUDIT:** 91/91 completed, 18:01:50Z–18:02:04Z, xbabe2 CPU. Median wall 0.36 s per context, max 0.67 s.
- **dev91 two-fold:** 91/91 records, 0 failed CV gates (no u = −2.0), 18:02:05Z–18:02:10Z.
- **P201b:** the pre-LOCK TRAIN records (same binary), bound to the LOCK; plus grok's existing 201 H412 TRAIN records
  as `h412.grok4p7_xhigh` (imported, not rerun), so the TRAIN board now carries all 6 frontier entrants.
- **Boards:** each score file was installed create-once. Each board's `scores/` diff is exactly the new file(s), and
  every stored score file is byte-unchanged. Each `LADDER.md` regenerates identically from the git copy.

## Where the scores are
- **git:** `research/elo/reference-board/boards/<board>/scores/h414.kimik3p1_max.jsonl` on the three boards (plus
  `p201b-train-audit-v1/scores/h412.grok4p7_xhigh.jsonl`), the regenerated `LADDER.md` / `ladder.json`, the reference
  board README and `SCORE_LOCATIONS.json`. Results summary: [`metrics.json`](metrics.json).
- **S3 (create-only; exact-version GET and every member's sha256 verified)** — each board's `ARCHIVE.json`:

  | Board | Archive | Version | Members |
  | --- | --- | --- | ---: |
  | `dev91-audit-v1` | `results-20261107T182859Z.tar.zst` | `DqmPEMjiNJS5FuJ8XiqG3ZouJyz.fjJE` | 1,490 |
  | `p201b-train-audit-v1` | `results-20261107T181208Z.tar.zst` | `GpYasPYVHJRwUAW6ZGSMUE_Nq9CVLVCE` | 1,062 |
  | `dev91-s1-two-fold-v1` | `results-20261107T181212Z.tar.zst` | `su61Z25ug8p1ZjpxFZndStJY4do9qskJ` | 23 |

  All under `s3://veox-jopedime-learning-use2/experiments/reference-board/<board>/`. The AutoML v3 evidence
  (576 cells) is archived separately; see `assets/automl-elo/v3/SOURCES.md` on `main`.
- **Run record:** the H414 closeout bundle (raw run records with the two-fold fold predictions, parity runs, binaries,
  build trees, LOCK, receipts and the code pull request's gate logs): [`CLOSEOUT.json`](CLOSEOUT.json).
- **Study refresh:** [`final_study.json`](final_study.json), written by [`final_study_h414.py`](final_study_h414.py).
  This is the H412 study script with the H413 and H414 entrants added: 31 dev91 AUDIT entrants, 14 two-fold entrants,
  six frontier LLMs and AutoML v3. It reads stored files only. Run from this directory,
  `python3 -B final_study_h414.py --repo <checkout>` regenerates it byte for byte. AutoML v3 is read at commit
  `706f2beca2` (the head of the code pull request's branch `feat/kimik3p1-max-automl`).
- Never rerun `h414.kimik3p1_max` on these boards: read the stored rows (board rule 1).

## Not done
- No model of ours, no reference model and no earlier frontier entrant was rerun, refit or rescored.
- grok4p7_xhigh has no two-fold score; that would be a new DL-Val91 read for an entrant the owner did not name.
- dev91 is a reuse cohort (H257 tuned on it), and 5 of its contexts are adaptively exposed (H381 / H410). Nothing was
  chosen on dev91.
- Sealed / holdout / DL-Test stay closed; DOPE `c51a56d0` is not touched; the Grok lane's PR is not touched.
