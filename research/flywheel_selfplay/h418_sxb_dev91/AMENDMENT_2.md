# H418 Amendment 2 — one GE replay cell scores the X0b fallback (published before any SXB value is read)

## 1. Decision and authority
- **Owner decision:** Jepson, 2026-11-10 3:22 PM MST (22:22Z), option B. His words: "Go with B, Amendment 2: score
  that cell with the H335_AGG fallback, report the deploy value as descriptive only, self-test, then run the analysis
  once into a new output."
- **PREREG basis:** §4 "On a failure": "A repair of the readout is a published amendment, made before any SXB value
  is read. No completed unit is rerun."
- This amendment is on `origin/v1/next` before the amended analysis runs. Until then, no SXB value has been computed
  into any output, printed or read.

## 2. What failed
- **The analysis run** (`analyze_h418.py`, 2026-11-10 21:49–21:55Z) stopped at harness gate `ge_replay_335102`:
  status `INCOMPLETE`, exit 3, no rows written.
- **Outputs kept unchanged:**
  - `INCOMPLETE_1.analysis.json`, a byte copy of `results/H418_DEV91.json`, sha256
    `ee9652bf293672ecc7c28bec460dea417e5477b72c734de359fb701ad572ce6a`;
  - `results/H418_DEV91.txt`, sha256 `bc58581d578006e8e5cd2cd08641b43b1e0240bcc53b91643198d97bfc6d7f59`.
- **The diagnosis**, from harness fields only, is in `INCOMPLETE_1.md`.
  - **The cell:** (context `43022ae6a8bc…`, seed 335102, source GE). D2's memo replay of the GE search swapped one of
    the 12 `AGG.g50.CV3@12` stock members: program `35f192f2…`, which only GL had evaluated, replaced `261e8073…`.
    The value delta is 3.43e-5.
  - **Every other cell matches:** the other 363 (context, seed, source) arm replays are identical, with members
    equal and delta 0.0. All other gates passed.
  - **Likely cause:** each GE arm process ran both seeds of a context, and the screener's per-(context, program)
    score cache was shared between them. Each deploy runs one process per seed.

## 3. The rule (changes PREREG §4.4 and §2 only)
- **Gate.** A (context, seed, source ∈ {GE, GL}) whose deploy replay differs from the arm record (value > 1e-9 or other
  members) is admitted only if it is listed in `a2_cells.json`.
  - The gate `<source>_replay_<seed>` passes iff the failing set equals the listed set for that seed and source.
  - With an empty list, this is the PREREG gate.
- **The list** has exactly one cell: (`43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`, 335102, GE).
  - The list must match the failures exactly. A listed cell that does not fail, any unlisted failure, a wrong source
    or seed, or a malformed file is `INCOMPLETE` on its own gate.
- **Value.** An admitted (context, seed) scores X0b, the stored `h335.H335_AGG` per-seed value, as an amendment-1
  exemption does. The source label is `a2_exempt_x0b`. The board value for the context is still the mean of its two
  per-seed values.
- **Descriptive only.** The deploy's own SXB value on that unit, and the Elo and rank with it substituted, appear
  under `readout.descriptive.a2_deploy_values`. They are never a headline.
- **Record format.** `harness_version` is `h418-sxb-v1-a2`, and `source` names both amendments plus the
  `a2_cells.json` sha256. The keys are unchanged.

## 4. What does not change
- **The scored data.** The entrant, recipe, seeds, LOCK `f923e5b7…`, every other gate and tolerance, the readout,
  the board and ladder rules, the descriptive sensitivities, and §5–§6 are all unchanged.
- **Compute.** No unit is rerun, and nothing is retrained or tuned. No new dev91 compute is done: the analysis reads
  the pulled, sha-checked records only, and no other entrant is rescored.
- **The amended readout** is `analyze_h418_a2.py`: `analyze_h418.py` at its published bytes plus the diff for §3
  (replay gate, `a2_exempt_x0b`, `a2_deploy_values`, harness label).
  - The published `analyze_h418.py` and `selftest_h418.py` are not edited.
  - `selftest_h418_a2.py` reruns every `selftest_h418` case against the amended readout with an empty list, and adds
    the listed-failure and wrong-list cases.

## 5. Steps (`run_h418_a2.sh`, on xbabe2)
1. **`stage`** (create-once `/home/ubuntu/h418-stage/a2`): the `pub-p1` code, checked against its CODE.sha256, plus
   the five A2 files at their `origin/v1/next` bytes.
2. **`selftest DIR`**: `selftest_h418_a2.py` on synthetic fixtures with the published `tools/ladder.py`. Every case
   must pass. A pass record for this stage's CODE.sha256 is required by `analyze`.
3. **`analyze`**, run once:
   - It first creates the noclobber receipt `ns/receipts/H418_A2_ANALYZE.txt` and refuses if the receipt already
     exists. The receipt is never removed.
   - It writes the new create-once outputs `results/H418_DEV91_A2.json` and `results/H418_DEV91_A2.txt`, plus the
     rows `results/h416.SXB.jsonl`.
   - It checks the INCOMPLETE outputs against the shas in §2 before and after.
   - Whatever this analysis returns is the H418 result; there is no further analysis.
4. **`board`**: `run_h418.sh`'s board step, read against `H418_DEV91_A2.json`.
5. **Closeout:** as PREREG §5. The closeout README discloses `INCOMPLETE_1`, this amendment and the descriptive deploy
   value.

## 6. Lesson for later waves (not applied here)
When a deploy replays an arm one seed per process, run each GP arm with one process per seed, or with no screener
score cache shared across seeds. The arm's search path then does not depend on the order of seeds within a process.
