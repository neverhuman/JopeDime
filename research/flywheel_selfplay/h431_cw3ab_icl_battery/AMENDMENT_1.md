# H431 Amendment 1 — Stage `screen_h425.py` and per-stage output root before any battery/table output

- **Time & state:** Recorded at `2026-11-11T16:06Z` after `pub-r1` exited at Step 1 (`python3: can't open file '.../screen_h425.py'`) immediately following `test_cw3.py` (`6/6`), **before** any `SCREEN_FITSEL.json`, `ICB_v1.npz`, battery task, or real-table monitor prediction was written.
- **Root cause:** Base stage `/home/ubuntu/apex-stage/h427/pub-r1` descended from `h426/pub-r1 -> h422/pub-r1` and therefore did not carry `screen_h425.py` from `h425/pub-r2`.
- **Change:**
  1. `stage_h431.sh` stages `screen_h425.py` from `origin/v1/next:research/flywheel_selfplay/h425_contamination_screen/screen_h425.py` and byte-verifies it against `/home/ubuntu/apex-stage/h425/pub-r2/code/screen_h425.py`.
  2. `driver_h431.sh` and `gates_h431.sh` bind `R` to `/home/ubuntu/jope-data/h335-cl/h431/$(basename "$ST")` (`pub-r2`) so `pub/` (`unit_tests.json` from `pub-r1`) is preserved untouched and every `pub-r2` file has creation time strictly after this amendment commit.
- **Unchanged:** Every pre-registered hypothesis (`H1`–`H3`), checkpoint SHA-256, `ICB_v1` task/reference specification, and Part B source file are 100% unchanged.
