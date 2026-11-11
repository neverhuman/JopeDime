# H425 PREREG addendum 1 — CONFIRM tables are not in the sandbox view; the feature screen runs on DEV only

**Status.** Published on `origin/v1/next` before the screen has produced any output. The first attempt (10:16Z,
published code) stopped with a file-not-found error on a `CONFIRM` table before writing anything.

## 1. What was found
The sandbox's table view (`~/jope-data/h335-cl/apex/tables`) holds 661 files: all 440 `FIT-ICL` tables, the 115 core
`DEV` tables, the 106 core `FIT-SEL` tables, and **none of the 241 `CONFIRM` tables**. H417 built the view that way:
`CONFIRM` cannot be read through the guarded path at all.

## 2. What is not done, and why
- **The feature screens S1–S3 are not run on `CONFIRM`.** Running them would mean copying the `CONFIRM` tables into
  the view or reading them from their source folders outside the guard. Either step removes a protection the
  campaign's plan put in place ("read once by the champion wave"). The relayed note asks for the `CONFIRM` screen
  but does not mention this; I did not take that step on my own.
- It is held for an explicit decision. If it is approved, the PREREG run (`--queries DEV,CONFIRM`) works unchanged
  once the 241 tables are staged read-only in the view, and takes under a minute.

## 3. What is done instead
- **`DEV`:** the PREREG's screen, unchanged in screens, thresholds and tiers, on the 115 core `DEV` contexts
  (`--queries DEV`). One consequence: "rare" in S3 now refers to 555 screened tables, not 796.
- **`CONFIRM`, manifest only (`confirm_meta_h425.py`; no table is opened):** identical dataset id, table hash or
  family id with any `FIT-ICL` table, and the source-name screen S4 (same rule; the code checks that its constants
  equal those of the screen). Output `CONFIRM_META.json`, frozen with the `DEV` list.
  - This was run once as a code test before this addendum: 0 shared ids or hashes; 53 of 241 contexts have a
    readable name; 1 name flag (`meta-album-md-5-bis-mini` against `meta-album-md-6-mini`, probably two different
    datasets of one benchmark).
- **Failure criterion for `CONFIRM` (PREREG §6):** it cannot be evaluated. What can be said is stated in the README:
  the manifest-only result, and the rate of copies the same screen finds in `FIT-SEL` (rehearsal) and `DEV`,
  which is the best estimate for `CONFIRM`. **That estimate is escalated in place of a flag list.**
- Sensitivity read of H424 (PREREG §5): unchanged.

## 4. Gates
`gates_h425.sh` is replaced: the screen ran on `DEV` only; no `CONFIRM` context is in its output and the view still
holds 661 files; the `DEV` list and `CONFIRM_META.json` are published before the sensitivity read; both this
addendum and the PREREG are older than the screen's output. The other gates are unchanged.

## 5. Files changed
`screen_h425.py` (accepts `--queries DEV`), `lists_h425.sh` (cuts the `DEV` list only), `gates_h425.sh`,
`confirm_meta_h425.py` (new), `stage_h425.sh`.
