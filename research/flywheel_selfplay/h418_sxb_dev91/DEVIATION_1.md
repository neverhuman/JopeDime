# H418 deviation 1 — the TRAIN bank-view link was missing from the H418 data root (fixed before any evaluation)

- **When:** 2026-11-10. The unlock was at 14:41Z. The first `gp GE` unit started at 14:41:26Z and exited at 14:41:31Z
  with status 1.
- **What failed:** `distributed_10gen_island_gp.ContextCoder.bank()` reads the merged TRAIN bank-view file
  `$H317_DATA/neural/views_c32_train1219.npz`. When that file is absent it falls back to per-context full views in
  `$H317_DATA/views/`. Neither existed under `/home/ubuntu/jope-data/h418-d91`, so `seed_A`, through `bank_nn`, raised
  `FileNotFoundError` on `h418-d91/views/002f7ead….npz` (a TRAIN bank context).
  - The traceback is saved in `logs/ge-attempt1/h418-gp-GE.err`.
- **Why:** H381 and H410 linked `$DATA/neural/views_c32_train1219.npz` to `/home/ubuntu/jope-data/h335-cl/neural/
  views_c32_train1219.npz`, and H410 pinned it in its LOCK `inputs_sha256` (`d91_h410.py:70`). My port into
  `d91_h418.py` dropped that line, so the H418 LOCK (`f923e5b7…`) does not list the file and the link was never
  made. The prep check and selftest never ran a GP unit, so neither caught it.
- **State at failure:**
  - `runs/GE` held only `config.json` and 182 work-table env files in `gproot_AGG/envs`. There was no memo, no
    stage record and no program evaluation.
  - GL had not started. No deploy had run, and no dev91 result of any kind existed.
- **Fix, at 14:47Z on xbabe0:**
  - Made the same symlink H381 and H410 used:
    `h418-d91/neural/views_c32_train1219.npz -> /home/ubuntu/jope-data/h335-cl/neural/views_c32_train1219.npz`.
  - Its sha256 is `eb042276f1cdc1e409db3f1d9086ac83c9d86417f1875d6615257c60b449bb56`
    (`h418-d91/merged_views.sha256`, 0444).
  - That sha equals the value pinned in the H335 LOCK, which locked the stored GS run that H418 replays, and in the
    H410 LOCK. The file is TRAIN-only (1,219 TRAIN contexts) and is already mounted read-only in the sandbox via
    `h335-cl/neural`.
  - No code, LOCK, receipt, PREREG or recipe parameter changed, and the unlock was not repeated.
- **Restart:** `run_h418.sh gp GE 16-63 12 32g`, the same step with the same arguments (PREREG §4 restart rule). This
  is not a rerun of a score. Nothing had been evaluated, and the scored entrant and protocol are unchanged.
- **Disclosure:** this note goes into the H418 closeout README and metrics. The analysis records the link's sha next
  to the LOCK inputs.

## Appended 2026-11-10T14:52Z — correction and a second defect (driver resume)
- **Correction:** the link was made at about 14:44Z, before the second launch, not at 14:47Z as stated above.
- **Second launch:** at 14:44:45Z it was refused in about 2 s with `run config mismatch on ['screener_manifest']`.
  Nothing was evaluated.
  - **Cause:** the first launch created `runs/GE/config.json` through `d91_h418._run_cfg`, then
    `gp_lab._record_screener_manifest`, called inside `LAB.build_rt`, appended a `screener_manifest` key.
    `_run_cfg` compares keys in both directions, so any later `gp` launch on that run dir is refused.
  - H416 and H381 compare only their `CONFIG_IDENTITY` keys, so this is a defect in my driver. The PREREG §4 claim
    that `gp` resumes after a reboot is therefore wrong for `gp`.
  - `deploy` is not affected: its config is `config_D<n>.json`, which holds the screener manifest from the start, and
    the D run dirs have no `config.json`.
- **Action:** the code is LOCKed, so it was not edited. Attempt 1 had no evaluations, so `runs/GE` was moved to
  `h418-d91/superseded/GE.attempt1` with a `.SUPERSEDED` note (183 files kept), and GE starts again in a new
  `runs/GE`.
- **If xbabe0 reboots during `gp`:**
  1. Copy the arm's `config.json` to `config.json.pre-resume`.
  2. Remove only its `screener_manifest` key. This puts the file back to what `_run_cfg` wrote.
  3. Re-run the same `gp` step. `_record_screener_manifest` then re-records the manifest, which must equal the saved
     copy (checked by `jq` before the analysis).

  Stored unit records are returned unchanged by `gp_h381.unit_lf`.
