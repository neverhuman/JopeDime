# Receiving-`main` (`92e1a55df284fe48643df266dcb29442aaea925e`) Receipt & Provenance Addendum

- **Recorded at (UTC):** `2026-10-19T08:52:30Z`
- **Receiving `origin/main` SHA:** `92e1a55df284fe48643df266dcb29442aaea925e` (`tree: 21260d88039712669de215355944b366bb7d20ff`)
- **Sole Source Integrator:** `2724936.captain_jopedime` (`jepsontaylor`, coordinated with `1061448.jopedime_flywheel`)
- **Independent Reviewer:** `487851.jope_review_claude` (`neverhumanbot`)

## Summary of Published Receipts & Preserved Historical Artifacts

1. **`MR03` (`PASS` on `92e1a55df284fe48643df266dcb29442aaea925e`):**
   - Receipt: [`MR03.json`](MR03.json)
   - Logs (embedded verbatim with byte-identical `content_utf8` and SHA-256 in [`RECEIVING_MAIN_92E1A55DF_BINDING_INDEX.json`](RECEIVING_MAIN_92E1A55DF_BINDING_INDEX.json) to preserve raw `cargo test` trailing-blank-line EOF bytes without `git diff --check` findings):
     - `MR03_spq1_prereg_provenance.f427d95e4.log` (`391` B, `sha256: ef8317081ffec82ff8c5a6f13bfc386d7ad9b66c6c35ffc2887186b47da6f604`, `3/3` passed)
     - `MR03_spq1_prereg_provenance.92e1a55df.log` (`391` B, `sha256: a3a5fee58059b6143fdeb878492835b0357aa9267416b9ef84899df5afbb3200`, `3/3` passed)
2. **`MR06` (`PARTIAL` on `92e1a55df284fe48643df266dcb29442aaea925e`, preserving historical [`MR06.json`](MR06.json) `f7c6f9ab...`):**
   - Receipt: [`MR06_92e1a55df_qualification_20261019T0821Z.json`](MR06_92e1a55df_qualification_20261019T0821Z.json)
   - Logs (embedded verbatim in [`RECEIVING_MAIN_92E1A55DF_BINDING_INDEX.json`](RECEIVING_MAIN_92E1A55DF_BINDING_INDEX.json)):
     - `MR06_label_admission_v2_preflight_verified.92e1a55df.log` (`684` B, `sha256: b4f740b22d665fd3df514c7b0bc4cefbed9572ef96052d47e42f6c89efafb8a8`, `4/4` passed)
     - `MR06_shard_index_payload_closure_verified.92e1a55df.log` (`367` B, `sha256: d2c93b245114f1e02d29cbc9ac78ac7b241d5259baca3e55946507b7f1dfdf93`, `1/1` passed)
3. **`MR12` (`PASS` on `92e1a55df284fe48643df266dcb29442aaea925e`, with `71f33add4` harvest & `9569bb60` archive):**
   - Accepted Receipt: [`MR12_92e1a55df_qualification_20261019T0841Z.json`](MR12_92e1a55df_qualification_20261019T0841Z.json) (and canonical copy [`MR12.json`](MR12.json)). Resolves full 40-char `92e1a55df284fe48643df266dcb29442aaea925e` from `.agent/worktrees/pr371-gate` git reflog (`HEAD@{2026-10-19T08:35:57+00:00}`) and binds:
     - [`logs/MR12_capability_join_audit_92e1a55df.log.txt`](logs/MR12_capability_join_audit_92e1a55df.log.txt) (`479` B, `sha256: e0bcaeaddaa60f6220d7ebcdb395af224927bad11bccc691fb8e890006559d69`, `3` unique tests)
     - [`logs/MR12_operation_profile_92e1a55df.log.txt`](logs/MR12_operation_profile_92e1a55df.log.txt) (`501` B, `sha256: be0067418e151dcd0273fe92be3d4b76e2c1fa44d83bc36bc4cc007aeac2e541`, `4` unique tests)
     - Total: `7` unique tests executed (`0` double-counted).
   - Preserved `71f33add4` Harvest: [`MR12_71f33add4_harvest_20261019T0810Z.json`](MR12_71f33add4_harvest_20261019T0810Z.json) (`sha256: 5199ad733f6f644f6c080255be530e98874654eae09ba9d1cd7f22b3779a2bb7`), with [`logs/mr12_main_rerun.71f33add4.log.txt`](logs/mr12_main_rerun.71f33add4.log.txt) (`1,230` B, `sha256: db7482df525c7badb1d0d8c32f4aa0557f20723f30c311a1ba6693c6ee4a9225`) and [`logs/mr12_main_rerun.71f33add4.sh.txt`](logs/mr12_main_rerun.71f33add4.sh.txt) (`463` B, `sha256: 6ad9f4796b73c9d01bdf8b34483589acd37bf6638f97cdb61734acf36c404806`).
   - Preserved `9569bb60` Archive: [`MR12.historical_9569bb60.json`](MR12.historical_9569bb60.json) (`sha256: 529d6ccc25e4be31490b0b63c1ea9048792780dbd14ebdd8aaf9a404223990b7`).
4. **`MR17`/`MR19` Bounded Read-First Production Readiness Audit:**
   - Audit Receipt: [`MR17_MR19_PRODUCTION_READINESS_AUDIT_20261019T0841Z.json`](MR17_MR19_PRODUCTION_READINESS_AUDIT_20261019T0841Z.json)
