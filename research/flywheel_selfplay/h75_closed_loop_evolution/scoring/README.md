# H75 CPU point scoring support

Standalone support for ASTRA's H75 score interface and OPUS's runner, in scoring/ only. This computes **point diagnostics**, not bootstrap intervals, the seven-contrast conjunction, an experiment result or a promotion. The unrelated H75 W4 shard is preserved.

Input is one JSON packet with `manifest` and `rows`. The caller freezes a `jope.h75-closed-loop.score-admission/v1` manifest containing explicit boolean `fixture_only`, `run_id`, `source_git_sha`, `prereg_sha256`, `execution_manifest_sha256`, frozen `dope_sha256`, `panel`, and `roster`. Each roster cell has `context_sha256`, resolved `family_id`, `fold`0..4, `seed`701/702/703, `role: DL-Train`, and paired `evaluation_identity_sha256`. All three seeds are required for every context; families cannot cross folds. Confirmed-U8 cells also pin the ordered three `confirmation_cut_sha256s`.

Rows follow `jope.h75-closed-loop.score/v1` in the H75 INTERFACES contract: execution identities, panel/policy, context/family/fold/seed/role, selected program SHA, evaluation identity, status and raw utility; confirmed-U8 rows include the exact three cut identities. Legacy uses A,B,C,D,replay_B,replay_D; confirmed_U8 uses A,B,C,D,G,R. Reject duplicates/conflicts, mixed panels, foreign roster/role/identity, unresolved families and nonfinite utilities. Missing cells or failed/timeout/unknown execution return **INCOMPLETE**, counts and no Elo/utility metrics. Completed rows require finite utility; no missing values are replaced with zero.

Seeds are averaged within context **before** the pinned repository Bradley–Terry helper sees context records. Each family then has equal weight across its contexts. A is anchored1500, tie1e-4, prior.01,400iterations. A is the preregistered r882 continuity control; prior exposure remains conditional. W/T/L count context-mean outcomes, not independent families. The existing helper rounds Elo to two decimals; unrounded input/context/family utility means are retained. Context-mean `val_reg_inner` is reported separately from equal-family mean utility; confirmed_U8 is never relabelled `val_reg_inner`. Raw utility and Elo can disagree.

Use the same-source repository helper `python/src/jope/north_star/elo_tracker.py`, SHA256 `9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835`. Exact bytes are hashed before module execution; its data loaders and baseline entry point are never called. NumPy is required. From the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B -m unittest discover -s research/flywheel_selfplay/h75_closed_loop_evolution/scoring -p test_score_panel.py -v
nice -n 10 python3 -B research/flywheel_selfplay/h75_closed_loop_evolution/scoring/score_panel.py packet.json --elo-source python/src/jope/north_star/elo_tracker.py
```

Tests use tiny fabricated development utility records, no dataset/model reads, fits, network, GPU or qualification lock. CLI records packet/admission/helper/scorer hashes and scorer Git HEAD; these do not prove a clean checkout or upstream provenance. `H75_ELO_SOURCE` can point tests to a byte-identical reference source.

**Integration remaining:** OPUS supplies a frozen complete authoritative roster and validates upstream role/alias, source and evaluation identities, actual legacy gate/failure mapping, and the preselected program plus all three unrounded confirmation utilities. This helper validates joined declared identities; it cannot prove how the manifest/utility was produced or detect a fabricated family assignment already present in it. OPUS must select the registered fold/track/contrast and prevent leakage/reranking. The receipt always declares bootstrap NOT_COMPUTED (0 of10000 required paired-family draws, seed20261031) and admission=null. Preregistered shared-index family intervals and the scientific decision remain unimplemented by this bounded shard.
