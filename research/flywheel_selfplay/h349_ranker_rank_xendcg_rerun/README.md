# H349 — rank_xendcg rerun

Hypothesis: the fixed listwise objective may improve TRAIN robustness. BASE is
archived unchanged H303 NESTED/H308 P0/P1/P2 x3roots x4folds. Only ranker
objective=rank_xendcg plus objective_seed=original random_state changes; native
sigmoid remains BASE1.0. Full inverse ASTs,38data-free controls,972nativeBASE
rankers,BASE9panels/36fold-root hashes+locks/3medians EXACT and1219TRAIN input
hashes+4cacheprefixes PASS. No BASErefits or treatment fits. See PREREG.md.

Outcome: NOT_RUN, awaiting ONE5second core/SMT gate and full36 runtime forecast.
Scientific metrics/verdict UNKNOWN. CPU only; DOPEc51a56d0unloaded;no dev91
or speed verdict. Preregistration commit: `d9886e2ecf5fda82727b8b6ecc9af723d7326757` (pushed 2026-11-05T04:07:47.620783+00:00, before any treatment fit).
