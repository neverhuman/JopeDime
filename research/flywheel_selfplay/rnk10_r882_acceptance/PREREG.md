# RNK-10 r882 CPU acceptance diagnostic

This one-shot execution binds the owner-designated r882 checkpoint and pointer to their exact
hashes, source identities, and the full 91-context DL-Val contract. The roster is regression-only,
spans 19 families and 19 catalog-listed datasets, and matches every contract context, pool hash,
and training-table hash. The source manifest records 93 excluded holdout datasets; no holdout
payload or outer row is admitted.

The guided arm loads the frozen r882 adapter. The paired unguided arm uses that checkpoint's
frozen anchor adapter. Both select four lowest-cost gate-passing proposals after the four
pre-revealed positions, with stable source-order tie handling. Utility is the best negative
fitness among the four revealed and four selected proposals. The four preregistered gates are:

1. Maximum per-context conditional KL from guided to anchor proposal probabilities, using
   softmax(-cost / 0.4208), is at most 0.35.
2. Median of the 19 family-mean guided-minus-anchor utility deltas is positive.
3. The one-sided 97.5% family-bootstrap lower bound is positive (10,000 draws, seed 214001).
4. Family-grouped Bradley-Terry Elo for guided exceeds 1500 with anchor fixed at 1500.

The fresh ledger permits one run and at most 800 unique selected candidate outcomes; the planned
maximum is 728 (91 contexts × four guided plus four anchor selections, with duplicates deduplicated
within context). Four labels per context are already declared as common revealed inputs. The pinned
scorer loads only the catalog-selected DL-Val pools and records selected terminal outcomes; no
training, optimizer update, GPU, signing, qualification lock, or live-loop write is used. CPU
execution is one math thread at nice >=10, host CPU <=85%, deadline 2026-11-01T07:00Z.

This cohort is the checkpoint's development contract and has prior checkpoint-selection exposure.
The result is a one-shot diagnostic, not independent generalization evidence and not RNK-10
qualification. M5, S4, S2, Q1, and production qualification remain separate.

The training source commit is identified by the full-run OPERATIONS record and the checkpoint's
bound source hashes; an exact run_manifest_round882.json and Git metadata for the source snapshot
are absent. The evaluator source is pinned separately to the repository commit recorded in the
manifest. See evaluation_manifest.json and outcome_ledger.json for the full machine-readable pins.
