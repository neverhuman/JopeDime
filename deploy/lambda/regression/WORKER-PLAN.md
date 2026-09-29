# Regression evolution: execution and recovery plan

Objective: explore new typed GP programs across eligible regression views, retaining independent three-fold CV vectors, full fit artifacts, historical provenance and exact resource events. Target stages are generations 10, 25, 50, 100 and at most 256; generation zero is initialization, not an evolved transition. Strongest observed CV candidate is a research outcome, not a guaranteed generalization winner.

Non-goals: canonical score/HWM/policy promotion, sealed-label access, source dataset migration, retroactive rewriting, deleting historical evidence, or treating test-selected legacy scores as fresh CV. Outer development and permutation-importance assessment are separate post-freeze work.

## Read first

1. `~/JopeDime/AGENTS.md`, `MASTER_PLAN.md`, `docs/adr/ADR-0005-canonical-learning-loop.md`, `docs/learning.md`, `catalogs/s3.json`.
2. Existing `schemas/jope.score.v1.schema.json`, `schemas/jope.hwm.v1.schema.json`, `crates/jope-learning/src/{record,canon,scores,hwm}.rs`.
3. This directory's `SCHEMA-EVOLUTION.md`, `schemas/REGISTRY.json`, `common.py`, `campaign.py`, `rust/src/main.rs`.
4. `~/scratch/jopedime-regression-lambda-v3-20260928/{s3-audit/census.json,history/COMPLETE.json,deployment/versions.json,protocol.json}` when present.

## Ownership and limits

Source is `~/JopeDime/.agent/experiments/regression-lambda-v3`; runtime evidence is `~/scratch/jopedime-regression-lambda-v3-20260928`. Only the S3 prefix `s3://veox-jopedime-learning-use2/experiments/regression-evolution/v1/` is writable for experiment data. Existing source, staging, research and canonical roots are read-only. No Git ref/index or product PR changes. Other agents own concurrent product and regression-fitness-v2 work; do not stop or modify it. Use the claimed shared build target with Cargo jobs at most two; never start a second competing build.

User-authorized Lambda ceiling is $1,000 total, including $13.6535432 previous conservative reservations. `Budget.reserve` charges each request at full configured timeout before invocation, using the higher x86 list rate plus 10%; no uncertain-work refunds. Qualification admits at most $20 new reservations. Fixed stop is 2026-09-29 05:00 UTC. Storage admission is 25 GiB, including 32 MiB per unresolved invocation and every mutable HEAD version. Stop before deadline, budget, storage exhaustion, provenance/hash mismatch or unexpected privileges. AWS monthly budgets have reporting delay; they do not enforce the local instantaneous ceiling.

The private mode-0600 `.env` holds only profile, run directory and campaign ID. Credential resolution uses the existing external profile provider. Never source `aws.env`, print credentials, archive `.env`, or place keys in source. Runtime has logs-only IAM; signed exact-input/output capabilities are transient and never written to receipts.

## Ordered execution

1. Audit with `audit_s3.py` and `sample_schemas.py`. Retain listing interval, current-version scope, bucket controls, original sample bytes and hashes. Do not infer replication from versioning/Object Lock.
2. `history_index.py` binds legacy observations to source receipts and parses exact DSL identities. Preserve raw and canonical hashes; incomplete old fit contexts are proposal-only. `prepare.py` keeps official outer training rows only, and `prepare_screened.py` explicitly marks lower fidelity. Every corpus view needs an outcome or exclusion.
3. Build locked/offline native and ARM Rust executables in the reused target; retain commands, hashes, toolchain and tests. Run `python3 test_boundaries.py`, `python3 schema_contract.py`. Check `qualify_storage.py` on real S3: concurrent writers admit one claim; restarts and completed claims cannot repeat.
4. `deploy.py` amends the existing approved budget and preserves notification destinations/stop-role identity, creates logs-only owned runtime and four ARM ZIP tiers. Read back numeric versions, memory, timeout, code hash, reserved concurrency and account headroom. Never amend the binary after protocol freeze without a new protocol/evaluator identity.
5. Qualify five representative small views: `python3 campaign.py --only registry-0014b9dad5725a1e--stored forest_fires__r0__f0 obesity_estimation__r0__f0 airfoil_self_noise__r0__f0 wine_quality__r0__f0 --through-generation 2 --workers 4 --qualification`. Run a completed stage again and prove zero additional invocation attempts. Inspect complete and failed models, exact metric replay, AWS REPORT duration/memory, versioned S3 packs and controller resume.
6. Increase owned concurrency only after qualification and preserve a readback receipt; leave at least 100 account executions unreserved. Coordinator also bounds archive liability to 256 concurrent admissions. Run breadth-first stage targets across all admitted views, then deepen within fixed limits. All failed/unknown claims stay seen. Do not change memory for the same claimed context to hide failures; a scientific reevaluation needs an explicitly different context and policy.
7. Derive query tables for evaluations, fold metrics, generations, invocation costs, resource peaks, exclusions and exact UTC events. Use program/context/attempt hashes as join keys. Keep row/model payloads in immutable archives. Publish audit, schema registry, source archive, manifests and create-only final COMMIT only when evidence closure verifies.
8. Restore representative complete contexts from exact S3 versions in a fresh local directory, independently replay metrics and verify hash/size closure. Report actual coverage, generation distribution, retained unknowns, observed Lambda cost versus reserved ceiling, and storage bytes. No zero-fill for missing metrics.

## Hard parts and failure interpretation

- Conditional HEAD admission is the serialization point. Never retry an invocation because the HTTP response was lost: reconcile the attempt object first. A missing context event must fail closed, never reset the seen set.
- Generation intent includes parents, proposal counter and proposed DSLs. Upload immutable generation state before committing local state; restore cached candidates from hash-verified packs. A generation with no complete new CV candidate stops that view.
- Exact duplicate numeric rows and declared groups stay in one inner fold. Temporal folds expand forward with time ties intact. Pre-split normalization in old CSV sources stays flagged; it cannot be repaired by renaming the score.
- Constant-target R², invalid-domain MAPE/RMSLE and absent held-out results are null with semantics. Do not add these metrics to the closed v1 legacy enum. Full successful models are serialized and prediction-replayed before acceptance.
- Fitness is the mean log fold RMSE/baseline ratio plus 0.2 fold SD, with a stable hash tie-break. Fold SD is descriptive, not an independent-sample confidence interval. Historical best scores and old test labels never enter this comparator.
- Some heavy models exceed per-program time, memory or archive limits. Retain explicit timeout/failure and any completed folds; do not call those programs evaluated successfully or silently relaunch them.

## Logging, live view and handoff

`watch_feed.py` produces the existing `jope.lambda-watch.snapshot.v1` contract. This is a disposable operator view; raw microsecond timestamps, monotonic durations, AWS report tails and all scientific records stay archived. Run the existing Rust viewer with the new `watch.env` and `--snapshot-only`. Never label a stopped feed live.

Keep source/deployment/qualification/dispatch receipts and per-view stop reasons. Append claim and final handoff to `AGENT_CHAT.md` under its lock; preserve exact source HEAD, owned paths, quota/deadline, remaining limitations and evidence locations. Final status must distinguish implementation, local tests, real AWS qualification, running stages and completed scientific outcomes. No product merge is part of this lane.
