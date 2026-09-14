# GP evaluation registry — never re-run the same individual on the same cut

Status: **plan + observability counter landed; live GP 774452 not swapped.**
Measured 2026-09-10 on xbabe2 / 8×A100. Communicate in `~/bullet/AGENT_CHAT.md` and `~/jope-dime/AGENT_CHAT.md`.

This is not a cache that invents scores. It is a **fail-closed identity lookup** that refuses to pay for a fit that has already been measured under the same evaluation identity.

## 0. What “same dataset” means

Never re-execute a **canonical program** on the **same prepared cut** (same `training_sha256` + `validation_sha256` + preprocessing + evaluator + feature view).

Sibling cuts of the same dataset are **different** evaluations. Transfer measurement requires them. Do not key the registry on dataset name alone.

## 1. Where waste happens today

| Layer | What exists | Gap |
|---|---|---|
| L0 in-search | `evaluated_programs` BTreeSet of canonical genomes; projection dups rejected before fit | Only this search. Teachers and neural seeds are still fitted. |
| L1 job resume | `observations.json` + attempt files | Only this output directory. |
| L2 pack | First `(program, cut)` wins | **Silent.** No `duplicate_program_cut` counter until this change. Claude: ~37% of 14.0M observations. |
| L3 teaching store | `INSERT OR IGNORE` first label | Same silence across ledgers. |
| L4 campaign | None | Generations 29–48 re-fit the **same incumbent/retained teachers** on the same 140 cuts, 3 arms, every generation. |
| Admission | 16-field `evaluation_identity_sha256` | Necessary, **not** sufficient: missing threads/OMP/Rayon, loaded libs, clean-completion receipt, full 30s window. |

Gen-48 neural, one cut (already measured): 3 incumbent + 7 retained + 6 cross-cut + 64 neural + 920 evolve = 1000 evals. The **10 teacher genomes are the same individuals on the same cut, every search.** Across 140 cuts × 3 arms × N generations that is the dominant repetitive waste.

Within-search evolve already skips canonical dups. The fleet waste is **cross-search / cross-generation / teacher re-fit**, then pack dropping the repeats invisibly.

`shared_teaching_cache.py` is a **memfd of tensors**, not a program-score registry. Do not overload it.

## 2. Identity — the only legal cache key

Reuse is allowed only when **all** of these match. Any unknown field is a **miss**.

**Admission 16 (already in `label_admission.rs`):**
`program_sha256`, family, partition, `prepared_sha256`, cut, `training_sha256`, `validation_sha256`, `evaluator_sha256`, `evaluator_protocol`, `preprocessing_sha256`, `target_scale`, `metric_units`, `vocabulary_sha256`, `feature_view_sha256`, `evaluation_seed`, `resource_protocol`.

**Execution manifest v1 (from cache-contract-review-v2, required before any hit):**
evaluator binary + entry point; loaded numerical libs; CPU/FP mode; OMP/Rayon/OpenBLAS/MKL threads and affinity (live GP sets threads=1 — bind that); fit RNG; sealed input bytes; 30s / 8 GiB window still fully available; admission-policy digest.

**Completion:** clean parent wait exit 0 **and** valid metrics **before** the deadline. `failure:null` plus a stray result file is **not** a hit (gp-clean-completion-v2). Timeouts, signals, disagreements → miss and quarantine, never a label.

Canonical program = `GenomeIr::parse(genome)?.canonical()`. Raw pre-projection strings are not keys.

## 3. Registries (fast path)

### R1 — in-search (exists, keep)
BTreeSet of canonical genomes. Reject before spawn. Count `rejected_proposals`.

### R2 — job ledger (exists, keep)
Resume from `observations.json`. Do not re-fit committed rows.

### R3 — cut-local score index (new, O(1))
SQLite (WAL) or mmap hash map **per prepared/cut**:

```
key = sha256(canonical_admission_json || execution_manifest_json)
value = { observation_sha256, result_sha256, rmse, units, semantic_sha256, completion_receipt, source_search }
```

- Lookup: one indexed PK get. Target **<100 µs** after the DB is warm (not a 30s fit).
- Publish: write payload → fsync → atomic manifest. Readers accept only committed rows.
- Conflict on same key, different score → **quarantine the key**, do not pick the better RMSE.

### R4 — campaign teacher/seed registry (new, highest ROI)
Before evaluating any seed with origin `incumbent_cut_teacher` | `retained_cut_teacher` | `neural_policy` | `same_family_cross_cut_teacher`:

1. Compute identity.
2. R3 lookup.
3. Hit → attach bound receipt, mark `evaluation_origin=reuse`, increment `registry_hit`, **do not spawn**.
4. Miss → fit, then publish.

This alone removes ~10 teacher refits × 140 cuts × 3 arms × every generation.

### R5 — generation-frozen snapshot (matched arms)
Before a generation starts, freeze R3 as **read-only**. Neural / generic / shuffled all see the **same** initial map. None may read another arm’s *new* discoveries during the generation. That keeps the comparison protocol honest while still refusing historical re-fits.

After the generation commits, publish new successful clean identities into R3 for the next generation.

### R6 — pack counters (landed in workspace, not live packer)
`duplicate_program_cut` increments on intra-ledger skip and inter-ledger `or_insert` / `INSERT OR IGNORE`. First label still wins. Provenance must show this bucket. Live packer SHA stays until a qualified native rebuild.

## 4. Fast lookup implementation notes

- Key is 32-byte SHA-256. SQLite `WITHOUT ROWID` table or a sorted `identities.bin` + `values.bin` with binary search is enough. Do not JSON-scan observations at fit time.
- Process-local LRU of decoded receipts (N=10k) in front of SQLite.
- Memory: ~100 bytes/key + path. 10M identities ≈ 1–2 GB index, acceptable on the 256 GB node.
- Never block the 104 GP workers on a single writer. One publisher thread, readers snapshot a generation-frozen file.
- Teaching-store `programs` table is **not** the fit cache: it lacks evaluator/runtime/completion identity.

## 5. Accounting (or the cache will lie)

Keep separate counters, never overload `completed_evaluations`:

| Counter | Meaning |
|---|---|
| `proposal_attempts` | Genome draws |
| `rejected_proposals` | In-search canonical / projection dups |
| `registry_lookup` | R3/R4 gets |
| `registry_hit` | Reused clean success |
| `registry_miss` | Executed fit |
| `registry_conflict` | Quarantined key |
| `duplicate_program_cut` | Pack/store first-wins drops |
| `executed_fits` | Evaluator actually ran |

A hit **consumes one evaluation-cap slot** in protocol v1 (cache-contract). Search that only replays hits is a **new protocol** and must be labeled. Do not claim 1000 new programs when 80 were reuse.

## 6. What we will not do

- Key on dataset name or displayed RMSE.
- Reuse across cuts, views, vocabularies, or evaluator SHAs.
- Cache failures, timeouts, or `failure:null` after a killed worker.
- Let one arm mutate another arm’s map mid-generation.
- Swap live GP 774452 or the hold controller for this.
- Add `macro_*` tokens or weaken the paper promotion gate.
- Dump 1,600 new cuts to “use” the registry.

## 7. Phased rollout

| Phase | Work | Go / no-go |
|---|---|---|
| **P0 — visibility** | `duplicate_program_cut` in `corpus.rs` + teaching store; tests | Counts appear; first-wins unchanged. **Done in workspace.** |
| **P1 — source registry** | R3 schema + lookup helper; no GP wire-up | Unit tests: hit/miss/conflict/short window/dirty completion. |
| **P2 — teacher/seed reuse** | R4 in a **separately pinned** native binary; R5 freeze | Matched-arm RNG tests still pass when registry is empty (uniform miss). Teacher evals drop to 0 on a replay generation. |
| **P3 — evolve reuse** | After propose, R3 lookup before spawn | `registry_hit` rises; unique programs per search do not silently fall without being reported. |
| **P4 — live qualify** | Node-built binary, exact-source fixture, no mid-queue swap | Same as any GP handoff: durable boundary, GP 774452 stays until empty or isolated queue. |
| **P5 — historical ingest** | Only rows that revalidate under current admission + clean completion | Unknown fields = miss. No score laundering. |

Causal experiment if funded: cache-off vs cache-on, same seeds/ancestry, same 1000/600 budget, report hits vs executed separately. Adaptive search will diverge; that is expected.

## 8. Relation to other defects

- **Hold stays.** A registry does not make another 3000-step train eligible.
- **Stage D:** serving prior still 0/140 seeds beat teacher. Reuse teachers; do not re-fit them to manufacture novelty.
- **Claude selection-noise:** better fitness is a separate lane (v45). Registry is complementary: stop paying 30s for a known teacher RMSE.
- **Clean-completion v2 / champion-recovery:** registry values must embed those receipts. Do not ship P2 without them in the value schema.
- **Enrichment:** cross-cut seeds were +35% vs teacher on gen 48. Reusing a bad seed is fine; do not treat reuse as qualification of enrichment.

## 9. xbabe2-only UI CI (not GitHub)

TUIWright (jankurai) and Playwright require this node’s authenticated CLIs. They **must not** be required checks on hosted CI.

Command: `scripts/xbabe2-local-ui-ci.sh` from `jope-dime` (and the same script copied under `~/bullet/scripts/`).

- TUIWright: `/home/ubuntu/.local/bin/tuiwright` from `/home/ubuntu/jankurai`.
- Playwright: `bullet-portal` mocked contract lane (`ops/ci/contract.sh`) when that checkout is present.
- Fail closed if the binaries are missing on this host; skip-with-error on any other machine (`XBABE2_LOCAL_UI_CI=1` required).
