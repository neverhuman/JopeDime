# R25-kNN-CE: one pre-registered scratch arm

R25-kNN-CE is a **new hypothesis**: teaching the policy the frozen HWM teacher programs of a context's nearest training neighbours lowers validation L6, relative to frozen v1.

H1 (GB-06) seeded GP with each context's *own* HWM, not with neighbours, so it is **not** evidence for this hypothesis. Power is **unmeasured**.

**Status: NON-QUALIFYING scratch.** R22 targets carry no seven-artifact closure, and there is no promotion, seal or live use.

## Approval

- ENG:watcher signed it off (owner decision) at 2026-10-13T07:42:55Z, reconfirmed it as a new hypothesis at 08:03:31Z, and approved option A (1,408 updates plus a KL-guard contingency) at 08:23:05Z.
- It is **one** run, with no search, and it is evaluated **once**.

## The arm (`configs/r25.json`)

**Matched control.** It is identical to sweep 6's `r23` arm, the matched control:
- start from frozen v1 `ea078456`, restoring v1's AdamW moments;
- LR 3.75e-5, batches of 16 contexts;
- the same 5,628-context L6 roster (full roster minus held-out lineages), with sweep 6's lineage holdout pinned;
- the same KL guard (alarm 0.03, stop 0.05);
- a stop at exactly **1,408** updates (4 epochs of 352). The harness caps one run at 7,020 s of wall time, and at the measured ~3.9 s per update 2,621 updates would not fit (option A).

**Objective.** The arm adds a single term:

`L6 + 0.2·KL(π‖v1) + 0.25·CE_kNN`

`CE_kNN(c)` is the mean trie-children negative log-likelihood, under context `c`, of the frozen native HWM teacher programs of `c`'s **k = 4** nearest neighbours.

**Neighbour table** (`knn_table.py`):
- distance is cosine on the frozen-normalised 2,051-d encodings;
- neighbours come only from the L6 roster, and only from contexts whose teacher is a trie leaf;
- a context's own parent lineage is never a neighbour;
- held-out-lineage, validation and every other non-roster context are never neighbours;
- the table is built once before any update;
- its digest is pinned in the config (`e9feb0ce…`), and the harness refuses any other table.

## Evaluation

The evaluation is G-JUDGE's pre-registration (prereg `4e4d2574`, script `6e4ddb3f`), with the CPU L6 run by CX-KV:

- **Primary.** The per-context paired ΔL6 = arm − frozen v1 on the 263-context full validation set, taken on the final checkpoint only. L6 is a loss, so lower is better. The 95% CI comes from a parent-lineage cluster bootstrap.
  - Candidate: the CI upper bound is below 0.
  - Regression alert: the CI lower bound is above 0.
  - Otherwise the result is INCONCLUSIVE.
- **Secondary.** The paired comparison against sweep 6's `r23` at its epoch-4 checkpoint (`dfe98994`, 1,408 updates).
- **KL-guard contingency.** The unchanged guard freezes the arm at an epoch boundary when probe KL ≥ 0.05. If it does, the primary look uses that frozen checkpoint, and the secondary compares it with `r23` at the same epoch.

Evaluation telemetry after each epoch, inherited from the v6 harness, is never treated as a look.

## Files

| File | What it is |
|---|---|
| `lpi_scratch_train_v7.py` | Harness. Revision 6 plus the kNN-CE term and `max_updates`. torch is required. |
| `lpi_scratch_data_v4.py` | Host-configurable, sha-pinned inputs (unchanged from revision 6). |
| `knn_table.py` | The neighbour table and its verifier. |
| `launch_v7.sh` | Launches inside a capped scope (the harness reads `RuntimeMaxSec` back and requires it). |
| `configs/r25.json` | The signed-off config. |
| `run_r25_offline.py`, `test-identities.json`, `tests/` | CPU fixtures with the exact identity roster. |

The training run needs the host-local native snapshot, the sealed v1 capture and a GPU. CI runs the CPU fixtures only.
