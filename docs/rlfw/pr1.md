# Research ledger, exact trie likelihood and decision records

This additive package records a comparable research incumbent, the actual trie
sampling likelihood and auditable draws. Importing it loads no deployed policy.
The existing service has no new default behavior.

The [ledger](../../deploy/dl-flywheel/pipelines/rlfw/research_hwm.py) hashes every
comparison identity, accepts measured integer `q_grid` in millionths, and uses a
SQLite WAL transaction with a version compare-and-swap. Its append-only event
history distinguishes `first_label`, strict `improvement`, ties, losses and
refusals. A fresh role callback runs inside each transaction. The caller must
verify the seven artifacts using the pinned evaluator before publishing;
artifact references by themselves do not qualify a result. State belongs under
the explicitly supplied benchmark `state/` directory.
Remote references retain their exact object version. The pinned local evaluator's
`path`, `sha256`, `bytes` references are accepted only beneath the sibling `runs/`
directory, with all seven byte hashes and lengths checked and no symbolic links.

The [likelihood API](../../deploy/dl-flywheel/pipelines/rlfw/trie_logprob.py) sums
token log-probabilities through EOS, normalizes only over sorted trie children,
uses float64 stable logsumexp, and makes singleton steps exactly zero. Grammar
normalization and mean-token NLL are refused. The pinned inference binding
teacher-forces prefixes using the deployed decoder's active-row batch shapes:
full-sequence and serial kernels can round differently in float32. Batch protocol
is part of replay identity. These are inference readouts, not differentiable
optimizer loss tensors.

The [decision trace](../../deploy/dl-flywheel/pipelines/rlfw/decision_trace.py)
writes a v2 record for each raw draw or selected slot, keeping predictions under
`predicted` and unknown propensities null. Replaying a decision ID is idempotent;
conflicting content is refused. Interrupted JSONL bytes are preserved and hash
verified before the complete prefix is repaired. Sealing atomically publishes a
manifest binding the exact JSONL bytes. Sealed traces refuse new records and
digest mismatches.

The [original incumbent adapter](../../deploy/dl-flywheel/pipelines/rlfw/hwm_adapter.py)
preserves the deployed native descriptor, including fold zero, and returns only
its original program and tokens. The program needs a fresh benchmark execution
before its score can act as a comparator.

## Acceptance and evidence

Run the data-free suite from the repository root:

```sh
TMPDIR=/home/ubuntu/.cache/jope-test-tmp/rlfw-l1 \
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 \
  python3 deploy/dl-flywheel/pipelines/rlfw/run_rlfw_offline.py \
  --receipt /home/ubuntu/jopedime_probe_exp/rlfw-20261009/logs/l1/pr1-fixtures.json
```

The registered fixtures cover an enumerable trie at temperatures 1.0 and 0.7,
mass and per-token error at most `1e-12`, competing EOS, the grammar
counterexample, identical/tied publications, every refusal reason, a 50-writer
monotone race with all 50 events retained, native identities, pin refusals,
crash/replay and manifest tampering. The runner is executed by current main's
Python CI lane and `just required`; both families have explicit coverage and
regression inventory entries. Source-only changes to `rlfw/` route to Python CI.

The four legacy offline runners named in the original plan are
`unavailable-on-main`, as recorded in L0's integration delta. They exist only on
the separate unmerged candidate. Their absence is not a passing test result.

Real acceptance uses only L0's immutable `code_pin` and `ckpt/current_latest`
copies, with the physical parameter pin
`686a6a605e262828a2a21fae4b9ce9f0a94fa6fe0ee3cd654fc610abda861c32`,
step 2769 and trie pin
`116154767f2df56fb065c9b6bc3ce07df38c3672843c0eb4b944d84e8e8473bd`.

```sh
PYTHONPATH=deploy/dl-flywheel/pipelines PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python/.venv/bin/python -m rlfw.verify_checkpoint \
  --policy-code-root /home/ubuntu/jopedime_probe_exp/rlfw-20261009/code_pin \
  --checkpoint-dir /home/ubuntu/jopedime_probe_exp/rlfw-20261009/ckpt/current_latest \
  --bench-root /home/ubuntu/jopedime_probe_exp/rlfw-20261009 \
  --out /home/ubuntu/jopedime_probe_exp/rlfw-20261009/runs/l1-pr1-device-1000 \
  --draws 1000 --batch-size 8 --seconds 2700 --device cpu
```

The host invocation is additionally bounded by an owned scope with CPU 200%,
MemoryMax 2 GiB and no swap, at nice 19 and idle I/O priority. The final replay
receipt and local required-gate outcome will be recorded here before review.

`--device cpu|cuda` defaults to CPU. Add `--compare-device cuda` to compare the
same complete sampled token rows on CUDA with CPU likelihoods and heads. The
receipt includes maximum absolute error in summed and per-token likelihood,
q, completion probability, and log1p runtime after the declared floor. All must
be within `1e-6`; unavailable CUDA fails instead of skipping. Same-seed replay
is tested on the selected sampling device. CPU and CUDA samples may differ
because their float32 kernels round differently, so cross-device comparison
uses identical token rows. GPU evidence is required before using GPU readouts.

An owner-scoped evaluator STOP can permit CPU-only inference with the explicit
`--evaluator-stop-sha256` flag. It accepts only the exact bytes of the known
evaluator parity STOP; a changed STOP, any other reason, or CUDA still stops
the replay. The flag defaults off and the receipt retains its hash. It neither
executes evaluators nor changes their STOP file.

U4 archive ranking is L0-owned. Records without emitted Rust parity remain an
unqualified archive diagnostic and cannot populate rewards, game comparators
or this ledger. The qualified ranking result remains pending; the final receipt
will distinguish its available coverage from the diagnostic.

Claim label for panel results: **Lane A, task-label-aware inner-CV search,
dev panel L=54, directional**. This PR's likelihood checks are numerical
acceptance evidence, not a search win-rate or quality-lift claim.
