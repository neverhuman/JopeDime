# Finite-support readouts and decode protocols

`trie_enumerate(policy, normalized_context)` returns registered leaf tokens,
program SHA, EOS-inclusive `logprob`, predicted `q_hat`, `p_complete` and runtime.
It never executes programs. `cpu_exact` uses singleton active-row float32 MATH
SDPA prefix+PAD forwards and float64 child normalization. Singleton edges have
log probability zero. A seeded 1% audit calls `PinnedPolicy.logprobs_many` with
one leaf per call, preserving the same geometry. A multi-row audit would bind
another arithmetic identity and is not certified at 1e-12.

`kv_analysis` uses incremental MATH SDPA and a byte-bounded same-context prefix
LRU. One instance owns one frozen context and rejects changed parameters or
cross-context inputs. Cache payload, transient activations and CUDA allocator
memory are separate quantities. Callers must provide a capped host scope and
CUDA allocator cap; this library neither claims GPUs nor launches processes.
Its numerical tolerance must be declared before a run and checked afterward.

`strict_decode` preserves the deployed batched sampler, per-row CPU RNG and CPU
float64 softmax. Its LRU caches whole forward geometries for one call, including
contexts, active rows, lengths and padding. It uses full forwards: **bitwise
incremental KV acceleration remains unqualified**. The small real transformer
fixture measures backend-dependent drift and requires 1e-6 closeness; zero drift
is valid on some backends. Existing-checkpoint GPU receipts retain measured
nonzero drift. This separation
lets the exact protocol be used without silently changing deployed probability.
`kv_sample` is explicitly `jope.kv-trie-analysis/v1`, with bounded memory,
independent row seeds, complete-leaf checks and a deadline.

`score_full_support` evaluates existing heads over every registered program
without computing likelihoods. The returned head field `q` is a prediction
from the existing PinnedPolicy API; it is named `q_hat` in enumeration output.
C12 can consume this helper and its elapsed time independently of enumeration.

`ranking_report` accepts already joined, qualified comparable measured labels
and separate predictions. It reports per-context Spearman (NULL on constant
ranks), measured-support top-8 recall and decile/length calibration. It refuses
duplicate programs and nonfinite labels. Top-8 recall is uninformative when
there are eight or fewer measured programs; boundary ties count fractionally.
Selected executions can bias these descriptive metrics. Full-trie ranking skill
requires unselected qualified labels and is not established by predicted scores.

Run the isolated 15-test CPU fixture suite (including 1,000 strict traces):

```sh
python deploy/dl-flywheel/pipelines/rlfw/run_kv_offline.py --receipt kv-fixtures.json
```

Use the repository's Python environment with CPU torch. The existing stdlib-only
RLFW fixtures stay in a separate interpreter because they assert lazy imports.
CI runs this additive suite separately through the Python lane.

Measured scratch evidence lives in the authorized CX-KV custody lane, outside
tracked source. Receipts bind checkpoint, code, trie, context and source hashes.
A single-context timing is a measured scratch result, not a workload throughput
qualification or a serving-promotion receipt.
