# Fixed best-of-32 reranking

An explicit `rerank32:<checkpoint>` arm collects the first 32 distinct canonical
programs from seeded trie sampling, stopping at 512 raw draws. Each decoder call
contains at most 32 seeds; all returned draws remain in custody, including repeats
and the final batch's unused programs. The existing immutable catalog verifies
each program's bytes and complete token sequence. Likelihoods use trie children
and sum through EOS, with the sealed checkpoint and batch protocol recorded.

The pinned policy teacher-forces every slate member and records q, completion
probability and runtime under `predicted`. Selection sorts by descending predicted
q, breaking ties by canonical SHA. It excludes completion probabilities below
0.5 when at least eight candidates pass; otherwise it ranks the complete slate.
The eight selected programs execute in that order. Raw likelihood remains in the
trace, while selection propensity is NULL because the deduplicated ranking's
probability is unknown. No predicted value enters a program descriptor, evaluator
request, measured score, ledger or KPI.

The scheduler commits the complete raw packet, slate, predictions, selection and
phase timings in its isolated SQLite outbox before evaluation. It also publishes
a create-only, fsynced packet under the run's `slates/` directory and emits every
raw decision before disclosing any score. A restart validates and reuses the
packet without sampling or head inference. A 512-draw shortfall with eight or more
distinct programs still selects eight and records the shortfall. Fewer than eight
preserves the packet and raw trace, then refuses evaluation. Exact cache hits keep
PR2's per-slot charging and qualification rules.

The behavior is enabled only by that arm in a frozen GameSpec with `slate_k=32`.
Baseline arms and the running service have no new default. Paired baseline and
rerank games may differ in arm-specific K while retaining identical contexts,
seeds, order, B=8, tau=0.01, frozen cache, original comparator, runtime and device.
The optional mixed exploration arm is outside this change.

## Acceptance

```sh
python3 deploy/dl-flywheel/pipelines/rlfw/run_rlfw_offline.py --receipt RECEIPT.json
just required
```

The registered data-free fixtures cover fixed selection, feasibility fallback,
SHA ties, exact leaf membership, duplicates, all raw draws, shortfalls, invalid
head fields, packet drift, restart replay, blind disclosure, eight charges and
prediction isolation. CI and independent exact-head review remain required.

The earlier physical head-binding proof scored the same 32 distinct complete
programs in a batch and individually on the sealed current-latest checkpoint.
It used CPU, one Torch thread and Torch 2.14.0+cu130, with zero evaluations or
optimizer updates. Maximum absolute differences were q `7.152557373046875e-7`,
completion probability `6.116551709567375e-12`, and raw log1p runtime head
`4.76837158203125e-7`, each within the fixed `1e-6` criterion. The receipt SHA is
`521da6c9270762434120c5320ed33e978f49b280bef0b2b2458ddf3857890dbd`;
the policy binding's unchanged bytes have SHA
`4cd16b3f5aa84a38f58094f0c0dfe620c2ab72ab81e3320e110035b4da79b0a1`.
The separate derived-seconds diagnostic exceeded `1e-6` at
`2.5767277911370456e-6`; its failed first receipt and all inputs are preserved.
This proof qualifies the declared raw model head units on that CPU environment.
It supplies neither a CUDA certificate nor a replica-host qualification.

## Runtime admission

The existing arena CLI accepts `--device cpu|cuda`, default CPU. Rerank policy
arms require the same sealed 1000-draw likelihood certificate and temperature
1.0 as sampling arms. CUDA additionally requires the complete identical-pin
CPU/CUDA likelihood and head parity certificate. The injected binding must expose
`sample_many(context_id, arm_name, seeds)` and
`score_heads(context_id, arm_name, complete_token_rows)`.

```sh
"$RLFW_PYTHON" -m rlfw.bench_cli arena --spec "$B/runs/rerank_T1/spec.json" \
  --runtime-binding "$B/l0/arena_binding_source_cpu.py" --policy-code-root "$B/code_pin" \
  --policy-qualification "$B/runs/l0-source-cpu-qualification-v5/qualification.json" \
  --device cpu --reuse-hwm-from "$B/runs/baseline_T1" --pool-workers 4
```

Set `RLFW_PYTHON` to the qualified source interpreter (Torch 2.14.0+cu130).
The source route verifies its sealed V5 facade and immutable CPU1000 receipt;
plain system Python (Torch 2.7.0) and the original writable facade are refused.

Use the reviewed merged source, qualified bounded campaign scope and freshly
pinned runtime family. The current native parity STOP blocks live games; this
source and the CPU binding proof do not clear it. No live rerank outcome, lift,
generalization result or serving promotion is claimed. Any eventual result uses
the label “dev panel, directional; CPU inference” and records unrun contexts as
unknown.
