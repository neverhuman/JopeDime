Frozen successor support (GB-05 / J13)
====================================

`jope_support` is a local Rust boundary for action availability. It accepts a pinned catalog,
fixed deployed vocabulary, frozen context roles and explicit external program records. Rust's
pipeline parser/printer checks canonical text; SHA-256 and unique byte-piece tokenization bind
each complete BOS…EOS leaf. An extension creates a new version with its parent's digest and
retains every original catalog leaf. Consumers provide the expected support digest and repeat
semantic verification, including TRAIN provenance for noncatalog actions. Unknown, validation,
heldout and sealed roles cannot extend support. Version one may require the exact deployed
6,640-leaf catalog; `require_v1_catalog=false` explicitly permits a declared fixture/new catalog.

All emitted artifacts say `qualified=false`, `qualified_use=BLOCKED_JEPSON_G12`, and
`availability_grants_context_outcome_eligibility=false`. A source or parity receipt digest is
retained provenance; this tool does not authenticate its measurement body or qualify a reward.
Context outcome eligibility, measured inner/incumbent losses, action availability and training
weights remain separate. Missing losses remain null. No teacher label, q-head target, learning
worker, grammar decoder, evaluator or serving pointer is changed. The q-head stays shadow-only.

The frozen schema supports the deployed 282-token byte-piece vocabulary and pipeline GenomeIr
form, with a maximum 640 tokens including BOS/EOS. Other vocabularies/forms require a separately
pinned implementation. A sorted-leaf trie supplies the same legal children to both categorical
sampling and teacher-forced likelihood. CPU float64 stable softmax uses the declared temperature
(0.1…2); likelihood sums every predicted token, including EOS. `weighted_archive_ce` consumes
these sequence sums with explicit nonnegative unit-mass weights, without length normalization.
Same-support KL comparisons must also pin unchanged reference parameters; a different support
or reference parameter digest is refused. This contract does not switch the live trainer's
objective. A Torch caller must use the exported child mask for its differentiable logits and
bind context, parameter, support and temperature identities separately.

Local usage:

```
cargo run -p jope-learning --bin jope_support -- freeze --input request.json --output support.json
cargo run -p jope-learning --bin jope_support -- verify --support support.json --sha DIGEST
cargo run -p jope-learning --bin jope_support -- extend --support support.json --sha DIGEST --candidates candidates.json --output support-v2.json
cargo run -p jope-learning --bin jope_support -- index --support support.json --sha DIGEST
cargo run -p jope-learning --bin jope_support -- score --support support.json --sha DIGEST --tokens 1,...,2 --logits rows.json --temperature 1
cargo run -p jope-learning --bin jope_support -- mismatch --rows comparison-rows.json
cargo test -p jope-learning --test support --locked -- --nocapture
```

`FreezeRequest` and `Candidate` in `version.rs` define the importer input. Candidates retain
`context_id`, canonical `Program` (DSL, program SHA, tokens), `source_receipt_sha256`, optional
`parity_receipt_sha256`, and optional measured `inner_loss`/`incumbent_inner_loss`. The role map
has its own expected canonical SHA. Clover's A3 winner export can map its context/program,
search/parity digests and measured RMSE fields to those explicit records; its qualification
fields do not authorize G12. Output files are create-only and fsynced. Interrupted files and
caller-pin mismatches fail consuming verification. The library can serve larger supports;
`index` deliberately bounds an exported prefix table to two million prefixes.

Portable real-checkpoint proof:

* `tests/fixtures/support-policy-v1.json`, byte SHA
  `f1fe0756517983376af62ec21c0b3422fa1f1abdfcfebdd1dde1d0212e725b6c`,
  contains 254 actual CPU float32 MATH-SDPA prefix-logit rows from frozen v1 parameters
  `ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb`,
  state SHA `3cdd73d8cef758e4c1895fc2ab7ea811c0c5c69b05709aca012eafc8e3492b19`.
  Geometry: d_model128, two layers, four heads, d_ff512, ctx_dim2051, vocab282, max_seq640;
  evaluation disables dropout. One pinned native TRAIN context, no incremental/cross-context KV.
* The fixture uses two actual catalog programs and external program
  `b30339169c103abd9bf4fa26ba708c15102652cf9aafc36224c8d5e2a7484c00`.
  The Rust test reproduces all 1,000 seeded token sequences exactly (339 external draws),
  checks EOS-inclusive per-step and sequence likelihoods at 1e-12 and unit support mass.
  Torch supplies float32 forward logits; reference normalization and Rust use float64.
  CI needs no Torch, checkpoint, host corpus or program execution.
* The capped lane-only forward adapter and full-catalog readback live in
  `.agent/rlfw-coord/lanes/CX-KV/gb05/`, outside product source. Its full support receipt
  verifies all 6,640 pinned catalog leaves plus that external action (6,641 total).
  Forward generation used main `0ed1a381174b86e1a2d6d624e77af6916451d8f8` and unchanged
  deployed model/tokenizer code; final source/test receipts bind the candidate commit separately.

`MismatchReport` aggregates already measured, uniquely identified per-context logprob rows.
Its sign is trie sequence LL minus grammar sequence LL: a positive value is extra grammar NLL,
not an L6 improvement. Historical val263 rows at source
`2018bb090ac01026d2bb66a496c78cab294dc12f` recorded mean73.04490611611129,
median62.7365273026155, range0.02868340350155929…399.6956786699671 across263contexts/54parents;
max sampler/recompute error3.552713678800501e-15. Those are prior read-only observations,
not a new training result. The Rust aggregation preserves sequence/EOS units and explicitly
reports `training_objective_changed=false`. L6 is a loss: LOWER is better.
