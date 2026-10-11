# Gate A probability and successor support

These adapters and fixtures are **SCRATCH, NON-QUALIFYING**. A successor trie
remains **BLOCKED_JEPSON** for qualified use pending the HWMKey/trie-extension
ruling. No worker entrypoint or continuous/L6 objective is changed.

`probability_contract.actor_logprob` calls the actual `PolicyLM.forward` with
exactly the prefix/PAD layout and active row ordering used by
`neural_decode.BatchedTrie.decode_many`. It disables dropout, normalizes over
sorted trie children using float64 log-sum-exp, includes EOS and returns the
sequence sum with its gradient graph. Archive weights must be finite,
nonnegative and sum to one. This is distinct from the existing L6
`objective.sequence_ce`: grammar support, float32 softmax, and mean token NLL.

The finite fixture uses the actual model, decoder and objective. It checks
sampled/recomputed likelihood, sum-to-one over three finite actions, raw action
ordering/duplicates, EOS, sequence sum, and the probability increase after an
actual weighted-CE optimizer step. The compact trie must produce exactly the
same tokens, traces and recomputed likelihood as the eager trie on this fixture.

```sh
CUDA_VISIBLE_DEVICES='' uv run --project python --extra cpu python \
  deploy/dl-flywheel/pipelines/rlfw/run_gate_a_offline.py \
  --receipt target/health/rlfw-gate-a.json
python3 deploy/dl-flywheel/pipelines/rlfw/run_rlfw_offline.py \
  --receipt target/health/rlfw-offline.json
```

Both fixture runners are registered in the CPU Python CI lane and `just
required`. They require nonzero collection, all collected cases executed and
no skips. An optional `--onpolicy-source PATH` adds a read-only integration
fixture against the actual CX-OP package, recording all package byte hashes.
That package's `sampling.py` schedules contexts/replay; program token sampling
is the actual neural decoder. The optional fixture does not vendor that
unmerged package or pretend it is part of this branch.

`SuccessorBuilder` checks the actual 6,640-program v1 catalog identity
`116154767f2df56fb065c9b6bc3ce07df38c3672843c0eb4b944d84e8e8473bd`, including
its vocabulary and maximum sequence length. Only measured programs from
explicit training-role contexts can extend it. Native audit inputs retain the
original spec, current role/exclusion authority and every detail shard's SHA.
Token grammar, canonical token/DSL roundtrip, model length and program SHA are
checked separately from outcomes. Unknown quality remains unknown; global
availability does not supply another context's reward or admit it to training.
A generation freezes against further additions. The manifest records size,
added program SHAs, operator coverage, input and role pins, and a versioned
support hash. Qualified use is blocked independently of fixture results.

The eager decoder caps its catalog at 100,000 programs. `CompactSupportTrie`
retains every leaf in a sorted uint16 prefix index (up to an explicit four
million leaves); it creates only visited child tables. It invokes the unchanged
actual `BatchedTrie.decode_many` algorithm. It neither truncates support nor
changes sampling temperature or normalization.

The reference is frozen start checkpoint parameter SHA
`ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb`
evaluated **under the successor mask**. A reference under the old mask is
refused: new positive action mass would make forward KL infinite. Changes in
old action probabilities due to support expansion are reported separately and
are not called policy drift.

`audit_native_support.py` reads only pinned local inputs on CPU. It reports one
seeded v1-trie sample per each of the 263 development contexts at temperature
one, comparing float64 summed grammar/trie likelihood and unchanged L6 mean
CE. Its new-support smoke uses the same checkpoint, a verified measured added
training action with finite likelihood, and an actual unforced noncatalog draw
under the whole successor support. Source/data hashes, command, times, raw
samples and batch protocols are retained. `--reuse-support` requires an explicit
prepared manifest SHA, checks availability/role/vocabulary identity, then
rechecks every native shard and raw token identity; it retains the pinned
preparation's grammar check as provenance.
Before freezing, every noncatalog leaf must also have a measured source from
the current TRAIN scan. A digest-consistent prepared extra leaf without that
source is refused; pinned cache contents alone cannot extend availability.

Native measurements lack the seven-artifact Rust qualification closure. This
is a development audit and a source/fixture proof, not a heldout capability,
live learning or serving result. Bulk artifacts remain outside Git in
`.agent/rlfw-coord/lanes/CX-SUP/`.

The complete native scratch audit produced 2,633,926 programs (2,627,286 additions), covering all 2,829,211 measured TRAIN detail rows. On all 263 development contexts / 54 lineages, grammar summed NLL exceeds v1-trie summed NLL by 73.044906 nats on average; exact sampler/actor recomputation differs by at most 3.55e-15. The actual successor sampler emits a measured noncatalog program with finite actor likelihood. See [small pinned results summary](probability-successor-results.json) for source/input/receipt identities, operator coverage, same-support reference and conditional lineage CI. These remain SCRATCH and BLOCKED_JEPSON for qualified use.
