# Policy lineage and optimizer epochs v1

I11 adds CPU-verifiable ancestry evidence and `jope.native_loop.v3`. It does not
issue a training grant, qualify native numerics, certify a release, or migrate
optimizer moments. The existing signed run approval, witness transition chain,
cumulative evaluation budget, deadlines and publication proofs still apply.

| Protocol | Initial weights | Optimizer windows | Successor |
| --- | --- | --- | --- |
| native loop v1 | accepted 60000 or completed v1 predecessor | carried 60000→63000→66000, then exact 6000 windows | unchanged v1 completion |
| native loop v2 | accepted 60000 | reset 0→3000, carry 3000→6000 | legacy readers still refuse a v1 successor |
| native loop v3 | verified completed v2/v3 descendant | carry that descendant's local epoch, e.g. 6000→9000→12000 | explicit ancestry plus both committed block edges |

The four existing continuation wire strings remain `cold_initialization`,
`weight_warm_start`, `new_corpus_transition`, and `exact_resume`. A reset starts
a distinct optimizer epoch at zero. A carry preserves that epoch and its recipe.
Exact resume is an attempt property and creates no new ancestry edge. No
architecture, vocabulary or support migration is supported by this version.

`native_loop/lineage.rs` is the sole definition of `PolicyLineage`,
`OptimizerEpoch`, `TrainingIntent`, edge/credit DTOs, and the optional signed
`DescendantLineagePolicy`. A lineage fixes an immutable anchor and scientific
protocol, and names ordered content-addressed edges. Every edge binds its
immediate weight parent, checkpoint/weights/optimizer identities, epoch, intent,
corpus, objective, recipe, architecture, vocabulary and support. Unknown anchor
objective permits only a fresh optimizer. Historical committed exposure remains
optional: a step-60000 label does not prove 60000 committed updates.
This version requires the anchor's committed count to be null: certifying a
historical count needs a separately adopted anchor-receipt protocol. Additional
descendant exposure is exact within the supplied committed receipt closure;
total exposure including the historical anchor remains unknown.
The neural continuation adapter hashes the requested partition's JSON identity
and the sampler identifier's UTF-8 bytes against the frozen intent. Optimizer
carry and exact resume require retained epoch and recipe identities as well.

Each edge needs a byte-bound committed credit receipt for every rank. Credit
binds epoch, commit, checkpoint, intent, training eligibility and the exact
update interval. Rank coverage must be complete, with one shared commit and
world identity. Repeated receipt delivery contributes nothing additional;
duplicate edges, rank credit, commit reuse within an epoch, forks and cycles
are refused. Exposure adds each logical update once across ranks. Checked
arithmetic and I02's nonnegative safe-integer range apply to every counter.

Lineages have an independently chosen maximum of 64 edges, 1 MiB per canonical
object and 8 MiB total evidence. At most 64 ranks may contribute to one edge.
The verifier checks bounds, schema, canonical bytes and domain-separated
identity on load. Its restricted typed JSON uses sorted keys and no floats;
identities follow I02's `sha256(schema || NUL || u64_be(length) || payload)`
formula. Historical checkpoint identities retain their raw-byte SHA256.

`LineageSuccessorEvidence` is an additive predecessor adapter naming the
unchanged original completion artifact and a bounded evidence bundle. A v3
configuration names this adapter in `predecessor_run`. Both 3000-update block
edges must match the preceding run's exact completion, parent, checkpoint and
corpus. Further v3 successors retain the verified ancestor prefix. The
controller also replays the original publication and acceptance proofs;
verification alone does not schedule work. Old config/completion fields and
absent optional fields retain their serialization.

Witness opt-in is a signed policy decision. The coordinator must add
`#[serde(default, skip_serializing_if = "Option::is_none")]`
`descendant_lineage_policy: Option<DescendantLineagePolicy>` to `WitnessPolicy`,
update its constructors, and add `../native_loop/lineage.rs` to **both** source
digest sets with the label `native_loop/lineage.rs`. The worker's epochs adapter
reads that field only from the policy inside the existing authenticated
transition chain. Until the coordinator hunk lands, the strict policy reader
rejects the field and production opt-in remains unavailable. An absent field,
including after valid source/executable rotation, keeps the legacy initial
60000 and exact 6000 successor rules. Same-window recovery cannot change the
opt-in; a subsequent transition cannot strip it. Separate completion, original
journal, sequence fencing, authority custody, expiry and cumulative budget
checks remain in force.

The signed opt-in fixes the run, science, separate completion hash and complete
lineage bundle. A 3000-step descendant can satisfy lineage shape, while a
native two-block successor still requires the completed 6000-step window.
Random 70000, numeric relabeling, absent/altered receipts, stale intents,
objective/recipe changes, overflow and reused credit fail closed.

I04's draft training-eligibility projection requires purpose `training`, role
set `{train}`, matching intent/corpus, and an exposure-closure digest. The
integrator must bind it to I04's accepted role/exposure verifier; hash equality
alone does not prove authority. I12 still owns executable intent freeze. I13
still owns complete atomic checkpoint/consumption binding: the committed-credit
projection is evidence input, not a replacement checkpoint commit protocol.
The neural ancestry bridge requires retained weight and optimizer bytes plus
manifest epoch/objective/corpus bindings and refuses missing bindings. It does
not upgrade the old ancestry report's possible exposure into committed credit.

Rollback disables v3 and the optional signed policy. Preserve original readers,
completions, journals, accepted parent, failed receipts and unknown history.
Real owner-signed adoption and native/GPU qualification remain separate holds.
