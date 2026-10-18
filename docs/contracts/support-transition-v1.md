# Support transition and behavior law v1

This contract keeps the deployed CLOVER byte-piece model and the native grammar policy as
separate lineages. CLOVER uses its pinned 282-token vocabulary, sequence cap 3–640 and
EOS-inclusive support trie. Native decoding uses a byte-bound 401-token vocabulary and a
maximum sequence of 128 tokens. Neither lineage may convert token IDs by matching vocabulary
cardinality or labels.

## Support extension

`FrozenSupport` remains immutable and nonqualifying (`qualified=false`, G12 blocked). A
`SupportTransition` binds parent and successor support digests and versions. A novel leaf is
admitted only when its canonical program is representable in the parent's vocabulary, its
context is already TRAIN in the frozen role binding, execution/parity receipts bind the
candidate program digest, and the I04 issuer verifier accepts the TRAIN evidence. The
transition pins admission-evidence and admitted-program digests. A refusal retains a digest
and reason without changing support. Admission grants action availability only; it grants no
outcome or label eligibility.

The I04 verifier is an integration seam: a production caller must supply the authenticated
capability implementation. Receipt hashes and role labels alone are not authentication.

## Behavior law

Native sampling and scoring share legal-token mask, positive finite temperature, the sampler's
f32-to-f64 conversion and stable softmax weights. Sequence propensity sums token log
probabilities through EOS. The law records vocabulary identity, EOS inclusion, mixture branch,
fallback/rejection and whether it describes a conditional draw or handed-off selection. A
top-k, rejection, surplus or dedup selection law cannot be inferred from the conditional token
draw law.

Unknown or unreconstructable propensity is represented explicitly and cannot support an
unbiased on-policy or off-policy claim. Continuous densities name their base measure; discrete
token probabilities remain separate. Neither a parent GP seed density nor a Thompson draw
probability substitutes for the child's action probability.

## Compatibility matrix

| Property | CLOVER support | Native policy |
| --- | --- | --- |
| Vocabulary | Pinned 282 byte-piece table | Byte-bound 401-token grammar |
| Sequence bound | 3–640 | 128 |
| EOS scoring | Teacher forcing includes EOS | Grammar sampler and scorer include EOS |
| Support identity | Frozen support SHA and immutable successor | Vocabulary SHA and explicit behavior law |
| Unknown selection law | No unbiasedness authority | Typed `Propensity::Unknown`; no unbiasedness authority |

This fixture contract does not qualify optimizer updates, checkpoint/export/reload, serving,
promotion, live execution custody or scientific benefit.
