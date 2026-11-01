# RNK-10 H-wave one-shot packet scaffold

This metadata-only scaffold packages a finalized H155, H156 or H157 candidate
for independent review before a separate one-shot evaluation. It requires a
complete result and passing audit, zero retries, at least one optimizer update,
frozen DOPE c51a56d0, a preregistered DL-Train inner selection, and
family-grouped Elo strictly above the frozen r882 reference. Speed is never an
admission gate. The normalized result must also include a passing access audit
with zero sealed, holdout, outer-test and meta-test reads.

The input is a publisher-normalized `jope.rnk10.hwave-candidate-result/v1`
record, including an independently audited `FROZEN_WINNER` record and the digest of the complete considered-result inventory. The winner rule is highest eligible family-grouped Elo vs r882, with lexical wave and candidate id as exact-tie breakers. The builder reads JSON metadata only; it does not read candidate bytes,
training/evaluation pools, terminal outcomes, S3, or protected data. It does not
run scoring or choose among outcomes adaptively. The result packet binds the
candidate, source, result, audit, protocol, evaluator, roster and evaluation
identity hashes. It creates an `UNUSED` zero-read ledger shell and always writes
`launch_authorized: false`.

Before invocation, an independent publisher must normalize the *final audited*
wave result under the schema and selection rule. The candidate result's source
hash and checkpoint hash identify bytes, but this tool intentionally does not
fetch or load those bytes. Fill evaluator, roster and evaluation identity pins
only from the separately qualified frozen evaluator contract. A ready packet
still requires independent HWV review, one-use owner authorization and a fresh
resource/host-capacity check before any outcome request. H-wave inner selection
is not itself fresh acceptance evidence or promotion.

Run the data-free unit checks with:

```sh
python3 -m unittest research.flywheel_selfplay.rnk10_hwave_packet.test_prepare -v
```
