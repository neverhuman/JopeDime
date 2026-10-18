# Proposal delivery v1 (draft worker contract)

This contract adds bounded delivery state over immutable draw references. It does not replace or
compact the I05 draw ledger. Every bundle retains its original ledger digest, draw IDs, slot-record
digests, canonical draw digests, multiplicities and denominators. Equal canonical draws remain
separate logical rows.

## Identity and authority

The physical request key is SHA256 of the canonical I05/I16 draft identity tuple: evaluation
context, recipe, fidelity, source, original execution-window digest, reservation ID and original
deadline. It deliberately excludes draw IDs and transport delivery count. Different contexts or
reservations produce different keys. Insertion requires an `AcceptedProposalInputs` verifier from
the coordinator to validate the accepted request/reservation and every draw reference against the
existing ledger on admission and startup recovery. Reusing a key with different immutable bundle
bytes is a visible conflict.

The worker namespace is `$JOPE_STATE/proposals/delivery-v1` as a draft allocation pending the
integrator's namespace confirmation. I05 supplies accepted draw/request IDs and I16 supplies the
original window/reservation fence. No alternate executor, reservation issuer, trainer, store
service or update-credit authority is created here.

## Durable lifecycle

One exclusive advisory lock protects a bounded request index. A proposal record is create-only and
fsynced before it is visible in memory; status replacement is atomic and fsynced through the
existing learning ledger helper. Lease generation, owner and expiry are persisted. Dispatch start
is persisted before the coordinator-supplied release hook runs. A started attempt with a lost
response remains unresolved even after lease expiry; another fit cannot begin until terminal/reap
proof resolves it. A terminal retry creates another bounded physical attempt and retains the cost
of every earlier attempt. Each terminal cost acknowledgement has a canonical content digest bound
to the immutable proposal record and lease generation.

A successful remote upload followed by a lost acknowledgement is recovered by reading and
verifying the same content-addressed remote bytes and resolving the same request identity. That
recovery never invokes the release hook again. The adapter's terminal record is not a numerical
optimizer commit receipt. Physical cost remains separate from committed update exposure.

## Capacity and loss behavior

`PendingLimits::DRAFT` bounds total requests, pending requests, total durable bytes, startup index
rows, draws per request, physical attempts, individual records and status size. The byte estimate
includes namespace and row index overhead plus any interrupted atomic-write temporary files. New
admission checks all arithmetic and reserves status headroom for terminalization. At a cap, admission fails closed
without deleting prior draws, attempts, cost or evidence. These values are prospective worker
defaults and require integrator admission before production use; they are not borrowed from the
event outbox's limits.

An unknown start is not treated as zero cost or safe to retry. The coordinator must pass a
`TerminalProofVerifier` that authenticates a terminal/reap receipt before the state can become
retryable. No API in this module commits optimizer credit or lets a stale lease adopt another
attempt's exposure.

## Integration boundary

`receiving::FencedRelease` is the draft pipeline-owned hook. The integrator owns its declaration
and call at `ExecutionPool::run`, the CLI driver, schema/coverage/inventory updates and actual
production wiring. This lane's fake-release tests prove adapter call-count behavior only; they do
not prove a live fit or production integration.
