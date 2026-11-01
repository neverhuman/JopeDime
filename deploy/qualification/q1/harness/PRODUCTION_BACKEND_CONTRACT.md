# Q1 production engine receipt contract

Version: `jope.q1-engine-contract/v1`

This contract binds the metadata-only `jope.q1-request/v1` emitted by `q1.py`
to one production-engine evaluation and its durable per-request receipt. It
does not authorize a Q1 run, read sealed or holdout rows, or make a reference
backend a production backend.

## Request-to-dataset binding

The request's `dataset.id`, `dataset.dataset_sha256`, `dataset.cut_sha256`,
`dataset.task`, `dataset.rows`, and `dataset.features` are exact identity
constraints. The engine resolves the full dataset SHA-256 and cut SHA-256
through the dataset/cut manifest selected by the request's pinned
`pins.protocol_sha256`. The manifest supplies the canonical dataset object,
target column, ignored columns, and deterministic train-fold partition. The
engine MUST verify all request metadata against that manifest and the loaded
object before evaluation. Prefix IDs are only lookup hints; they never replace
the full SHA-256 check. No target, split, row count, feature count, or cut is
inferred from a dataset profile, a suggested target, or a caller default.

The engine receives the original closed request unchanged. Dataset resolution
is an engine-side operation under the pinned protocol, not additional caller
fields that could escape `q1.py`'s closed request schema. An unknown, ambiguous,
or mismatched mapping is refused before charging an attempt. Qualification
requests contain no payload; the engine uses only its authorized training
partition and never opens the holdout group.

## Engine operation and receipt

The engine accepts one request per evaluation and durably records one
`jope.q1-evaluation-receipt/v1` before returning it. A receipt contains:

```json
{
  "schema": "jope.q1-evaluation-receipt/v1",
  "request_sha256": "<sha256 of canonical original q1 request>",
  "identity": {
    "source_commit": "<40 hex>",
    "plan_sha256": "<sha256>",
    "pins": {"<same closed pin names and values as request>": "<sha256>"},
    "dataset_id": "<16 hex>",
    "dataset_sha256": "<sha256>",
    "cut_sha256": "<sha256>",
    "guidance": "learned or heuristic",
    "metric": "rmse",
    "seed": 0,
    "requested_budget": 24,
    "dataset_binding_sha256": "<sha256 of resolved manifest entry>"
  },
  "charged": 24,
  "attempts": [
    {
      "attempt": 0,
      "outcome": "completed",
      "fold_scores": [0.0, 0.0],
      "loss": 0.0,
      "program_sha256": "<sha256>"
    },
    {
      "attempt": 1,
      "outcome": "failed",
      "fold_scores": [],
      "failure": {"code": "<stable-code>", "reason": "<nonempty detail>"}
    }
  ],
  "champion_program": "<nonempty engine program identity or null if none>",
  "adapter": "<nonempty engine adapter identity>"
}
```

`charged` is the count the engine actually admitted and evaluated, not the
requested budget. `attempts` has exactly one entry for each charged attempt,
with unique contiguous zero-based IDs. A completed attempt has at least two
finite, non-negative RMSE fold scores, a finite non-negative aggregate RMSE,
and a program SHA-256. A failed attempt has no scores/loss/program and has a
typed nonempty failure. Unknown/interrupted attempts are never relabeled as
failed or completed; the backend refuses a nonterminal receipt. The engine
receipt is append-only and its returned bytes must be identical to the durable
receipt bytes retained for audit.

The engine operation is `POST /v1/q1/evaluations` with the original request as
the JSON body. The successful response is the receipt above. The endpoint is
available only on the qualified local production engine; transport, identity,
manifest, capability, and nonterminal-result errors fail closed. The endpoint
must resolve the requested guidance exactly; it must not downgrade learned to
heuristic or report a policy different from the request.

## Adapter projection

`production_backend.py` validates the receipt against the canonical request
digest, every request identity and pin, the exact requested policy/seed/budget,
the resolved mapping identity, and the attempt inventory. It then projects to
the existing closed `jope.q1-response/v1`:

- `charged` is the receipt's actual charged count;
- `completed` is the number of `completed` attempt records;
- `fold_scores_completed` is the sum of their actual fold-score lengths;
- `candidate_failures` is derived one-for-one from actual failed attempt
  records, retaining each attempt number and typed reason;
- `loss` is the lowest actual completed-attempt aggregate RMSE;
- `champion_program` is the program identity on the lowest-loss completed
  attempt (or `null` for an all-failed receipt, which produces status `failed`);
- `adapter` is the engine's reported adapter identity.

The adapter emits no response when receipt validation fails. The outer Q1
runner's existing `response_valid` remains the final closed-schema check.

## Qualification boundary

Unit tests may use synthetic receipts solely to test validation and projection;
they are not execution evidence. Production qualification still requires both
authorized policies to execute real, non-sealed development data with matching
identities and charged budgets, plus exact-head M2/M3 proof references. The
candidate remains r782 and DOPE remains frozen at c51a56d0. No sealed/holdout
access, owner signing, or Q1 launch is part of this adapter slice.
