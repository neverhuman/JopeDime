# Original query selection commitment

`original_query_selection.py` creates an actual immutable selection artifact from bound metadata, then independently verifies its derivation. It performs no fitting, prediction, target access or native authority mutation. It does not grant query access.

Use the root containing the native `runs/` metadata paths. A source-freeze manifest may describe the full repository; its file map must include `parallel/tools/original_query_selection.py` with the exact bytes of the executing source version.

```sh
python3 -B tools/original_query_selection.py freeze \
  --root /home/ubuntu/jope-dime/parallel \
  --input runs/experiment/selection-request.json \
  --sha256 ORIGINAL_REQUEST_SHA256 \
  --output runs/experiment/selection.json

python3 -B tools/original_query_selection.py verify \
  --root /home/ubuntu/jope-dime/parallel \
  --input runs/experiment/selection.json \
  --sha256 RETURNED_SELECTION_SHA256
```

These commands are interfaces, not authorization to run a scientific experiment. Root owns remote testing and any actual metadata invocation. The focused remote suite is `python3 -B -m unittest test_original_query_selection` from `parallel/tools`.

## Request inputs

The strict `parallel.original_query_selection_request.v1` request contains:

| Field | Contract |
|---|---|
| `experiment_id`, `scope` | Stable nonempty task-scope labels, bound into every selection key. |
| `role` | `training_diagnostic` selects only `meta_train`; `promotion_validation` selects only `meta_validation`. Neither opens sealed/development payloads. |
| `source_release` | Exact `{path, sha256}` reference to `parallel.registry_source_freeze.v1`. Its file-map digest and this producer's source identity are checked. This does not prove deployment or a complete approved runtime. |
| `context_index` | Exact reference to native `parallel.context_index.v1`, including context/encoding receipt identities. No encoding-vector file is opened. |
| `family_inventory` | Exact reference to existing `parallel.family_contamination_inventory.v1`, bound to that context index. Preparation, family, dataset, source and fold identities must agree. Known conflicts reject; absence of listed conflicts does not prove independent family membership. |
| `evaluation` | Exact evaluator/runtime/numerical-contract/vocabulary SHA256 values and the native regression query contract: RMSE, original target units, strict finite outcomes, raw feature space. These are commitments, not evidence of approved execution. |
| `arms` | Distinct `{name, parent_checkpoint_sha256, original_parent_receipt}` entries. The original receipt is an exact metadata pointer, bound but not opened or certified. The retired shuffled arm is rejected. |
| `rng_seeds` | Distinct native-compatible unsigned evaluation seed identities. These are not inferred from recovered training seeds. |
| `contexts` | Explicit `{prepared_sha256, cut}` selections, each uniquely joined to both existing indices. |
| `programs_per_pool` | Positive bounded count selected in each requested context/arm/seed cell. This does not redefine any experiment's required count. |
| `pools` | One `{prepared_sha256, cut, arm, rng_seed, population, original_proposal_receipt}` per cell of the complete context × arm × seed grid. `population` points to native `parallel.seed_population.v1`; the original proposal receipt pointer is retained for native verification later. |

The producer reads every input once into a bounded byte/digest capture, rejects changes before publication, and selects candidates by ascending canonical program-byte SHA256. It never orders by outcome, teacher score, rank weight or a caller “verified” flag. Original candidate slots and origin labels remain in the result. Duplicate programs in one pool, including origin relabeling, reject rather than increasing the denominator. Different populations can legitimately contain the same program; each selection key includes its complete context, arm, seed, parent and original receipt references.

Unknown request/candidate fields reject. Legacy `policy_logprob` is accepted only as the existing alias for diagnostic rank weight; neither alias becomes a behavior propensity or selection probability. Both aliases in one candidate reject. No program is executed or declared native-grammar-valid by this helper.

## Publication and verification

Inputs are bounded regular JSON files beneath descriptor-pinned non-symlink directories. Raw/sealed/development payload paths, FIFOs, symlink files, duplicate JSON keys and nonfinite values reject. Publication writes complete bytes to a fresh inode and links them into a new destination; it never replaces an existing history. The output includes the full selected population, its deterministic keys, original input references and a local clock observation.

`verify` reopens the exact original inputs, checks the source producer identity and recomputes all selections. A self-rehashed artifact with altered origins, candidates, arms, seeds or provenance cannot pass against unchanged original inputs. A newly authored, internally consistent request can create a new **metadata commitment**; it cannot retroactively establish original nomination or prior publication. Local timestamps are not independent ordering evidence. The verifier rejects impossible future wall timestamps and any assertion that its local clock was independently verified, but it cannot prove historical timing.

Every successful result keeps `query_access_allowed=False`, `original_receipt_execution_verified=False` and `independent_publication_order_verified=False`. This is a reusable positive producer for metadata selection, not a positive scientific-access verifier.

## Native integration still required

The production `query::authorize` must remain held until the authority independently joins:

1. This exact selection artifact and nominated parent/source/role/family population to a durable selection event, committed and independently witnessed before the first protected outcome access. A caller timestamp or hash is insufficient.
2. Original proposal and parent receipts to their completed native execution/journal records, including the selected candidate's canonical program, context, seed and declared origin. Merely creating the referenced files does not establish membership.
3. The chosen selection key to an original successful fit completion/model and exact query rows, targets, preprocessing, feature view, query contract, task identity and original deadline. This module does not materialize `QueryInputs` or read those payloads.
4. The independent access receipt to the exact native serialized query descriptor and the current authority/witness policy, with sealed-family role enforcement, nomination/look accounting and existing access limits.

The present service deserializes sealed `QueryInputs` before `authorize`; that descriptor already contains rows and targets. Authority-owned publication of protected descriptors therefore remains necessary before claiming that all outcome access follows the selection barrier. The current authorization call does precede opening retained authority model/outcome blobs and creating a newly accepted query.

Existing query acceptance binds scope/task/access identity and the original request including its deadline. Retrieval must retain those identities and the original completed journal membership. A later transport deadline permits retrieval; it must not create a fresh query, alter its selection or repeat evaluation. No native API is changed by this module.
