# REL-07 M3 fixture preflight and source parity

M3's §14 release criterion spans historical program features, full-recipe landmark extraction, the R0 `full_train_cv` evaluator and comparable proposer frequencies. The provider runs eight exact source assertions for the available 1,309-program feature/composition slice, `proj16`, real-parent determinism/language filtering and a generated-only proposer diagnostic. Each must execute once with no skips/failures; the full candidate/source binding is verified around every command. The diagnostic explicitly reports a non-comparable population. These controls remain unchanged and do not produce a release PASS: the four complete scientific comparisons are still missing.

The new candidate-free command checks fixture custody before a candidate or Cargo build is available:

```sh
python3 -B deploy/qualification/v1.0/providers/m3.py --preflight-inputs \
  --store ABSOLUTE_CAS_DIRECTORY --catalog-sha256 FULL_SHA256
```

It verifies the caller-pinned `jope.golden-custody/v1` catalog and every referenced object. It rejects symlink traversal, non-files, unsafe logical names, invalid hashes/sizes/counts, duplicate JSON keys, contradictory duplicate-object lengths and missing/corrupt bytes. Totals include the catalog itself; shared object hashes are verified once and deduplicated independently from logical file counts. The output preserves declared reconstruction scope and original-state recovery separately, reports zero tests executed, and lists the unchanged missing scientific comparisons. It neither invokes Cargo/a candidate/an evaluator/GPU nor loads weights or writes the gate result. A successful integrity check reports `INPUTS_VERIFIED_NOT_QUALIFIED`, never the gate-execution schema or PASS. Future consumers must recheck inputs at use time; preflight does not freeze mutable storage.

The real accepted R5 reconstruction demonstrates the path: 1,024 evaluated records, 405 parent records, 1,137 unique definitions; 2,688 unlogged generated items excluded. Its scoped `historical_corpus_complete` field does not establish original pre-proposal recovery or global M3 parity. Its catalog plus payloads total 655 logical files/43,934,017 bytes, represented by 591 unique custody objects. The source provider's scientific refusal is preserved.

Alternatives rejected: trusting a prior qualification boolean, treating scoped reconstruction completeness as global parity, or forwarding more coordinator environment variables. This slice changes no coordinator, production code, frozen weights, signing, API or release-result schema.

Controls run with `python3 -B deploy/qualification/v1.0/test_provider_m3 -v` and an external `JOPE_QUALIFICATION_TEST_TMP`. Data-free fixtures exercise integrity/refusal/CLI behavior; a real complete R5 custody replay separately checks the positive path. Neither supplies numerical or signed-candidate release qualification. Without arguments, the existing five-binding candidate provider path is retained.
