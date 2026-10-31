# CPU receipt status validation

The CPU pre-stage schema requires a `passed` aggregate to contain only passed
checks and a nonempty finish time. Every passed check requires exit status zero
and a nonempty finish time. These constraints enforce states the existing
controller already emits; they add no fields or release gate decisions.

Run the software controls from an existing checkout, with an evidence directory
outside the checkout:

```sh
nice -n 19 env JOPE_QUALIFICATION_TEST_TMP=/path/to/evidence/scratch \
  python3 -B deploy/qualification/v1.0/test_receipt_schema
```

The controls inventory only their own tiny stdlib child commands. They preserve
real child exits and unittest counts, then test contradictory metadata against
the schema. Planned, failed and incomplete observations remain valid; unknown
test counts remain allowed for non-test commands. No release campaign, dataset,
model, GPU or signed candidate is used. The existing controller suite provides
additional producer compatibility coverage.

This checks metadata consistency. Artifact hashes, source admission, inventory
completeness and actual command execution need their own verification. The schema
continues to require null signed candidate/owner approval, false
`release_qualified`, and unqualified release gates. CPU checks never establish
signed release qualification.
