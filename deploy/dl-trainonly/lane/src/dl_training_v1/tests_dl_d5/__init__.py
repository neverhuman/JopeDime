"""Offline tests for tools/p2_decisions.py version 2 (the DL-D5 owner-answer expansion on P2 generation 2, PLAN §5
P2/P3, §7.2).

Run only inside the lane slice with TMPDIR=RUN/tmp and the offline prefix (AWS_CONFIG_FILE=/dev/null):
  tmp/p2fix3/py.sh -m unittest discover -s tests_dl_d5 -t . -v
The bound g2 inputs (review.csv aad99b1a..., registry.pre-review.json e21690dd..., alias-sources.json f818efaf...,
inputs/) are read-only. Every write goes under RUN/tmp/dl-d5-g2: a process audit hook fails any test that writes
elsewhere. No test touches S3 or the network, loads an encoder, or imports s3io/boto3 into the tool's expand/check
paths.

Modules:
- test_d5_mapping: the v2 shape and its builtin pin, and dl-d5-map/v2 on the real g2 review.csv, with no build.
- test_d5_parser: the strict owner-text grammar dl-d5-confirm/v1 (D-MAP-1/2/3/7).
- test_d5_refusals: R1..R12.
- test_d5_check: check on a stand-in pre-review model, the registry and rehearsal-code gates, and the R7 code binding.
- test_d5_unpromotable: D-MAP-6, DRAFT and what-if refused by families and holdout.
- test_d5_dryrun: the rehearsal plan, its refusals, and the orchestration through stand-in entry points.
- test_d5_interface: the shared constants, the agreed owner line, and families.publish feeding holdout.check_gate.
- test_d5_simulate: the real families.run_build on g2. It runs only while the dltrain family code still equals the
  pre-review registry's 'code' block, and is otherwise skipped with the drifted module names.
"""
