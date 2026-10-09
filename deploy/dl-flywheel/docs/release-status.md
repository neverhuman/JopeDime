# Release candidate status and provenance

Source assembly on 2026-10-09 UTC for the user-requested major JopeDime release. Current repository workspace version is0.2.0. This PR prepares v1.0; it does not change every Cargo crate version or create a release tag before full qualification.

Canonical source custody: `/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-20261009-r4`, manifest SHA256 `738273c9737b343d3a6a6c375a0c486aa7fdc778238753cdbf6398b1896d6db8`. This immutable packet retains source, tests and dated evidence, including failed attempts. Product source is organized here; custody is not substituted for a reviewed PR.

Completed qualification:40 cumulative optimizer updates,5 epochs,75 presentations,13 unique training and3 validation data runs. Final fixed-panel losses0.847004/0.923811.115 program proposals,28 new executed and published results,87 cache hits,0 failures. The source-aligned checkpoint/AdamW/RNG and publication receipts are verified separately from unit tests.

The first current policy has consumed zero original native data runs. The original inventory is14,987 exact views across3,003 dataset hashes;12,581 views are eligible.7,120 published selected finals and4,581 published terminal records are distinct.880 final records await publication. New138 augmented variants are a separate pool. These counts are dated metadata observations; see the machine-readable inventory for provenance and limitations.

Outstanding release gates: full original corpus join/admission and actual optimizer exposure; asynchronous durable GP queue integration and bounded throughput qualification; original all-program source parsing/support audit; exact-head repository CI; independent eligible native PR approval; merge/readback; final serving qualification. Missing gates remain explicit.

Existing Grok research runs and source lanes are preserved. Their historical HWM outcome models are not the same model as this program-generating policy.

The portable package passed226 offline tests on xbabe2 after updating the Kumo component path-binding contract to the actual published phase-relative schema. The CI Python lane now calls the same full suite with an exact226-case count and rejects skipped cases. Hosted exact-head CI remains a separate gate. The live dashboard includes xbabe1, xbabe2 and xbabe3; foreign Grok model progress is not merged into this policy.
