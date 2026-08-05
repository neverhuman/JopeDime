# JOPE-Dime ownership and project boundaries

`jope-dime` is the version-controlled source and control-plane authority for
regression, binary, multiclass, and time-series work. Its runtime state plane
is `/home/ubuntu/jope-flywheel/test_exact_gen0`: datasets, queues, receipts,
immutable evidence, checkpoints, and service status live there and are owned
by Dime.

`/home/ubuntu/flywheel` is a retained Priority-10 workspace and legacy/archive
boundary. Its code and services are inventoried in
`manifests/legacy-flywheel-inventory.v1.json`; they are not current JOPE-Dime
orchestration authority. New Dime work must not import from that workspace.

`/home/ubuntu/jope-slime` is an independent black-box optimization project. It
may carry prose-level, explicitly attributed lessons about safety or evidence,
but it must not consume Dime source, runtime artifacts, datasets, checkpoints,
receipts, service names, or filesystem snapshots. Run
`python3 scripts/project_boundary_check.py` before a control-plane change.

The boundary manifest is versioned at
`manifests/project-boundary.v1.json`. The checker reads tracked executable and
configuration content, tracked symlink targets, the native service manifest,
and every active unit source. Active units must execute the content-addressed
runtime bundle, use the JOPE-Flywheel runtime working directory, and pass only
bundled Python or shell assets. Prose-only lessons are permitted only beneath
the declared documentation roots.

A passing check means only that the source/runtime boundary is internally
consistent; it is not evidence that a checkpoint improved. The check remains
red while an active entrypoint depends on the pre-Dime `/home/ubuntu/jope`
checkout. That dependency must be migrated into an append-only JOPE-Flywheel
inventory rather than waived.
