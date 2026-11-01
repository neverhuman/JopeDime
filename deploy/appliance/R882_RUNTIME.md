# Internal r882 proposal runtime (REL-07 t0250)

The full profile now names `proposal.r882.regression`, the explicit internal v1
Engine path selected by `jopedime run --guidance learned --landmarks full` for
numeric regression. The previous `proposal.step60000.regression` promise named a
different generative policy and could not be implemented using r882 ranker bytes.
It remains uncompiled; there is no alias or substitution for that policy.

Ordinary in-process and daemon engines authenticate their own executable, read the
complete embedded r882 closure, independently check its release signature and
frozen semantic/tensor catalog, and install `AdmittedRanker::r882`. They then use
the existing discovery, proposal admission, fold evaluation, native final refit,
persisted model, export and prediction paths. No numerical algorithm is replaced.
The daemon retains that same activity/ranker object when opening its listener.
The embedded bridge uses the existing Unix-only appliance asset reader. Its
dependency is target gated so standalone Windows service builds retain their
existing graph; this change does not qualify a Windows embedded appliance.

The closure is the existing RNK-10 export's 56 files, under `model/r882/`, including
DOPE, JOPE, feature metadata, bank, golden tensors, semantic manifest and release
catalog. `models/bundles/r882.json` is the existing export descriptor, imported
without changing its hashes or internal rights. The six legacy foundation assets
and their restricted-rights acknowledgement stay in the full profile.

Profile locking refuses absent or differently bound required assets. Normal full
startup checks every required decoded size/hash before allocating model assets.
Normal full
packing also probes the supplied executable: its method inventory must contain
the compiled r882 adapter, and it must report an embedded Engine,
no development-key trust, and the requested exact source. This probe requires a
host capable of executing the target binary. Inspection packing is unchanged.
`methods` distinguishes a compiled adapter from an executable authenticated
closure; `doctor` refuses missing/broken promised capabilities. An integrity
failure stops startup instead of selecting heuristic guidance.

The alternative of flipping the old method's flags would misstate both execution
and lineage. Reimplementing the discovery algorithm would duplicate its numerical
contracts. This change instead supplies the missing production loader/startup
bridge. It changes no API, worker, bundle or fitted-state schema.

Authentication and an unsigned candidate stage are not release qualification.
Auto remains quality gated; legacy `fit --method` retains its native registry.
The full signed-candidate integration, 28 gates, M2/M3/M5/S4, Q1 and publication
remain separate. The preexisting signed internal asset package can test the new
embedded loader, but its empty method list cannot enable the new full profile.
