# H121 target smoothing

CPU preparation and H97-backed executor implementation are complete; no scientific outcomes or checkpoint/data payload reads. The protocol is registered in [PREREG.md](PREREG.md). H90 previously tested epsilon 0.1 with no admission, so H121 is a fresh-fold replication plus epsilon 0.3 dose and H114 continuous-regret co-readout.

Synthetic proof exercises rowwise smoothing, deterministic whole-family split construction, a synthetic optimizer update, all-candidate regret aggregation, and distinct retry scratch identities. This is protocol validation only. The live prerequisite snapshot and required launch gates are in [EXECUTION_READINESS.md](EXECUTION_READINESS.md). The executor is implemented in the current candidate but is not yet merged or authorized for execution. H99/H100 terminal records are still missing, so the coordinator refuses to launch. GPU/FIFO and real-data access also wait for post-merge checks and explicit owner authorization.
