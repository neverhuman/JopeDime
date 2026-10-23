# H27 spare-core probing policy

The human authorized fast parallel probing across xbabe1, xbabe2, xbabe3 and xbabe4 without disrupting competing jobs. Historical four-CPU research ceilings do not apply to later qualified probe execution. The completed isolated backend build retained its frozen two-CPU cap; scientific counts, source/input/role integrity and physical resource protections remain required. No research compute deadline applies.

The first three hosts each have 64 physical cores. The fourth is an Apple M3 Ultra with 32 cores and 512 GiB RAM and needs a native-compatible runtime before fitting. Fresh headroom favored the first and third Linux hosts. The proposed pilot uses disjoint whole-context assignments, single-thread probe children, low scheduling priority, fixed per-host CPU/RAM caps and pressure-driven pause/resume of only owned workers. Host metadata alone is not runtime or fit qualification.

Every context/probe/fold has one assigned work identity. The one-fold feature reuses fold0 from the same five-fold partition; no additional fit is needed. The planned16-context pilot has640 fits; it is included in the full850-context34000 total. Features are shared across all neural arms. Neither planned count is an executed result. No competing jobs are stopped, reprioritized or evicted.

Inputs are the exact previously qualified curated GP outer-TRAIN views. New fold transforms and target scaling fit only on that fold’s TRAIN portion. Earlier view curation is not newly certified fold-local; these probe scores are curated-view descriptors, not unbiased raw-data generalization estimates. Actual outer-TEST cells and ranking holdout outcomes do not enter feature generation.
