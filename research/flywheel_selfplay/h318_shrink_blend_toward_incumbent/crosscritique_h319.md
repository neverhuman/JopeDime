# H319 cross-critique: scope, packet guards and family-copy folds

Same-agent source review and metadata-only replay of H319 commit
`f7763b9e4973a1759e2b808c0f070819ade97bb0`. This adds no H318 decision criterion,
feature read, outcome evaluation or model fitting.

H319's published negative result supports rejecting its tested nine-member
bootstrap/voting recipe. It does not rule out every selector-level variance
reduction method or establish the cause of the dev91 tail. Its lower distinct-family
coverage, altered internal validation grouping and changed vote threshold are
several simultaneous changes; the result does not isolate any one as causal.

The [H319 source](https://github.com/neverhuman/JopeDime/blob/f7763b9e4973a1759e2b808c0f070819ade97bb0/research/flywheel_selfplay/h319_bagged_ens_vote2_members/run_h319.py)
prepares all TRAIN packets before wrapping held-out outer packets in
`GuardedPackets`. Its report honestly counts 10,971 outer packets present in
memory. Zero guarded accesses therefore does not mean zero physical outer-row
reads before locks. This is an input-boundary distinction, not evidence that
those packets influenced fitting.

A separate concrete issue occurs in `relabel`: bootstrap copies receive family
IDs `<original>#<copy>`. These IDs enter H302 `fit_and_freeze_member`, then the
pinned H287 `ensemble.family_folds`. That function permutes the new IDs into four
folds; `veto.run_cell_veto` fits on three and validates on the fourth. Copies of
one original family can therefore appear on both sides of internal validation.

[The metadata replay](crosscritique_h319.json) reconstructs H319's exact bootstrap
seeds, family draws, relabelling and pinned fold assignment using TRAIN family
identifiers only. It finds overlap in **324/324 members and 1,296/1,296 internal
splits**, with **7–27 original families** shared across each internal train/validation
split. No features, utilities, fitted models or outer scores were read by this
replay. All source-family membership remains within the respective outer training
fold; this finding does **not** establish held-out outer-family leakage.

Reproduction code and per-member overlap counts are in the H318 custody archive:
`receipts/crossreview_h319.py` and `receipts/h319-internal-fold-overlap-detail.json`.
H302's source hash is verified through H319's pinned H303 manifest; H287's fold
source is hash-verified against H302's pins. The initial auxiliary replay attempt
stopped at a manifest-key lookup before reading metadata; the corrected lookup
follows that transitive pin chain. Both attempt receipts are preserved.

Any future bagging comparison should preserve original family identity across
internal validation splits while representing bootstrap multiplicity separately,
and enforce the declared physical outer-input boundary. Such a comparison was
not run here. H318's fitted models and decision rule remain unchanged.
