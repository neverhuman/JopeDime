# H27 full-cohort landmarking findings

All 34,000 unique GP feature fits have passed saved-model prediction, input/fold/model identity and independent q-score verification. The complete fixed-width feature set covers 850 historically exposed TRAIN contexts in 162 lineage components. The earlier 640 pilot fits were reused; the remaining 33,360 fits were performed once. There are zero unknown fits or scoring discrepancies. Verification uses the same native predictor kernel with independent input/fold/scoring reconstruction; it is not an independent predictor implementation.

One-fold and five-fold best-probe sets differ on 179 of 850 contexts (21.1%; 22.8% with equal component weights). Depth-4 boosted trees are best on 492 contexts, elastic net on 177. Raw and robust ridge q scores differ by at most 1e-8 in 4,211 of 4,250 fold pairs. The ridge backend restandardizes inputs, largely cancelling affine robust scaling. Tie-aware best counts are not mutually exclusive.

These are descriptive CV feature findings. No H27 neural models have been fitted and no improvement in neural validation loss or routing fitness has been measured. Historical stored-view curation was not newly certified as fold-local under this feature CV5 partition. These contexts are not an independent generalization cohort.

The fixed next comparison is zero, one-fold and five-fold probe features crossed with real versus zero Dope state, three fixed seeds and two component folds: 36 fresh models, 400 updates each. Gradient-held-out monitor components are excluded from gradients and fitted scalers. All fits must close before final OOF scoring. The two primary contrasts use raw chosen-GP constrained fitness, equal component weighting and 100,000 paired component bootstrap draws with adjusted intervals. Falling loss alone is insufficient.

Two fresh clean-default Dope encodings completed successfully and match the stored float32 features descriptively. Historical input-CSV/runtime parity remains unknown. A full 850 refresh was predeclared before the neural outcomes: reuse these two, then encode 848 fresh cases. The first zero-data/zero-fit code-pin failure and its later narrow correction remain separate receipts.

Full-cohort QA source: b46ab86e8781cf3c9c8dd240b55d0de5da605f9d902849756da57a1e5c7cd473. Union verification: 84771208d1c1e279fad35d7574d15995133c18004e135b8a5496a8d312c3c470. Feature file: c544d47f766b1dea941f7eb97d6d6bb035229b0a67b996ec51d280fe81d208db. Dope two-case results: a1167873088c08659a51a105b5901beb6bd97330a42f8338b33191c1c409cd66.

The full saved-model QA coordinator took 361.52 seconds, with 616.50 retained unit CPU-seconds across bounded workers. All workers and empty parent slices were stopped. Closed raw models/predictions remain on their original hosts; compact qualified features and receipts have verified Mac and 5 TB copies. The new H27 raw-results S3 archive remains pending. Earlier S3 archives do not cover these files.
