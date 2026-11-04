# JAIN cell sensitivity and variable-size tables

## What the table measures

The frozen fused ranker selects one of eight GP proposals. Take the lowest cost among the other seven proposals and subtract the cost of the original winner. This gap is the winner's selection margin.

For every shown FIT cell, replace that cell with its column's median over the original FIT roster. Recompute the native high-landmark encoding and replay the frozen DOPE, R882, critics, and five saved G folds. Score the absolute change in the original winner's selection margin. Store the signed change, the winner-cost change, and the counterfactual winner separately. Feature cells and the target use the same test. The displayed inputs and target values are the original CSV values.

The ranker responds to a complete table and can capture interactions across cells. The score is a local sensitivity to this specific one-cell replacement. Exact weights, candidate programs, and FIT row rosters remain fixed during the tests.

## What the screen encodes

- Use the within-dataset percentile across 85 measured cell effects for each feature fill. Average ties and put exact zeros at the low end.
- Use the same percentile for each target cell's purple shade. Keep the target header purple and rightmost. Remove the footer and its colormap legend.
- Scale each cell effect relative to the case maximum. Average those relative effects by feature column over the 17 shown rows. Display each feature's share of the four means and color its header relative to the largest mean. The header averages follow the dataset and remain stable when the same rows shuffle.
- Keep the existing 0.28-second shuffle cadence. Move row IDs, original numeric values, percentile ranks, and score bars together.

## Input roster and proof

The original inputs were all generated at 96 rows. The constant-zero control had four all-zero feature columns. The revised, deterministically prepared tables have 64, 72, 80, 88, 104, 112, 120, 128, 144, and 160 rows. The constant control becomes a quantized staircase case. The predominantly zero sparse fixture becomes a clipped Cauchy case. The independent audit checks full CSV hashes and counts, all five columns for finite values and variation, and the 17 shown FIT rows for all-zero columns. Individual valid zero values remain in the tables.

Each case runs the native FIT and HOLDOUT split, the saved fused inference, and GP fitting again. The capture preserves the historical native binary's exact Git identity with a detached read-only checkout. The renderer uses fresh Blender plates from the new trace, fresh measured fitted-export timings, and the actual generated unfitted source packages. C++ and JavaScript share the Ridge chapter and switch badges as the file changes.

Case D's legacy fitted emitter failed a self-check on nearly equal quantile edges that sort differently after interpolation. The diagnostic helpers select the existing partition-point quantile profile with `JOPE_VISUAL_PARTITION_POINT_QUANTILES=1`. The fitted proof and its timed export use the same profile. The displayed recipe's kernel uses that profile for case D. Keep default exporter identities and the original failure receipt.

## Reproduction

1. Run `animation_capture.py prepare-varied` and the two `capture-varied` batches. Assemble both native archives and run the neural capture and strict validation.
2. Build the release diagnostic helpers on the pinned toolchain. Preserve the historical binaries and place the new binaries under the external capture's `gp-source-pp-build` directory. Run `h256_gp_source.py` with the profile enabled and validate the enriched trace. Run `round_trip_metrics.py` with the same profile and matching helper.
3. Run `neural_cell_influence.py` against that enriched trace, the exact native CSVs, FIT descriptors, program features, and critic references. Require bit-identical baselines, 850 interventions, validated delta arithmetic, and unchanged weights.
4. Prepare the emitted code with `short_recipe_source.py --partition-point-case case-d` and `multilingual_wave_source.py --schedule variable`. Check that the emitted files agree with the original selected DSLs. Record execution scope for the Wasm and CUDA targets.
5. Create the 1080p30 Blender scene with `jain_cinematic.py` and the validated trace, metrics, source proof, and dataset catalog. Render its case loops using the authorized GPU worker. Compose with `intelligence_wave.py --neural-influence <receipt>` and verify the full decode, case counts, and sample table colors and headers.

The previous cuts and measurements retain their original source and dataset bindings. Keep the large inputs, raw traces, fitted objects, and media in external custody.
