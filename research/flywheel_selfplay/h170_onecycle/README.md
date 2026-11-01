# H170: one-cycle LR and inverse momentum

Hypothesis: one-cycle LR/momentum regularization improves r882 continuation at a fixed 40-update budget. CTRL, OC3, OC15 and CTRL_MEANMATCH use three seeds and five family assessment folds; only the 29,211-parameter adapter trains while DOPE remains frozen. [Preregistration](PREREG.md) fixes discrete endpoints, the .0001481125 matched mean LR, strict selection, conditional family inference, drift and pseudoreplication checks.

Status: preregistration; no H170 real-data scores or optimizer updates. Twelve data-free controls cover schedules/native AdamW application,None gradients, family exclusion/strict selection, utility identity/adapter distinction, resource aggregation and launch binding. The native 91-context baseline/raw-cost identity is required before TRAIN outcome pools or fitting. No speed metrics or serving change.

Closeout will replace this status with median/CI over all three seeds, regression utility/Elo, step 40 drift, duplicated prediction/adapter disclosures, verdict and next step. Large artifacts and source bundles go to versioned S3; Git retains small metrics and the index row.
