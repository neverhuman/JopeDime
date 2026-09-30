# Kernel capability matrix

This declaration is checked against the frozen grammars and available routes.

## Authority

The two frozen token samplers use registry authority; every later grammar uses the manifest. Each grammar's file hash selects the rule.

| Consumer | Authority |
| --- | --- |
| External product generators | Registry |
| Frozen token samplers | Each file's flags, checked under registry authority |
| Kernel profiles and later grammars | Manifest |

## Expression elements

| Kind | Element | Class | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |
| --- | --- | --- | --- | --- | --- | --- | --- |
| function | abs | baseline | supported | supported | planned | supported | unsupported |
| function | sqrt | baseline | supported | supported | planned | supported | unsupported |
| function | log1p | baseline | supported | supported | planned | supported | unsupported |
| function | sin | baseline | supported | supported | planned | supported | unsupported |
| function | cos | baseline | supported | supported | planned | supported | unsupported |
| function | tanh | baseline | supported | supported | planned | supported | unsupported |
| function | sigmoid | baseline | supported | supported | planned | supported | unsupported |
| function | min | baseline | supported | supported | planned | supported | unsupported |
| function | max | baseline | supported | supported | planned | supported | unsupported |
| function | clip | baseline | supported | supported | planned | supported | unsupported |
| function | step | baseline | supported | supported | planned | supported | unsupported |
| function | exp | runtime only | supported | supported | planned | supported | unsupported |
| function | log | runtime only | supported | supported | planned | supported | unsupported |
| function | asinh | runtime only | supported | supported | planned | supported | unsupported |
| function | atan | runtime only | supported | supported | planned | supported | unsupported |
| function | erf | runtime only | supported | supported | planned | supported | unsupported |
| function | sign | runtime only | supported | supported | planned | supported | unsupported |
| function | floor | runtime only | supported | supported | planned | supported | unsupported |
| function | round | runtime only | supported | supported | planned | supported | unsupported |
| variable | x | baseline | supported | supported | planned | supported | unsupported |
| variable | col_under_8 | baseline | supported | supported | planned | supported | unsupported |
| variable | col_8_plus | runtime only | supported | supported | planned | supported | unsupported |
| variable | row_mean | baseline | supported | supported | planned | supported | unsupported |
| variable | row_sum | runtime only | supported | supported | planned | supported | unsupported |
| variable | row_l1 | runtime only | supported | supported | planned | supported | unsupported |
| variable | row_l2 | baseline | supported | supported | planned | supported | unsupported |
| variable | row_min | baseline | supported | supported | planned | supported | unsupported |
| variable | row_max | baseline | supported | supported | planned | supported | unsupported |
| variable | row_std | baseline | supported | supported | planned | supported | unsupported |
| binary | + | baseline | supported | supported | planned | supported | unsupported |
| binary | - | baseline | supported | supported | planned | supported | unsupported |
| binary | * | baseline | supported | supported | planned | supported | unsupported |
| binary | / | baseline | supported | supported | planned | supported | unsupported |
| binary | ^ | runtime only | supported | supported | planned | supported | unsupported |
| unary | neg | baseline | supported | supported | planned | supported | unsupported |

## Policies

| Policy | Class | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| columns | baseline | supported | supported | planned | supported | unsupported | indices under 8; wider indices wait for Phase 4 |
| constants | baseline | supported | supported | planned | supported | unsupported | birth bins; finite values remain continuous afterward |
| negation | baseline | supported | supported | planned | supported | unsupported | a negative constant only |

## Operations

| Element | Kind | Kernel tier | V3 admission | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bag | Directive | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| bin_onehot | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| blom | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| cheb | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| corr_screen | Transform | not generated | RequiresReceipt | unsupported | unsupported | unsupported | supported | unsupported |
| count_sketch | Transform | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| derive | Transform | baseline | Absent | supported | supported | planned | supported | unsupported |
| elasticnet | Head | baseline | Absent | supported | supported | supported | supported | subset only |
| ensemble | Head | not generated | Absent | supported | supported | deferred | supported | unsupported |
| ensemble_ridge_gbm | Head | not generated | Absent | supported | supported | deferred | supported | unsupported |
| extra_trees | Head | not generated | RequiresReceipt | unsupported | unsupported | unsupported | supported | unsupported |
| fastfood_rbf | Transform | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| fft_mag | Transform | not generated | Absent | supported | supported | planned | supported | unsupported |
| gam_boost | Head | not generated | RequiresReceipt | unsupported | unsupported | unsupported | supported | unsupported |
| gbm | Head | not generated | Absent | supported | supported | planned | supported | unsupported |
| gpr | Head | baseline | Absent | supported | supported | planned | supported | unsupported |
| hinge | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| histgbm | Head | baseline | Absent | supported | supported | planned | supported | unsupported |
| histgbm2 | Head | baseline | Absent | supported | supported | planned | supported | unsupported |
| hurdle2 | Head | not generated | Absent | deferred | deferred | deferred | supported | unsupported |
| impute | Transform | not generated | RequiresReceipt | supported | supported | planned | supported | unsupported |
| input | Input | not generated | Baseline | deferred | deferred | deferred | supported | unsupported |
| krr | Head | not generated | Absent | supported | supported | deferred | supported | unsupported |
| ksmooth | Head | baseline | Absent | supported | supported | planned | supported | unsupported |
| lag | Transform | not generated | RequiresReceipt | supported | supported | planned | supported | unsupported |
| landmark_dist | Transform | not generated | Absent | supported | supported | supported | supported | subset only |
| logistic | Head | not generated | Baseline | supported | supported | supported | supported | subset only |
| logit_ridge | Head | not generated | Baseline | unsupported | unsupported | unsupported | supported | unsupported |
| map | Transform | baseline | Absent | supported | supported | planned | supported | unsupported |
| maxabs_scale | Transform | not generated | Baseline | unsupported | unsupported | unsupported | supported | unsupported |
| multiclass_diagonal_lda | Head | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| multiclass_nearest_centroid | Head | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| multiclass_ovr_logistic | Head | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| multiclass_ridge | Head | not generated | Baseline | unsupported | unsupported | unsupported | unsupported | unsupported |
| norm_features | Transform | not generated | Baseline | supported | supported | supported | supported | subset only |
| norm_row | Transform | not generated | Baseline | supported | supported | supported | supported | subset only |
| nystrom | Transform | not generated | Absent | supported | supported | planned | supported | unsupported |
| nystrom_leverage | Transform | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| oblique_hinge | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| pair_prod | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| pair_ratio | Transform | not generated | Absent | supported | supported | planned | supported | unsupported |
| piecewise_linear | Transform | not generated | RequiresReceipt | unsupported | unsupported | unsupported | supported | unsupported |
| pls | Head | not generated | RequiresReceipt | unsupported | unsupported | unsupported | supported | unsupported |
| pls_project | Transform | not generated | RequiresReceipt | unsupported | unsupported | unsupported | supported | unsupported |
| polar | Transform | not generated | Absent | supported | supported | planned | supported | unsupported |
| poly_interact | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| portfolio | Head | not generated | RequiresReceipt | supported | supported | deferred | supported | unsupported |
| power_transform | Transform | not generated | RequiresReceipt | unsupported | unsupported | unsupported | supported | unsupported |
| quantile_bins | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| rand_proj | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| rank | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| rff_cos | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| rff_halton | Transform | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| rff_orthogonal | Transform | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| rff_sin | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| ridge | Head | baseline | Baseline | supported | supported | supported | supported | subset only |
| ridge_gcv | Head | baseline | Absent | supported | supported | planned | supported | unsupported |
| robust_scale | Transform | baseline | Baseline | supported | supported | supported | supported | subset only |
| roll | Transform | not generated | RequiresReceipt | supported | supported | planned | supported | unsupported |
| row_diff_stats | Transform | not generated | Absent | supported | supported | supported | supported | subset only |
| row_spectral | Transform | not generated | Absent | supported | supported | planned | supported | unsupported |
| row_stats | Transform | baseline | Absent | supported | supported | supported | supported | subset only |
| select_corr | Transform | not generated | RequiresReceipt | supported | supported | planned | supported | unsupported |
| sigmoid | Transform | baseline | Baseline | supported | supported | supported | supported | subset only |
| spline | Transform | not generated | Absent | supported | supported | planned | supported | unsupported |
| stack | Head | not generated | Absent | supported | supported | deferred | supported | unsupported |
| standardize | Transform | baseline | Baseline | supported | supported | supported | supported | subset only |
| step | Transform | not generated | Baseline | supported | supported | supported | supported | subset only |
| svd_project | Transform | not generated | Absent | supported | supported | planned | supported | unsupported |
| symlog | Transform | not generated | Absent | supported | supported | planned | supported | unsupported |
| tab_boost | Head | not generated | RequiresReceipt | unsupported | unsupported | unsupported | supported | unsupported |
| target | Target | not generated | Baseline | supported | supported | planned | supported | unsupported |
| target_bin_encode | Transform | baseline | RequiresReceipt | supported | supported | supported | supported | subset only |
| tensor_sketch | Transform | not generated | Absent | unsupported | unsupported | unsupported | unsupported | unsupported |
| winsor_clip | Transform | baseline | Baseline | supported | supported | supported | supported | subset only |
| winsor_moments | Transform | not generated | Baseline | supported | supported | supported | supported | subset only |

## Program forms

| Form | Kernel tier | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline | baseline | supported | supported | supported | supported | subset only |
| symbolic | not generated | supported | supported | planned | supported | unsupported |
| nested_portfolio | not generated | deferred | deferred | deferred | supported | unsupported |
