# Independent ridge PRESS and Schur proof

This test-only B11 follow-up checks the merged certified tools through their public API.
The existing brute-force PRESS helper refits with the same Cholesky factorization as the
fast route. `ridge_press_oracle.rs` instead builds row-major normal equations and solves
them with partial-pivoted Gaussian elimination. It does not call the production solve,
brute-force helper, leverage calculation, or PRESS identity to construct expected residuals.

The deterministic corpus contains 81 designs: widths 0–8, three column patterns (including
duplicate columns), and absolute penalties 0.25, 2, and 16. Labels have an exactly
representable full-design mean of 4. The oracle removes each row while retaining that
original mean and absolute penalty. All 1,296 deleted-row residuals and 81 PRESS totals
must agree within the existing B11 relative bound of 1e-10. The corpus explicitly requires
nonzero expected residuals; zero-valued output cannot pass by accident.

Every prefix width is also checked: 405 Schur gains against independently solved explicit
ridge objectives, with the existing objective-scaled 1e-10 bound, and 405 exact prefix
factor extensions. No golden, expected fixture, tolerance, product code, serialized model,
emitter, profile, or default changes.

Acceptance:

```sh
cargo test -p feat-math --locked --test ridge_press_oracle -- --nocapture
```

Evidence is retained under `/home/ubuntu/jope-evidence/kernel2/bt/B11_PRESS_ORACLE_PROOF/`.
This is a bounded numerical regression proof, not additional throughput, real-campaign
utility, or complete-card qualification.
