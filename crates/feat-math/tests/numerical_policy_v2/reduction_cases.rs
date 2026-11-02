use super::*;

#[test]
fn asinh_repairs_the_huge_finite_input_only_in_the_new_kernel() {
    let (value, faults) = asinh_v2(1.0e200);
    let expected = 461.2101657793691_f64;
    assert!(value.to_bits().abs_diff(expected.to_bits()) <= 2);
    assert_eq!(faults, FaultSet::EMPTY);
    let old = MathExpr::parse("asinh(c0)").unwrap();
    assert_eq!(old.eval_row(&[1.0e200]).unwrap().to_bits(), 0);
    for input in [0.0_f64, -0.0, f64::from_bits(1), -f64::from_bits(1)] {
        assert_eq!(asinh_v2(input), (input, FaultSet::EMPTY));
        assert_eq!(asinh_v2(input).0.to_bits(), input.to_bits());
    }
}

#[test]
fn neumaier_preserves_the_cancellation_term_in_input_order() {
    let row = [1.0e16, 1.0, -1.0e16];
    assert_eq!(reduce_v2(Reduction::Sum, &row), (1.0, FaultSet::EMPTY));
    assert_eq!(
        reduce_v2(Reduction::Mean, &row),
        (1.0 / 3.0, FaultSet::EMPTY)
    );
    assert_eq!(
        MathExpr::parse("row_sum")
            .unwrap()
            .eval_row(&row)
            .unwrap()
            .to_bits(),
        0
    );
}

#[test]
fn finite_extreme_means_clamp_at_non_power_of_two_widths() {
    for count in 2..=32 {
        for sign in [1.0_f64, -1.0] {
            let row = vec![sign * f64::MAX; count];
            let (value, faults) = reduce_v2(Reduction::Mean, &row);
            assert_eq!(
                value.to_bits(),
                (sign * 1.0e12).to_bits(),
                "width {count}, sign {sign}"
            );
            assert_eq!(
                faults,
                FaultSet::MAGNITUDE_CLAMPED,
                "width {count}, sign {sign}"
            );
        }
    }
}

#[test]
fn overflowing_mean_preserves_exact_cancellation_and_small_residuals() {
    let maximum = f64::MAX;
    let previous = f64::from_bits(maximum.to_bits() - 1);
    let gap = maximum - previous;
    for sign in [1.0_f64, -1.0] {
        for residual in [0.0, 1.0, f64::MIN_POSITIVE, f64::from_bits(7)] {
            // The first five binary-rational values sum to exactly zero.
            let mut row = vec![maximum, maximum, -previous, -previous, -2.0 * gap];
            if residual != 0.0 {
                row.push(residual);
            }
            for value in &mut row {
                *value *= sign;
            }
            let expected = if residual == 0.0 {
                0.0
            } else {
                (sign * residual) / row.len() as f64
            };
            let (value, faults) = reduce_v2(Reduction::Mean, &row);
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "sign {sign}, residual {residual:e}"
            );
            assert_eq!(
                faults,
                FaultSet::EMPTY,
                "sign {sign}, residual {residual:e}"
            );
        }
    }
}

pub(super) struct MeanProbe {
    pub(super) row: Vec<f64>,
    pub(super) expected: f64,
    pub(super) faults: FaultSet,
}

pub(super) fn mean_properties() -> Vec<MeanProbe> {
    let mut probes = Vec::new();
    let mut seed = 0xb0703_u64;
    for width in 1..=128 {
        for sign in [1.0_f64, -1.0] {
            let mut families = vec![
                (
                    vec![sign * f64::MAX; width],
                    sign * 1e12,
                    FaultSet::MAGNITUDE_CLAMPED,
                ),
                (
                    vec![sign * f64::MIN_POSITIVE; width],
                    sign * f64::MIN_POSITIVE,
                    FaultSet::EMPTY,
                ),
                (
                    vec![sign * f64::from_bits(7); width],
                    sign * f64::from_bits(7),
                    FaultSet::EMPTY,
                ),
                (vec![sign * 1e12; width], sign * 1e12, FaultSet::EMPTY),
                (
                    vec![sign * f64::from_bits(1e12_f64.to_bits() + 1); width],
                    sign * 1e12,
                    FaultSet::MAGNITUDE_CLAMPED,
                ),
            ];
            let mut paired = vec![0.0; width];
            for pair in paired.chunks_exact_mut(2) {
                pair[0] = sign * f64::MAX;
                pair[1] = -sign * f64::MAX;
            }
            families.push((paired, 0.0, FaultSet::EMPTY));
            if width >= 3 {
                let mut mixed = vec![sign; width];
                mixed[0] = sign * f64::MAX;
                mixed[1] = -sign * f64::MAX;
                families.push((
                    mixed,
                    sign * (width - 2) as f64 / width as f64,
                    FaultSet::EMPTY,
                ));
            }
            if width >= 8 {
                for big in [f64::MAX, f64::from_bits(0x7e70_0000_0000_0000)] {
                    for residual in [0.0, 1.0, f64::MIN_POSITIVE, f64::from_bits(7)] {
                        if width == 8 && residual != 0.0 {
                            continue;
                        }
                        let a = f64::from_bits(0x5f30_0000_0000_0000);
                        let b = f64::from_bits(0x71f0_0000_0000_0000);
                        let mut row = vec![big, big, a, b, -b, -a, -big, -big];
                        row.resize(width, 0.0);
                        if residual != 0.0 {
                            row[8] = residual;
                        }
                        for x in &mut row {
                            *x *= sign;
                        }
                        let expected = if residual == 0.0 {
                            0.0
                        } else {
                            sign * residual / width as f64
                        };
                        families.push((row, expected, FaultSet::EMPTY));
                    }
                }
            }
            for (row, expected, faults) in families {
                probes.push(MeanProbe {
                    row: row.clone(),
                    expected,
                    faults,
                });
                for _ in 0..2 {
                    let mut permuted = row.clone();
                    for index in (1..width).rev() {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        permuted.swap(index, (seed as usize) % (index + 1));
                    }
                    probes.push(MeanProbe {
                        row: permuted,
                        expected,
                        faults,
                    });
                }
            }
        }
    }
    assert_eq!(probes.len(), 11136);
    probes
}

#[test]
fn mean_is_overflow_safe_across_adversarial_families_at_every_width() {
    for probe in mean_properties() {
        let (actual, faults) = reduce_v2(Reduction::Mean, &probe.row);
        check_oracle(
            actual,
            probe.expected,
            &format!("mean property {:?}", probe.row),
        );
        assert_eq!(faults, probe.faults, "mean property {:?}", probe.row);
    }
}

#[test]
fn sum_retains_nested_finite_cancellation_across_widths_and_orders() {
    for width in 6..=128 {
        for exponent in [120_u64, 500, 1000] {
            let big = f64::from_bits((exponent + 1023) << 52);
            let small = f64::from_bits((exponent / 2 + 1023) << 52);
            for residual in [0.0, 1.0, f64::MIN_POSITIVE, f64::from_bits(7)] {
                if width == 6 && residual != 0.0 {
                    continue;
                }
                for sign in [1.0, -1.0] {
                    let mut row = vec![big, 1.0, small, -small, -1.0, -big];
                    row.resize(width, 0.0);
                    if width > 6 {
                        row[6] = residual;
                    }
                    for value in &mut row {
                        *value *= sign;
                    }
                    for _ in 0..3 {
                        let (actual, faults) = reduce_v2(Reduction::Sum, &row);
                        let expected = if residual == 0.0 {
                            0.0
                        } else {
                            sign * residual
                        };
                        check_oracle(actual, expected, &format!("nested sum {row:?}"));
                        assert_eq!(faults, FaultSet::EMPTY);
                        row.rotate_left(1);
                        row.reverse();
                    }
                }
            }
        }
    }
}

#[test]
fn l2_retains_small_squares_across_widths_and_orders() {
    for width in (1..=128).chain([8191, 8192, 8193, 16383, 16384, 16385]) {
        let expected = f64::from_bits(1.0_f64.to_bits() + (width as u64 - 1 + 3) / 8);
        for sign in [1.0, -1.0] {
            let mut row = vec![sign * 2.0_f64.powi(-27); width];
            row[0] = sign;
            for _ in 0..3 {
                let (actual, faults) = reduce_v2(Reduction::L2, &row);
                check_oracle(actual, expected, &format!("small square width {width}"));
                assert_eq!(faults, FaultSet::EMPTY);
                row.rotate_left(1);
                row.reverse();
            }
        }
    }
}

#[test]
fn reduction_clamp_faults_preserve_exact_boundary_tails() {
    for tail in [2.0_f64.powi(-15), 1.0, f64::MIN_POSITIVE, f64::from_bits(1)] {
        for sign in [1.0, -1.0] {
            for width in 2..=128 {
                let mut row = vec![sign * 1e12, sign * tail];
                row.resize(width, 0.0);
                for reduction in [Reduction::Sum, Reduction::L1, Reduction::L2] {
                    let expected = if reduction == Reduction::Sum {
                        sign * 1e12
                    } else {
                        1e12
                    };
                    assert_eq!(
                        reduce_v2(reduction, &row),
                        (expected, FaultSet::MAGNITUDE_CLAMPED)
                    );
                }
                row[1] = -sign * tail;
                assert_eq!(reduce_v2(Reduction::Sum, &row).1, FaultSet::EMPTY);
            }
        }
        assert_eq!(
            reduce_v2(Reduction::L2, &[6e11, 8e11, tail]),
            (1e12, FaultSet::MAGNITUDE_CLAMPED)
        );
        let inside = f64::from_bits(1e12_f64.to_bits() - 1);
        assert_eq!(reduce_v2(Reduction::L2, &[inside, tail]).1, FaultSet::EMPTY);
    }
}

#[test]
fn scaled_l2_saturates_huge_finite_norms_and_retains_tiny_norms() {
    assert_eq!(
        reduce_v2(Reduction::L2, &[3.0e200, 4.0e200]),
        (1.0e12, FaultSet::MAGNITUDE_CLAMPED)
    );
    let (value, faults) = reduce_v2(Reduction::L2, &[3.0e-200, 4.0e-200]);
    assert!(value.to_bits().abs_diff(5.0e-200_f64.to_bits()) <= 2);
    assert_eq!(faults, FaultSet::EMPTY);
    assert_eq!(
        reduce_v2(Reduction::L2, &[f64::MAX, f64::MAX]),
        (1.0e12, FaultSet::MAGNITUDE_CLAMPED)
    );
}

#[test]
fn nonfinite_reads_have_a_positive_zero_domain_fallback() {
    for input in [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8000000000042),
        f64::from_bits(0xfff8000000000042),
    ] {
        let result = asinh_v2(input);
        assert_eq!(result.0.to_bits(), 0);
        assert_eq!(result.1, FaultSet::NONFINITE_INPUT);
        for reduction in [
            Reduction::Sum,
            Reduction::L1,
            Reduction::Mean,
            Reduction::L2,
        ] {
            let result = reduce_v2(reduction, &[1.0, input, 2.0]);
            assert_eq!(result.0.to_bits(), 0);
            assert_eq!(result.1, FaultSet::NONFINITE_INPUT);
        }
    }
}

#[test]
fn empty_reductions_keep_their_defined_identity_or_fault() {
    for reduction in [Reduction::Sum, Reduction::L1, Reduction::L2] {
        assert_eq!(reduce_v2(reduction, &[]), (0.0, FaultSet::EMPTY));
    }
    assert_eq!(
        reduce_v2(Reduction::Mean, &[]),
        (0.0, FaultSet::EMPTY_ROW_REDUCTION)
    );
    assert_eq!(
        reduce_v2(Reduction::L1, &[-3.0e200, 4.0e200]),
        (1.0e12, FaultSet::MAGNITUDE_CLAMPED)
    );
}
