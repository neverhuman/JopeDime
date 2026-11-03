use super::*;

/// Review note PR0-O-WINNER-TEST: the independent reviewer's same-winner test (review 05),
/// adopted with its name and intent. The committed fixture's tied pairs always share a
/// program, so a count of tied scores would also pass it; here a tie with different programs
/// is not the same winner, and a win with the same program is.
#[test]
pub(crate) fn review05_same_winner_means_program_identity_including_score_ties() {
    let runs = [
        paired_run("A", "d0", 1, 1.0, "p"),
        paired_run("B", "d0", 1, 1.0, "q"),
        paired_run("A", "d0", 2, 2.0, "p"),
        paired_run("B", "d0", 2, 1.5, "p"),
    ];
    let result = paired(&["d0".to_owned()], &runs).unwrap();
    assert_eq!(
        (
            result.overall.wins,
            result.overall.ties,
            result.overall.same_winner
        ),
        (1, 1, 1)
    );
    let tied = paired(&["d0".to_owned()], &runs[..2]).unwrap();
    assert_eq!(tied.overall.same_winner, 0);
    let same = paired(&["d0".to_owned()], &runs[2..]).unwrap();
    assert_eq!(same.overall.same_winner, 1);
}

/// Review note PR0-O-SIGN-UNDERFLOW: a unanimous split of `n` pairs has the exact two-sided
/// p `2^(1-n)`. It is returned exactly at `n = 1074` (`2^-1073`) and at `n = 1075`
/// (`2^-1074`, the smallest positive f64), and is 0 at `n = 1076`, where the exact `2^-1075`
/// is half the smallest positive f64 and rounds to even. Also the review's independent
/// integer oracle for every split up to 120 pairs.
#[test]
pub(crate) fn ablation_sign_test_is_exact_down_to_the_smallest_positive_f64() {
    for (n, expected) in [
        (1074, f64::from_bits(2)),
        (1075, f64::from_bits(1)),
        (1076, 0.0),
    ] {
        assert_eq!(sign_test(n, 0).to_bits(), expected.to_bits(), "n {n}");
        assert_eq!(sign_test(0, n).to_bits(), expected.to_bits(), "n {n}");
    }
    // 2^-1074 = 2^-1022 * 2^-52.
    assert_eq!(f64::from_bits(1), f64::MIN_POSITIVE * f64::EPSILON);
    for n in 1u32..=120 {
        let mut binomial = 1u128;
        let mut sum = 1u128;
        for tail in 0..=n / 2 {
            if tail > 0 {
                binomial = binomial * u128::from(n - tail + 1) / u128::from(tail);
                sum += binomial;
            }
            let exact = (2.0 * sum as f64 / 2f64.powi(n as i32)).min(1.0);
            let p = sign_test(u64::from(tail), u64::from(n - tail));
            assert!(
                (p - exact).abs() <= 1e-12 * exact,
                "n {n} tail {tail}: {p} against {exact}"
            );
        }
    }
}

/// Review notes PR0-O-FSYNC and PR0-O-REPLACE-METADATA: with the directory synced after the
/// rename, [`replace_document`] still writes exactly [`write_document`]'s bytes, into a new
/// nested directory and again over its own earlier document, and leaves nothing else in the
/// directory. A replaced read-only `--out` gets the mode of any new file in that directory,
/// not its old mode.
#[test]
pub(crate) fn ablation_document_replacement_syncs_its_directory_and_keeps_the_bytes() {
    let root = scratch();
    let report = run(
        &legacy_scratch_config(6, 20_260_907),
        &root.path().join("s"),
    )
    .unwrap();
    let plain = root.path().join("plain.json");
    let plain_sha256 = write_document(&plain, &report.document).unwrap();
    let expected = std::fs::read(&plain).unwrap();
    let out = root.path().join("a/b/ablation.json");
    create_document_parent(&out).unwrap();
    for _ in 0..2 {
        assert_eq!(
            replace_document(&out, &report.document).unwrap(),
            plain_sha256
        );
        assert_eq!(std::fs::read(&out).unwrap(), expected);
        assert_eq!(names(&root.path().join("a/b")), set(&["ablation.json"]));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        let fresh = root.path().join("a/b/fresh.json");
        std::fs::write(&fresh, b"{}\n").unwrap();
        std::fs::set_permissions(&out, std::fs::Permissions::from_mode(0o400)).unwrap();
        replace_document(&out, &report.document).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), expected);
        assert_eq!(mode(&out), mode(&fresh));
    }
}

// review 06 (PR0-R9): the sign test is the correctly rounded exact tail probability.

/// The correctly rounded value of `s 2^(1-n)`, capped at 1, for an exact tail sum `s` and
/// `n <= 1200`, computed apart from [`round_scaled`]. A normal result is the f64 conversion
/// of `s` (to nearest, ties to even) scaled by two exact powers of two. A subnormal result is
/// `s / 2^(n-1075)` rounded to an integer, ties to even, which is its bit pattern.
pub(crate) fn correctly_rounded(s: u128, n: u32) -> f64 {
    assert!(n <= 1200);
    let length = 128 - s.leading_zeros();
    if length >= n {
        return 1.0;
    }
    if length + 1022 >= n {
        return ((s as f64) * 2f64.powi(-600) * 2f64.powi(601 - n as i32)).min(1.0);
    }
    let units = if n <= 1075 {
        s << (1075 - n)
    } else {
        let shift = n - 1075;
        let quotient = s >> shift;
        let remainder = s & ((1_u128 << shift) - 1);
        let half = 1_u128 << (shift - 1);
        if remainder > half || (remainder == half && quotient & 1 == 1) {
            quotient + 1
        } else {
            quotient
        }
    };
    f64::from_bits(units as u64)
}

/// Review 06, PR0-R9: the sign test is the correctly rounded exact probability, bit for bit,
/// for every split up to 120 pairs and for every split with at most 12 on the smaller side up
/// to 1200 pairs, in both orientations. That covers the review's (9, 1086), whose last bit the
/// earlier arithmetic changed in the normal range, and the review's cases that it sent to 0
/// although the exact value rounds to the smallest positive f64.
#[test]
pub(crate) fn review06_sign_test_is_the_correctly_rounded_exact_tail() {
    let check = |n: u32, max_tail: u32| {
        let mut binomial = 1_u128;
        let mut sum = 1_u128;
        for tail in 0..=max_tail.min(n / 2) {
            if tail > 0 {
                binomial = binomial * u128::from(n - tail + 1) / u128::from(tail);
                sum += binomial;
            }
            let expected = correctly_rounded(sum, n);
            for (wins, losses) in [(tail, n - tail), (n - tail, tail)] {
                let p = sign_test(u64::from(wins), u64::from(losses));
                assert_eq!(
                    p.to_bits(),
                    expected.to_bits(),
                    "{wins} wins, {losses} losses: {p:e} against {expected:e}"
                );
            }
        }
    };
    for n in 1..=120 {
        check(n, n / 2);
    }
    for n in 121..=1200 {
        check(n, 12);
    }
    // The review's subnormal cases, as bit patterns of multiples of 2^-1074.
    for (wins, losses, bits) in [
        (1, 1085, 1),
        (2, 1093, 1),
        (6, 1121, 1),
        (12, 1157, 1),
        (1, 1076, 270),
        (2, 1074, 289_714),
        (2, 1084, 288),
    ] {
        assert_eq!(
            sign_test(wins, losses).to_bits(),
            bits,
            "{wins} wins, {losses} losses"
        );
    }
}

/// Review 07, PR0-R10: the sign test's count domain. Up to `MAX_SIGN_TEST_PAIRS` untied pairs
/// it is the correctly rounded exact tail, right up to the bound. Above it, or when
/// `wins + losses` does not fit in a u64, it is NaN, which `paired_counts` refuses as not
/// finite. Every case here costs at most a few binomial steps.
#[test]
pub(crate) fn review07_sign_test_bounds_its_count_domain() {
    let max = MAX_SIGN_TEST_PAIRS;
    assert_eq!(max, 1 << 20);
    // At the bound: 2^(1-max) and a 3-against-(max-3) tail are far below 2^-1075, so 0, and
    // an even split is 1.
    for (wins, losses, expected) in [
        (max, 0, 0.0),
        (0, max, 0.0),
        (max - 3, 3, 0.0),
        (3, max - 3, 0.0),
        (max / 2, max / 2, 1.0),
    ] {
        assert_eq!(
            sign_test(wins, losses).to_bits(),
            f64::to_bits(expected),
            "{wins} wins, {losses} losses"
        );
    }
    // Above the bound, and sums that do not fit in a u64.
    for (wins, losses) in [
        (max + 1, 0),
        (0, max + 1),
        (max, 1),
        (max / 2 + 1, max / 2),
        (u64::MAX, 0),
        (0, u64::MAX),
        (u64::MAX, 1),
        (u64::MAX, u64::MAX),
        (u64::MAX / 2 + 1, u64::MAX / 2 + 1),
    ] {
        assert!(
            sign_test(wins, losses).is_nan(),
            "{wins} wins, {losses} losses"
        );
    }
    // The paired summary refuses a p that is not finite, as it refuses any such number.
    assert!(finite("paired sign test p", f64::NAN).is_err());
}
