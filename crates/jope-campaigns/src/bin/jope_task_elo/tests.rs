use super::*;

#[test]
fn robust_bank_candidates_are_their_own_producer_not_generic_gp() {
    assert_eq!(
        producer_of("gp_origin", "cross_dataset_repeated_robust_gp_winner"),
        "robust_bank"
    );
    assert_eq!(
        producer_of("gp_origin", "robust_gp_seed_bank_coverage_probe"),
        "robust_bank"
    );
    assert_eq!(
        producer_of("gp_origin", "deterministic_rust_mutation_crossover"),
        "gp_origin"
    );
    assert_eq!(
        producer_of("dl_origin", "current_checkpoint_dl_proposal"),
        "dl_origin"
    );
    assert_eq!(producer_of("dl_descendant", "whatever"), "dl_descendant");
}

#[test]
fn sampling_is_deterministic_and_exactly_the_quota() {
    let values: Vec<f64> = (0..100).map(|value| value as f64).collect();
    let seed = stride_seed("multiclass", "00b383718f6f4fd8");
    let first = sample_n(&values, 7, seed);
    let again = sample_n(&values, 7, seed);
    assert_eq!(first.len(), 7);
    assert_eq!(first, again, "ELO must be replayable from the same corpus");
    // A different dataset must not reuse the same stream position.
    let other = sample_n(&values, 7, stride_seed("multiclass", "0f7f4024a9b28e9c"));
    assert_ne!(first, other);
    // Fewer values than the quota returns everything, never pads.
    assert_eq!(sample_n(&values[..3], 7, seed).len(), 3);
}

/// The whole reason this binary exists: with equal attempts the arms are
/// judged on quality, and the raw view must visibly disagree when one arm
/// simply had more draws.
#[test]
fn draw_control_changes_the_verdict_when_attempts_are_lopsided() {
    let mut accumulator = TaskAccumulator::default();
    // GP gets 20 mediocre attempts; DL gets 2 good ones. Best-of-20 beats
    // best-of-2 on volume alone, but per-attempt the DL is stronger.
    let mut rows: Vec<(String, String, String, f64, f64)> = Vec::new();
    for index in 0..20 {
        rows.push((
            format!("gp{index}"),
            "gp_origin".into(),
            "deterministic_rust_mutation_crossover".into(),
            0.50 + index as f64 * 0.005,
            1.0,
        ));
    }
    for index in 0..2 {
        rows.push((
            format!("dl{index}"),
            "dl_origin".into(),
            "current_checkpoint_dl_proposal".into(),
            0.58 + index as f64 * 0.001,
            1.0,
        ));
    }
    absorb_dataset(&mut accumulator, "binary", "dataset0", &rows);

    let raw_gp = accumulator.raw.get("gp_origin").expect("gp raw").value;
    let raw_dl = accumulator.raw.get("dl_origin").expect("dl raw").value;
    assert!(
        raw_gp > raw_dl,
        "raw best-of-20 (0.595) should beat best-of-2 (0.581)"
    );

    let ctl_gp = accumulator
        .controlled
        .get("gp_origin")
        .expect("gp controlled")
        .value;
    let ctl_dl = accumulator
        .controlled
        .get("dl_origin")
        .expect("dl controlled")
        .value;
    assert!(
        ctl_dl > ctl_gp,
        "with 2 attempts each the DL's better proposals must win: \
             dl={ctl_dl} gp={ctl_gp}"
    );
}

#[test]
fn thin_evidence_is_published_as_absent_not_as_a_number() {
    let mut table = BTreeMap::new();
    table.insert(
        "dl_origin".to_owned(),
        Rating {
            value: 1600.0,
            matches: 3,
            wins: 3.0,
        },
    );
    table.insert(
        "gp_origin".to_owned(),
        Rating {
            value: 1300.0,
            matches: 500,
            wins: 300.0,
        },
    );
    let rows = rating_rows(&table, None);
    let thin = rows
        .iter()
        .find(|row| row["name"] == "dl_origin")
        .expect("dl row");
    assert!(thin["elo"].is_null(), "3 matches must not publish a rating");
    assert_eq!(thin["insufficient_evidence"], json!(true));
    let solid = rows
        .iter()
        .find(|row| row["name"] == "gp_origin")
        .expect("gp row");
    assert!(solid["elo"].is_number());
    assert_eq!(solid["insufficient_evidence"], json!(false));
}

#[test]
fn shrinkage_keeps_a_lucky_short_run_below_a_proven_one() {
    let lucky = Rating {
        value: 1500.0,
        matches: 4,
        wins: 4.0,
    };
    let proven = Rating {
        value: 1400.0,
        matches: 2_000,
        wins: 1_400.0,
    };
    assert!(
        proven.shrunk() > lucky.shrunk(),
        "4 wins must not outrank 2000 matches: lucky={} proven={}",
        lucky.shrunk(),
        proven.shrunk()
    );
}
