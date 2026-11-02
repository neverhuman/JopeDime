use super::*;

fn run(history: &str) -> String {
    let repo = repo();
    let value = elo_history::sensitivity_of(
        &repo.join(elo_history::RECORD),
        &repo,
        history,
        Zstd::Builtin,
    )
    .expect("sensitivity");
    pyjson::dumps(&value, true, None)
}

#[test]
fn same_cut_sensitivity_matches_the_python_replay() {
    // Addition: the Python replay's `--sensitivity` outputs recorded in the slice-2 baseline.
    assert_eq!(
        run("H-same-cut-v1-screen"),
        r#"{"extended_converged": true, "extended_iterations": 123, "historical_cap": 250, "historical_converged": true, "history": "H-same-cut-v1-screen", "max_abs_rating_shift": 0.0, "positions_changed": 0, "top10_unchanged": true}"#
    );
    let repo = repo();
    assert_refused(
        elo_history::sensitivity_of(
            &repo.join(elo_history::RECORD),
            &repo,
            "H-legacy-round1-internal-elo",
            Zstd::Builtin,
        ),
        "^H-legacy-round1-internal-elo: no replayable history$",
    );
}

#[test]
fn golden45_sensitivity_matches_the_python_replay_bit_for_bit() {
    // Addition: 2,000 synchronous Newton iterations must land on the same double as the
    // Python replay (0.30718335953474707), so the accumulation order is exactly preserved.
    assert_eq!(
        run("H-sotarena-golden45-published"),
        r#"{"extended_converged": false, "extended_iterations": 2000, "historical_cap": 250, "historical_converged": false, "history": "H-sotarena-golden45-published", "max_abs_rating_shift": 0.30718335953474707, "positions_changed": 6, "top10_unchanged": true}"#
    );
}
