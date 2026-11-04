use super::*;

/// PR #114 review round 1, P2 (the reviewer's probe): the context window and its execution moved
/// 120,000 ms later, the teacher's original 60,000-ms window and all its evidence unchanged. The
/// context was produced entirely after the teaching deadline, so it never acquires teaching
/// authority.
#[test]
fn a_context_produced_after_the_teaching_window_is_refused() {
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let mut later = f.window.clone();
    later.started_unix_ms += 120_000;
    later.deadline_unix_ms += 120_000;
    assert!(later.started_unix_ms > f.window.deadline_unix_ms);
    let run = producer_run(&f, &teaching_context(), &later, |_| {});
    assert_refused(
        &f,
        &submission(&f, &run),
        "not in the teacher's authorized experiment",
        "a context produced after the teaching window",
    );
}

/// P2: a context produced in another experiment, over the teacher's window times, is refused.
#[test]
fn a_context_from_another_experiment_is_refused() {
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let mut other = f.experiment.clone();
    other.methods_sha256 = "6".repeat(64);
    let window = ExecutionWindow {
        experiment_sha256: other.identity().unwrap(),
        ..f.window.clone()
    };
    let run = producer_run_in(&f, &teaching_context(), &other, &window, |_| {});
    assert_refused(
        &f,
        &submission(&f, &run),
        "not in the teacher's authorized experiment",
        "a context from another experiment",
    );
}

/// P2: a context produced in a window that ends after the teaching deadline (an experiment with a
/// longer allowance, the job exiting after the teacher's deadline) is refused.
#[test]
fn a_context_window_that_ends_after_the_teaching_deadline_is_refused() {
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let mut longer = f.experiment.clone();
    longer.resources.total_seconds *= 3;
    let window = ExecutionWindow {
        experiment_sha256: longer.identity().unwrap(),
        started_unix_ms: f.window.started_unix_ms,
        deadline_unix_ms: f.window.started_unix_ms + longer.resources.total_seconds * 1000,
    };
    assert!(window.deadline_unix_ms > f.window.deadline_unix_ms);
    let deadline = f.window.deadline_unix_ms;
    let run = producer_run_in(&f, &teaching_context(), &longer, &window, |p| {
        p.outcome.consumption.gate_released_unix_ms = Some(deadline + 1_000);
        p.outcome.consumption.observed_exit_unix_ms = Some(deadline + 1_010);
        p.seal();
    });
    assert_refused(
        &f,
        &submission(&f, &run),
        "not in the teacher's authorized experiment",
        "a context window that ends after the teaching deadline",
    );
}
