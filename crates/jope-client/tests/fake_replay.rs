use std::time::Duration;

use jope_client::fake::FakeTransport;

const FIXTURES: &[(&str, &str)] = &[
    (
        "successful run",
        include_str!("../../jope-appliance/tests/fixtures/events/success.jsonl"),
    ),
    (
        "warnings",
        include_str!("../../jope-appliance/tests/fixtures/events/warnings.jsonl"),
    ),
    (
        "failed candidate burst",
        include_str!("../../jope-appliance/tests/fixtures/events/failed-candidate-burst.jsonl"),
    ),
    (
        "paused and continued run",
        include_str!("../../jope-appliance/tests/fixtures/events/paused-continued.jsonl"),
    ),
    (
        "cancelled run",
        include_str!("../../jope-appliance/tests/fixtures/events/cancelled.jsonl"),
    ),
];

#[test]
fn fixtures_validate_as_public_events_and_replay_deterministically() {
    let interval = Duration::from_millis(25);

    for (scenario, jsonl) in FIXTURES {
        let replay = || {
            FakeTransport::from_jsonl(jsonl, interval)
                .unwrap_or_else(|error| panic!("{scenario} fixture is invalid: {error}"))
                .collect::<Vec<_>>()
        };

        let first = replay();
        let second = replay();
        assert!(
            !first.is_empty(),
            "{scenario} fixture should contain events"
        );
        assert_eq!(first, second, "{scenario} replay changed between runs");
        assert_eq!(first[0].delay_before, Duration::ZERO);
        assert!(first
            .iter()
            .skip(1)
            .all(|event| event.delay_before == interval));
    }
}
