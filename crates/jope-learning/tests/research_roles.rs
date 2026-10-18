//! Offline contract checks for the I49 candidate matrix.
//!
//! The fake lease source below tests the proposed expiry/refresh state machine only. It is not
//! wired to AWS and does not claim the current jope-datasets static store provider refreshes.
use serde_json::Value;

const CANDIDATES: &str = include_str!("../../../deploy/iam/jope-research-role-candidates.json");

fn matrix() -> Value {
    serde_json::from_str(CANDIDATES).expect("role candidate JSON is valid")
}

fn roles<'a>(matrix: &'a Value) -> &'a [Value] {
    matrix["roles"].as_array().expect("roles array")
}

fn role<'a>(matrix: &'a Value, id: &str) -> &'a Value {
    roles(matrix)
        .iter()
        .find(|role| role["id"] == id)
        .unwrap_or_else(|| panic!("missing candidate {id}"))
}

fn allows(role: &Value, capability: &str) -> bool {
    let allow = role["allow"].as_array().unwrap();
    let deny = role["deny"].as_array().unwrap();
    !deny.iter().any(|entry| entry["capability"] == capability)
        && allow.iter().any(|entry| entry["capability"] == capability)
}

#[test]
fn candidate_is_explicitly_uninstalled_and_live_checks_remain_unexecuted() {
    let matrix = matrix();
    assert_eq!(matrix["schema"], "jopedime.research-role-candidates.v1");
    assert_eq!(matrix["status"], "candidate_only_not_provisioned");
    assert_eq!(matrix["deployable_iam_document"], false);
    assert_eq!(matrix["operational_status"], "NOT_EXECUTED");
    assert_eq!(matrix["account_binding"], Value::Null);
    assert_eq!(matrix["trust_principals"], Value::Null);
    assert_eq!(roles(&matrix).len(), 6);
    assert!(roles(&matrix).iter().all(|role| {
        role["status"] == "not_provisioned" && role["trust_principal"] == Value::Null
    }));
    assert_eq!(
        matrix["live_qualification"]["positive_tests"],
        "NOT_EXECUTED"
    );
    assert_eq!(matrix["live_qualification"]["denied_tests"], "NOT_EXECUTED");
}

#[test]
fn actor_cannot_read_judge_rows_or_write_accepted_pointers() {
    let matrix = matrix();
    let actor = role(&matrix, "actor");
    assert!(!allows(actor, "read_judge_rows"));
    assert!(!allows(actor, "write_accepted_pointer"));
    assert!(allows(actor, "read_training_inputs"));

    let judge_readers: Vec<_> = roles(&matrix)
        .iter()
        .filter(|candidate| allows(candidate, "read_judge_rows"))
        .map(|candidate| candidate["id"].as_str().unwrap())
        .collect();
    assert_eq!(judge_readers, vec!["judge"]);

    let pointer_writers: Vec<_> = roles(&matrix)
        .iter()
        .filter(|candidate| allows(candidate, "write_accepted_pointer"))
        .map(|candidate| candidate["id"].as_str().unwrap())
        .collect();
    assert_eq!(pointer_writers, vec!["publisher"]);
    for candidate in roles(&matrix) {
        let allows_delete = candidate["allow"].as_array().unwrap().iter().any(|entry| {
            entry["actions"].as_array().is_some_and(|actions| {
                actions
                    .iter()
                    .any(|action| action == "s3:DeleteObject" || action == "s3:DeleteObjectVersion")
            })
        });
        assert!(
            !allows_delete,
            "{} must not have delete actions",
            candidate["id"]
        );
        assert!(candidate["deny"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| { entry["capability"] == "delete_any_object" }));
    }

    let publisher = role(&matrix, "publisher");
    let pointer = publisher["allow"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["capability"] == "write_accepted_pointer")
        .unwrap();
    assert!(pointer["conditions"]
        .as_array()
        .unwrap()
        .iter()
        .any(|condition| condition.as_str().unwrap().contains("If-Match")));
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FakeLease {
    fixture_id: u8,
    expires_at: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefreshError {
    RefreshFailed,
    StillExpired,
}

struct FakeProvider {
    next: Result<FakeLease, RefreshError>,
    calls: usize,
}

fn lease_for_operation(
    cached: &FakeLease,
    now: u64,
    refresh_before_expiry: u64,
    provider: &mut FakeProvider,
) -> Result<FakeLease, RefreshError> {
    if cached.expires_at > now.saturating_add(refresh_before_expiry) {
        return Ok(cached.clone());
    }
    provider.calls += 1;
    let refreshed = provider.next.clone()?;
    if refreshed.expires_at <= now {
        return Err(RefreshError::StillExpired);
    }
    Ok(refreshed)
}

fn refresh_before_expiry(matrix: &Value) -> u64 {
    matrix["proposed_refresh_contract"]["refresh_before_expiry_seconds"]
        .as_u64()
        .expect("declared refresh margin")
}

#[test]
fn expired_fake_lease_refreshes_once_and_uses_new_lease() {
    let matrix = matrix();
    let current = FakeLease {
        fixture_id: 1,
        expires_at: 100,
    };
    let mut provider = FakeProvider {
        next: Ok(FakeLease {
            fixture_id: 2,
            expires_at: 300,
        }),
        calls: 0,
    };
    let refreshed =
        lease_for_operation(&current, 100, refresh_before_expiry(&matrix), &mut provider).unwrap();
    assert_eq!(refreshed.fixture_id, 2);
    assert_eq!(provider.calls, 1);
}

#[test]
fn refresh_failure_or_expired_replacement_fails_closed_without_stale_fallback() {
    let matrix = matrix();
    assert_eq!(
        matrix["proposed_refresh_contract"]["on_refresh_failure"],
        "fail_closed_without_stale_or_ambient_credential_fallback"
    );
    assert_eq!(
        matrix["proposed_refresh_contract"]["max_refresh_attempts_per_operation"],
        1
    );
    let current = FakeLease {
        fixture_id: 1,
        expires_at: 100,
    };
    let mut failing = FakeProvider {
        next: Err(RefreshError::RefreshFailed),
        calls: 0,
    };
    assert_eq!(
        lease_for_operation(&current, 100, refresh_before_expiry(&matrix), &mut failing),
        Err(RefreshError::RefreshFailed)
    );
    assert_eq!(failing.calls, 1);

    let mut stale_replacement = FakeProvider {
        next: Ok(FakeLease {
            fixture_id: 2,
            expires_at: 99,
        }),
        calls: 0,
    };
    assert_eq!(
        lease_for_operation(
            &current,
            100,
            refresh_before_expiry(&matrix),
            &mut stale_replacement
        ),
        Err(RefreshError::StillExpired)
    );
    assert_eq!(stale_replacement.calls, 1);
}
