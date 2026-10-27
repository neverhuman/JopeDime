use super::*;
use jope_records::replay_admission::{ADOPTION_PATH, COMPONENT_AUTHORITY_PATH};
use std::collections::BTreeSet;
use std::fs;

mod fixture;
use fixture::{canonical, float_hex, Fixture, EPISODE, UNTIL};

fn refused<T>(result: Result<T>, expected: &str) {
    let message = match result {
        Ok(_) => panic!("expected refusal containing {expected:?}"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        message.contains(expected),
        "expected {expected:?}, observed {message}"
    );
}

#[test]
fn admitted_fit_components_bind_exact_scope_and_keep_authority_identity() {
    let fixture = Fixture::new();
    let requests = fixture.load().unwrap();
    let authenticated = requests.resolve(&fixture.scope).unwrap();
    let identity = authenticated.identity();
    assert_eq!(identity.schema, "jope.component-admission/v1");
    assert_eq!(identity.subject_sha256, fixture.subject);
    assert_eq!(identity.exposure_receipt_sha256, fixture.proof);
    assert_eq!(identity.policy_sha256, fixture.policy);
    assert_eq!(
        identity.authority_sha256,
        sha256(&fs::read(fixture.root.join(COMPONENT_AUTHORITY_PATH)).unwrap())
    );
    assert_eq!(identity.fit_scope_digest, fixture.scope.digest().unwrap());
    assert_eq!(identity.episode, EPISODE);
    assert_eq!(identity.purpose, "episode_fit");
    assert_eq!(fixture.scope.fold.as_deref(), Some("fixture-inner-fold-0"));
    assert_eq!(
        fixture.scope.physical_rows_sha256,
        sha256(&serde_json::to_vec(&[701u64, 5, 109, 42, 8, 9001, 63, 2]).unwrap())
    );
    let landmark_scope = sha256(b"native landmark scope");
    let bound = authenticated.bind(&fixture.scope, &landmark_scope).unwrap();
    assert_eq!(bound.source_digest, landmark_scope);
    assert_eq!(bound.recipe, FUSION_COMPONENT_RECIPE);
    assert_eq!(bound.foundation1024, vec![1.25; 1024]);
    assert_eq!(bound.target512, vec![-2.5; 512]);
    authenticated.clone().require_active().unwrap();
}

#[test]
fn scope_relabeling_cannot_change_fold_fit_rows_or_preparation() {
    let fixture = Fixture::new();
    let authenticated = fixture.load().unwrap().resolve(&fixture.scope).unwrap();
    for case in 0..7 {
        let mut wrong = fixture.scope.clone();
        match case {
            0 => wrong.fold = Some("another-fold".into()),
            1 => wrong.fold = None,
            2 => wrong.physical_rows_sha256 = sha256(b"another ordered FIT roster"),
            3 => wrong.rows -= 1,
            4 => wrong.preparation_source = "0".repeat(40),
            5 => wrong.source_digest = sha256(b"another prepared source"),
            _ => wrong.task = "Binary".into(),
        }
        refused(
            authenticated.bind(&wrong, &"a".repeat(64)),
            "FIT/fold scope differs",
        );
        // An attacker can rewrite request selectors, but cannot relabel the admitted subject.
        fixture.write_request(&wrong, EPISODE);
        refused(
            fixture.load().unwrap().resolve(&wrong),
            "FIT/preparation/episode/recipe differs",
        );
    }
}

#[test]
fn authenticated_subject_still_requires_the_supported_component_recipe() {
    let mut fixture = Fixture::new();
    fixture.artifact.recipe = "unknown-normalizer/v999".into();
    fixture.publish();
    refused(
        fixture.load().unwrap().resolve(&fixture.scope),
        "recipe differs",
    );
}

#[cfg(unix)]
#[test]
fn every_known_prohibited_model_is_refused_before_opening_its_payload_path() {
    use jope_records::replay_admission::ArtifactStore;
    let fixture = Fixture::new();
    let controller = authority::ControllerStore::load(&fixture.root).unwrap();
    let manifest: crate::bundle::ModelBundleV1 =
        serde_json::from_slice(include_bytes!("../../../artifacts/model-bundle.v1.json")).unwrap();
    let mut count = 0;
    for model in manifest.files.iter().filter(|file| {
        file.license.redistribution != "permitted" || !file.license.policy_eligibility.eligible
    }) {
        // An attempted descriptor-relative read would produce a symlink error instead.
        std::os::unix::fs::symlink("/p4-must-never-open", fixture.blob(&model.sha256)).unwrap();
        refused(
            controller.read_blob(&model.sha256),
            "prohibited model weights",
        );
        count += 1;
    }
    assert!(count >= 3);
}

#[test]
fn request_and_subject_episodes_cannot_override_receipt_episode() {
    let mut fixture = Fixture::new();
    fixture.write_request(&fixture.scope, "another-episode");
    refused(
        fixture.load().unwrap().resolve(&fixture.scope),
        "outside its declared episode/view",
    );
    fixture.artifact.episode = "another-episode".into();
    fixture.publish();
    refused(
        fixture.load().unwrap().resolve(&fixture.scope),
        "FIT/preparation/episode/recipe differs",
    );
}

#[test]
fn every_producer_ancestor_must_stay_in_training_fit_view_and_episode() {
    for case in 0..5 {
        let fixture = Fixture::with_ancestor(|ancestor| match case {
            0 => ancestor.within_task_view = "inner_validation".into(),
            1 => ancestor.original_role = "train_query".into(),
            2 => ancestor.meta_partition = "meta_dev".into(),
            3 => ancestor.prior_exposure = Some(vec!["judge".into()]),
            _ => ancestor.episode = Some("another-episode".into()),
        });
        let expected = if case == 4 {
            "outside its declared episode/view"
        } else {
            "FIT-only training ancestry"
        };
        refused(fixture.load().unwrap().resolve(&fixture.scope), expected);
    }
}

#[test]
fn producer_references_must_name_retained_subjects_in_the_actual_admitted_closure() {
    let mut fixture = Fixture::new();
    let unadmitted = fixture.put(b"present but unadmitted producer");
    fixture
        .artifact
        .producer_inputs
        .insert("foundation".into(), unadmitted);
    fixture.publish();
    refused(
        fixture.load().unwrap().resolve(&fixture.scope),
        "not in the admitted ancestry",
    );
    fixture.artifact.producer_inputs.remove("foundation");
    fixture.publish();
    refused(
        fixture.load().unwrap().resolve(&fixture.scope),
        "not in the admitted ancestry",
    );
}

#[test]
fn missing_authority_and_request_selected_authority_are_refused() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.root.join(COMPONENT_AUTHORITY_PATH)).unwrap();
    refused(
        fixture.load(),
        "controller component authority is not installed",
    );
    let mut request: serde_json::Value =
        serde_json::from_slice(&fs::read(&fixture.request).unwrap()).unwrap();
    request["authority_path"] = fixture.root.display().to_string().into();
    fs::write(&fixture.request, serde_json::to_vec(&request).unwrap()).unwrap();
    refused(fixture.load(), "unknown field");
}

#[test]
fn retained_subject_receipt_and_producer_tampering_invalidates_existing_capability() {
    for case in 0..3 {
        let fixture = Fixture::new();
        let requests = fixture.load().unwrap();
        let authenticated = requests.resolve(&fixture.scope).unwrap();
        let digest = match case {
            0 => &fixture.subject,
            1 => &fixture.proof,
            _ => &fixture.foundation,
        };
        fs::write(fixture.blob(digest), b"altered immutable CAS bytes").unwrap();
        refused(
            authenticated.require_active(),
            "altered component proof or subject",
        );
        refused(
            authenticated.bind(&fixture.scope, &"a".repeat(64)),
            "altered component proof or subject",
        );
        refused(
            requests.resolve(&fixture.scope),
            "altered component proof or subject",
        );
    }
}

#[test]
fn capability_rechecks_policy_time_without_waiting_for_wall_clock_expiry() {
    let fixture = Fixture::new();
    let authenticated = fixture.load().unwrap().resolve(&fixture.scope).unwrap();
    authenticated.require_active_at(UNTIL - 1).unwrap();
    refused(
        authenticated.require_active_at(UNTIL),
        "purpose or policy epoch not effective",
    );
}

#[test]
fn changing_adoption_or_authority_refuses_existing_capability() {
    let fixture = Fixture::new();
    let authenticated = fixture.load().unwrap().resolve(&fixture.scope).unwrap();
    let mut adoption: jope_records::replay_admission::PolicyPin =
        serde_json::from_slice(&fs::read(fixture.root.join(ADOPTION_PATH)).unwrap()).unwrap();
    adoption.issuer = "another-controller".into();
    fs::write(fixture.root.join(ADOPTION_PATH), canonical(&adoption)).unwrap();
    refused(
        authenticated.require_active(),
        "controller replay adoption changed",
    );
    refused(fixture.load(), "controller replay adoption changed");
}

#[test]
fn installed_fixture_revocation_invalidates_cached_and_fresh_admissions() {
    let fixture = Fixture::new();
    let authenticated = fixture.load().unwrap().resolve(&fixture.scope).unwrap();
    // Deliberate isolated test-controller installation, not a claim of production ledger import.
    fixture.install_authority(BTreeSet::from([fixture.foundation.clone()]));
    refused(
        authenticated.require_active(),
        "component controller authority changed",
    );
    refused(
        fixture.load().unwrap().resolve(&fixture.scope),
        "RevokedEvidence",
    );
}

#[test]
fn ledger_inventory_changes_need_a_fresh_controller_snapshot() {
    let fixture = Fixture::new();
    let authenticated = fixture.load().unwrap().resolve(&fixture.scope).unwrap();
    fs::create_dir_all(fixture.root.join("revocations/ledger")).unwrap();
    fs::write(
        fixture.root.join("revocations/ledger/HEAD.json"),
        b"new unreviewed head",
    )
    .unwrap();
    refused(
        authenticated.require_active(),
        "component revocation state changed",
    );
    refused(fixture.load(), "component revocation state changed");
}

#[test]
fn component_vectors_refuse_wrong_width_noncanonical_bytes_and_nonfinite_values() {
    for case in 0..3 {
        let mut fixture = Fixture::new();
        let expected = match case {
            0 => {
                fixture.artifact.foundation1024_le_f32.truncate(8);
                "vector width differs"
            }
            1 => {
                fixture.artifact.target512_le_f32 = "0000A03F".repeat(512);
                "noncanonical component float bytes"
            }
            _ => {
                fixture.artifact.target512_le_f32 = float_hex(f32::INFINITY, 512);
                "nonfinite component value"
            }
        };
        fixture.publish();
        refused(fixture.load().unwrap().resolve(&fixture.scope), expected);
    }
}
