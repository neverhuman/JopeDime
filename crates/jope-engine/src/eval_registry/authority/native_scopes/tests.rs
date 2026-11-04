use super::*;
use ed25519_dalek::{Signer, SigningKey};
fn signed() -> SignedNativeScopeRegistration {
    let key = SigningKey::from_bytes(&[94; 32]);
    let a = Artifact {
        path: "evidence.json".into(),
        sha256: "a".repeat(64),
    };
    let body = NativeScopeRegistrationBody {
        schema: "jope.native_scope_registration.v1".into(),
        root: "/data".into(),
        authority_root: "/data/authority".into(),
        authority_instance: "a".repeat(64),
        authority_manifest_sha256: "b".repeat(64),
        authority_epoch: "a".repeat(64),
        journal_sequence: 12,
        journal_tip_sha256: "c".repeat(64),
        run: "run".into(),
        scope: ScopePolicy {
            scope: "native-production".into(),
            training_sha256: (0..96).map(|i| format!("{i:064x}")).collect(),
            numerical_contract_sha256: vec!["d".repeat(64)],
            fit_seeds: vec![0],
        },
        preparation_overlay: a.clone(),
        scientific_policy: a,
        not_before_unix_ms: 100,
        expires_unix_ms: 200,
    };
    let bytes = encoded(&body).unwrap();
    let mut message = DOMAIN.to_vec();
    message.extend(&bytes);
    SignedNativeScopeRegistration {
        authorization_public_key: hex::encode(key.verifying_key().to_bytes()),
        body_sha256: hash(&bytes),
        body,
        signature: hex::encode(key.sign(&message).to_bytes()),
    }
}
#[test]
fn signature_binds_original_head_epoch_all96training_and_fixed_key() {
    let s = signed();
    verify_signature(&s, &s.authorization_public_key).unwrap();
    for field in 0..4 {
        let mut bad = s.clone();
        match field {
            0 => bad.body.journal_sequence += 1,
            1 => bad.body.authority_epoch = "f".repeat(64),
            2 => bad.body.scope.training_sha256[2] = "f".repeat(64),
            _ => bad.body.scope.fit_seeds = vec![1],
        };
        assert!(verify_signature(&bad, &s.authorization_public_key).is_err());
    }
    assert!(verify_signature(
        &s,
        &hex::encode(SigningKey::from_bytes(&[95; 32]).verifying_key().to_bytes())
    )
    .is_err());
}
#[test]
fn scope_training_postimages_replay_without_erasing_original_namespace() {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch("CREATE TABLE scope_training(scope TEXT NOT NULL,training TEXT NOT NULL,PRIMARY KEY(scope,training)) WITHOUT ROWID;").unwrap();
    db.execute_batch(TABLE_SQL).unwrap();
    let s = signed();
    db.execute(
        "INSERT INTO scope_training VALUES('original','retained')",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO native_scopes VALUES(?,?,?)",
        params![
            s.body.scope.scope,
            "a".repeat(64),
            serde_json::to_string(&s.body.scope).unwrap()
        ],
    )
    .unwrap();
    let row = json!({"scope":s.body.scope.scope,"training_json":serde_json::to_string(&s.body.scope.training_sha256).unwrap()});
    Authority::replay_scope_training(&db, &s.body.scope.scope, &row).unwrap();
    Authority::replay_scope_training(&db, &s.body.scope.scope, &row).unwrap();
    assert_eq!(
        Authority::scope_training_postimage(&db, &s.body.scope.scope).unwrap(),
        row
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM scope_training WHERE scope='original'",
            [],
            |r| r.get::<_, u64>(0)
        )
        .unwrap(),
        1
    );
    let mut bad = row;
    bad["training_json"] = json!("[]");
    assert!(Authority::replay_scope_training(&db, &s.body.scope.scope, &bad).is_err());
}
#[test]
fn original_signed_scope_event_cannot_be_moved_or_change_postimages() {
    let signed = signed();
    verify_signature(&signed, &signed.authorization_public_key).unwrap();
    let b = &signed.body;
    let sha = hash(&encoded(&signed).unwrap());
    let mut event = Event {
        instance: b.authority_instance.clone(),
        seq: b.journal_sequence + 1,
        previous: b.journal_tip_sha256.clone(),
        kind: "native_scope_register".into(),
        payload: json!({"scope":b.scope.scope,"registration_sha":sha}),
        effects: vec![
            StateRow {
                table: "native_scopes".into(),
                key: b.scope.scope.clone(),
                row: json!({"scope":b.scope.scope,"registration_sha":sha,"json":serde_json::to_string(&b.scope).unwrap()}),
            },
            StateRow {
                table: "scope_training".into(),
                key: b.scope.scope.clone(),
                row: json!({"scope":b.scope.scope,"training_json":serde_json::to_string(&b.scope.training_sha256).unwrap()}),
            },
        ],
    };
    verify_scope_event_bindings(&event, &signed, &sha, &b.authority_manifest_sha256).unwrap();
    event.seq += 1;
    assert!(
        verify_scope_event_bindings(&event, &signed, &sha, &b.authority_manifest_sha256).is_err()
    );
    event.seq -= 1;
    event.effects[1].row["training_json"] = json!("[]");
    assert!(
        verify_scope_event_bindings(&event, &signed, &sha, &b.authority_manifest_sha256).is_err()
    );
}
