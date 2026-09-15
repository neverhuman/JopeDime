//! Independently reviewed additive native scopes in the original v6 journal.
use super::*;
use crate::native_graph::Artifact;
use ed25519_dalek::{Signature, VerifyingKey};
const DOMAIN: &[u8] = b"jope.native_scope_registration.signature.v1\0";
pub(super) const TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS native_scopes(scope TEXT PRIMARY KEY,registration_sha TEXT NOT NULL,json TEXT NOT NULL) WITHOUT ROWID";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeScopeRegistrationBody {
    pub schema: String,
    pub root: PathBuf,
    pub authority_root: PathBuf,
    pub authority_instance: String,
    pub authority_manifest_sha256: String,
    pub authority_epoch: String,
    pub journal_sequence: u64,
    pub journal_tip_sha256: String,
    pub run: String,
    pub scope: ScopePolicy,
    pub preparation_overlay: Artifact,
    pub scientific_policy: Artifact,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedNativeScopeRegistration {
    pub authorization_public_key: String,
    pub body_sha256: String,
    pub body: NativeScopeRegistrationBody,
    pub signature: String,
}
fn verify_signature(s: &SignedNativeScopeRegistration, pinned: &str) -> Result<()> {
    let b = &s.body;
    ensure!(
        s.authorization_public_key == pinned
            && b.schema == "jope.native_scope_registration.v1"
            && !b.run.is_empty()
            && b.run.len() <= 256
            && !b.scope.scope.is_empty()
            && b.scope.scope.len() <= 256
            && b.not_before_unix_ms < b.expires_unix_ms
            && b.expires_unix_ms - b.not_before_unix_ms <= 7 * 86400 * 1000,
        "native scope review identity/window invalid"
    );
    for h in [
        &b.authority_instance,
        &b.authority_manifest_sha256,
        &b.authority_epoch,
        &b.journal_tip_sha256,
        &s.body_sha256,
    ] {
        ensure!(valid_sha(h), "native scope review hash invalid");
    }
    ensure!(
        b.root.is_absolute()
            && b.authority_root.starts_with(&b.root)
            && b.scope.training_sha256.len() == 96
            && b.scope.numerical_contract_sha256.len() == 1
            && b.scope.fit_seeds == [0]
            && b.scope.training_sha256.iter().all(|h| valid_sha(h))
            && b.scope
                .numerical_contract_sha256
                .iter()
                .all(|h| valid_sha(h))
            && b.scope.training_sha256.windows(2).all(|w| w[0] < w[1]),
        "native scope must contain exact96sorted training identities/original contract/seed0"
    );
    let bytes = encoded(b)?;
    ensure!(
        hash(&bytes) == s.body_sha256,
        "native scope signed body differs"
    );
    let key = VerifyingKey::from_bytes(
        &hex::decode(pinned)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("review key width"))?,
    )?;
    ensure!(!key.is_weak(), "weak scope review key");
    let mut message = DOMAIN.to_vec();
    message.extend(bytes);
    key.verify_strict(
        &message,
        &Signature::from_slice(&hex::decode(&s.signature)?)?,
    )?;
    Ok(())
}
fn verified(s: &SignedNativeScopeRegistration) -> Result<()> {
    crate::native_review::require_signer(&s.authorization_public_key)?;
    verify_signature(s, &s.authorization_public_key)
}
fn table_exists(db: &Connection) -> Result<bool> {
    Ok(db.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_scopes')",
        [],
        |r| r.get(0),
    )?)
}
fn verify_scope_event_bindings(
    event: &Event,
    signed: &SignedNativeScopeRegistration,
    sha: &str,
    manifest_sha: &str,
) -> Result<()> {
    let b = &signed.body;
    ensure!(
        b.authority_instance == event.instance
            && b.authority_manifest_sha256 == manifest_sha
            && b.journal_sequence.checked_add(1) == Some(event.seq)
            && b.journal_tip_sha256 == event.previous
            && event.payload["scope"] == b.scope.scope,
        "scope registration was replayed at a different original head"
    );
    let expected = vec![
        StateRow {
            table: "native_scopes".into(),
            key: b.scope.scope.clone(),
            row: json!({"scope":b.scope.scope,"registration_sha":sha,"json":serde_json::to_string(&b.scope)?}),
        },
        StateRow {
            table: "scope_training".into(),
            key: b.scope.scope.clone(),
            row: json!({"scope":b.scope.scope,"training_json":serde_json::to_string(&b.scope.training_sha256)?}),
        },
    ];
    ensure!(
        encoded(&event.effects)? == encoded(&expected)?,
        "scope journal postimages differ from signed original registration"
    );
    Ok(())
}
impl Authority {
    pub(super) fn verify_native_scope_event(&self, event: &Event) -> Result<()> {
        if event.kind != "native_scope_register" {
            return Ok(());
        }
        ensure!(
            self.manifest.schema == NATIVE_SCHEMA,
            "scope registration in legacy journal"
        );
        let sha = event.payload["registration_sha"]
            .as_str()
            .context("scope event registration absent")?;
        let signed: SignedNativeScopeRegistration = serde_json::from_slice(&self.read_blob(sha)?)?;
        verified(&signed)?;
        verify_scope_event_bindings(event, &signed, sha, &self.manifest_sha)
    }
    pub(super) fn native_scope_table_exists(&self) -> bool {
        table_exists(&self.db).unwrap_or(false)
    }
    pub(super) fn registered_native_scope(&self, scope: &str) -> Result<Option<ScopePolicy>> {
        if self.manifest.schema != NATIVE_SCHEMA || !table_exists(&self.db)? {
            return Ok(None);
        }
        let row: Option<(String, String)> = self
            .db
            .query_row(
                "SELECT registration_sha,json FROM native_scopes WHERE scope=?",
                [scope],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((sha, json)) = row else {
            return Ok(None);
        };
        let bytes = self.read_blob(&sha)?;
        let signed: SignedNativeScopeRegistration = serde_json::from_slice(&bytes)?;
        verified(&signed)?;
        ensure!(
            signed.body.authority_instance == self.manifest.instance
                && signed.body.authority_manifest_sha256 == self.manifest_sha
                && signed.body.scope.scope == scope
                && encoded(&signed.body.scope)? == json.as_bytes(),
            "native scope indexed identity differs from original signed registration"
        );
        Ok(Some(signed.body.scope))
    }
    pub(super) fn retained_scope(
        &self,
        scope: &str,
        allow_native: bool,
    ) -> Result<Option<ScopePolicy>> {
        if let Some(s) = self.manifest.scopes.iter().find(|s| s.scope == scope) {
            return Ok(Some(s.clone()));
        }
        if allow_native {
            self.registered_native_scope(scope)
        } else {
            Ok(None)
        }
    }
    pub(super) fn all_retained_scopes(&self) -> Result<Vec<ScopePolicy>> {
        let mut scopes = self.manifest.scopes.clone();
        if table_exists(&self.db)? {
            let mut q = self
                .db
                .prepare("SELECT scope FROM native_scopes ORDER BY scope")?;
            for name in q.query_map([], |r| r.get::<_, String>(0))? {
                let name = name?;
                ensure!(
                    !scopes.iter().any(|s| s.scope == name),
                    "native scope overwrites immutable original scope"
                );
                scopes.push(
                    self.registered_native_scope(&name)?
                        .context("registered native scope disappeared")?,
                );
            }
        }
        Ok(scopes)
    }
    pub(super) fn scope_state_keys_sql(table: &str, column: &str) -> String {
        if table == "scope_training" {
            "SELECT scope FROM native_scopes ORDER BY scope".into()
        } else {
            format!("SELECT {column} FROM {table}")
        }
    }
    pub(super) fn scope_training_postimage(db: &Connection, scope: &str) -> Result<Value> {
        let mut q =
            db.prepare("SELECT training FROM scope_training WHERE scope=? ORDER BY training")?;
        let training = q
            .query_map([scope], |r| r.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(json!({"scope":scope,"training_json":serde_json::to_string(&training)?}))
    }
    pub(super) fn replay_scope_training(db: &Connection, key: &str, row: &Value) -> Result<()> {
        ensure!(row["scope"] == key, "scope-training replay key differs");
        let hashes: Vec<String> = serde_json::from_str(
            row["training_json"]
                .as_str()
                .context("original training scope postimage absent")?,
        )?;
        ensure!(
            hashes.len() == 96
                && hashes.iter().all(|h| valid_sha(h))
                && hashes.windows(2).all(|w| w[0] < w[1]),
            "original scope-training postimage malformed"
        );
        let encoded: String =
            db.query_row("SELECT json FROM native_scopes WHERE scope=?", [key], |r| {
                r.get(0)
            })?;
        let scope: ScopePolicy = serde_json::from_str(&encoded)?;
        ensure!(scope.training_sha256 == hashes, "scope postimages disagree");
        for training in hashes {
            db.execute(
                "INSERT OR IGNORE INTO scope_training(scope,training) VALUES(?,?)",
                params![key, training],
            )?;
        }
        Ok(())
    }
    pub(super) fn verify_scope_registration_evidence(
        &self,
        s: &SignedNativeScopeRegistration,
    ) -> Result<()> {
        verified(s)?;
        let b = &s.body;
        ensure!(
            b.authority_instance == self.manifest.instance
                && b.authority_manifest_sha256 == self.manifest_sha
                && self.manifest.scopes.iter().any(|v| v
                    .numerical_contract_sha256
                    .contains(&b.scope.numerical_contract_sha256[0])),
            "scope changes original authority or numerical contract"
        );
        let overlay = crate::preparation_overlay::load(&b.root, &b.preparation_overlay, &b.run)?;
        ensure!(
            b.not_before_unix_ms >= overlay.not_before_unix_ms
                && b.expires_unix_ms <= overlay.expires_unix_ms,
            "scope registration exceeds signed preparation window"
        );
        let actual = crate::preparation_overlay::training_identities(
            &b.root,
            &b.preparation_overlay,
            &b.run,
        )?;
        ensure!(
            actual.into_iter().collect::<Vec<_>>() == b.scope.training_sha256,
            "scope training differs from actual signed overlay views"
        );
        let policy: Value = crate::native_genome::load(&b.root, &b.scientific_policy)?;
        ensure!(
            policy["schema"] == "jope.native_learning_policy.v1"
                && policy["sealed_partition"] == "meta_test"
                && policy["training_target"] == "continuation_only_v1",
            "native scope scientific policy differs"
        );
        Ok(())
    }
    /// Metadata-only preparation; this does not sign, register, import, or activate.
    pub fn native_scope_inspect(
        &mut self,
        root: &Path,
        overlay: &Artifact,
        run: &str,
    ) -> Result<Value> {
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), false, false)?;
        self.verify_tail()?;
        ensure!(
            self.manifest.schema == NATIVE_SCHEMA,
            "native scope inspection requires v6"
        );
        let training = crate::preparation_overlay::training_identities(root, overlay, run)?;
        let epoch:String=self.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta))",[],|r|r.get(0))?;
        Ok(
            json!({"schema":"jope.native_scope_inspection.v1","authority_root":self.root,"authority_instance":self.manifest.instance,
            "authority_manifest_sha256":self.manifest_sha,"authority_epoch":epoch,"journal_sequence":self.checkpoint.seq,
            "journal_tip_sha256":self.checkpoint.tip,"run":run,"root":root,"preparation_overlay":overlay,"training_sha256":training,
            "original_scopes":self.manifest.scopes,"authorization_issued":false,"final_tests_opened":false}),
        )
    }
    /// Adds a reviewed namespace only. Complete imported coverage and authenticated
    /// current-epoch native grants remain separately necessary for every numerical fit.
    pub fn register_native_scope(&mut self, s: &SignedNativeScopeRegistration) -> Result<String> {
        ensure!(
            self.manifest.schema == NATIVE_SCHEMA,
            "native scope registration requires original v6 store"
        );
        verified(s)?;
        let b = &s.body;
        let raw = encoded(s)?;
        let sha = hash(&raw);
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), true, false)?;
        self.verify_tail()?;
        if table_exists(&self.db)? {
            let previous: Option<String> = self
                .db
                .query_row(
                    "SELECT registration_sha FROM native_scopes WHERE scope=?",
                    [&b.scope.scope],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(old) = previous {
                ensure!(
                    old == sha,
                    "native scope name already has a different original registration"
                );
                self.registered_native_scope(&b.scope.scope)?;
                return Ok(old);
            }
        }
        ensure!(
            !self
                .manifest
                .scopes
                .iter()
                .any(|p| p.scope == b.scope.scope),
            "immutable original scope cannot be replaced"
        );
        self.verify_scope_registration_evidence(s)?;
        let epoch:String=self.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta))",[],|r|r.get(0))?;
        let now = super::super::native_scq::policy::now_ms()?;
        ensure!(
            b.authority_root == self.root
                && b.root.canonicalize()? == b.root
                && b.authority_epoch == epoch
                && b.journal_sequence == self.checkpoint.seq
                && b.journal_tip_sha256 == self.checkpoint.tip
                && b.not_before_unix_ms <= now
                && now < b.expires_unix_ms,
            "scope registration original head/epoch/window differs"
        );
        self.store_blob(&raw)?;
        self.mutation_under_lock(
            "native_scope_register",
            &json!({"scope":b.scope.scope,"registration_sha":sha}),
            |db| {
                db.execute_batch(TABLE_SQL)?;
                db.execute(
                    "INSERT INTO native_scopes(scope,registration_sha,json) VALUES(?,?,?)",
                    params![b.scope.scope, sha, serde_json::to_string(&b.scope)?],
                )?;
                for training in &b.scope.training_sha256 {
                    db.execute(
                        "INSERT INTO scope_training(scope,training) VALUES(?,?)",
                        params![b.scope.scope, training],
                    )?;
                }
                Ok((sha.clone(), true))
            },
        )
    }
}
#[cfg(test)]
mod tests {
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
            verify_scope_event_bindings(&event, &signed, &sha, &b.authority_manifest_sha256)
                .is_err()
        );
        event.seq -= 1;
        event.effects[1].row["training_json"] = json!("[]");
        assert!(
            verify_scope_event_bindings(&event, &signed, &sha, &b.authority_manifest_sha256)
                .is_err()
        );
    }
}
