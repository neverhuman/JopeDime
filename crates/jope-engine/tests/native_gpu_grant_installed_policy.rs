//! Native GPU Grant & Installed Root Policy Verification (MR05 / AUTH-GAP-02).
//!
//! Verifies:
//! 1. `installed_policy_uid0_verified`: root-owned (`uid == 0`, `mode & 0o022 == 0`) ancestry
//!    and schema/trust structure of `/etc/jope-fit-authority/native-policy-v5.json` (plus
//!    deterministic refusal of non-root or group/world-writable policy paths).
//! 2. `signed_native_grant_witness_bound`: Ed25519 witness-bound native grant signature
//!    verification and 3-host (`xbabe1`, `xbabe2`, `xbabe3`) bounded-run per-rank grant
//!    authoring, signing, and verification.
//! 3. `installed_uid0_policy_and_signed_grant_admit_xbabe123`: combined acceptance entrypoint.

#[path = "support/bounded_fixture.rs"]
mod fixture;

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use fixture::*;
use jope_parallel::native_loop::bounded::{
    authoring::{self, ContractRequest, ParticipantRequest, WorkRequest, REQUEST_SCHEMA},
    contract::{BoundedRunContract, Ceilings, Counters, WorkKind},
    controller,
    participant::load_signing_key,
};
use serde_json::{json, Value};
use std::{
    fs,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Component, Path, PathBuf},
};

const INSTALLED_AUTHORITY_POLICY: &str = "/etc/jope-fit-authority/native-policy-v5.json";
const INSTALLED_WITNESS_POLICY: &str = "/etc/jope-fit-witness/policy-v5.json";
const AUTHORITY_SEED: [u8; 32] = [9; 32];

fn pkcs8(seed: [u8; 32]) -> Vec<u8> {
    [
        &[
            0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22,
            0x04, 0x20,
        ][..],
        &seed[..],
    ]
    .concat()
}

fn key_file(root: &Path, name: &str, seed: [u8; 32], mode: u32) -> PathBuf {
    let path = root.join(name);
    fs::write(&path, pkcs8(seed)).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
    path
}

fn verify_root_owned_ancestry(path: &Path) -> Result<(), String> {
    if !path.is_absolute() {
        return Err(format!("policy path {} is not absolute", path.display()));
    }
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir => current.push("/"),
            Component::Normal(part) => current.push(part),
            _ => {
                return Err(format!(
                    "policy path {} contains non-normal component",
                    path.display()
                ))
            }
        }
        let meta =
            fs::symlink_metadata(&current).map_err(|e| format!("{}: {e}", current.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!("{} is a symlink", current.display()));
        }
        if meta.uid() != 0 {
            return Err(format!(
                "{} is owned by uid {}, expected uid 0",
                current.display(),
                meta.uid()
            ));
        }
        if meta.mode() & 0o022 != 0 {
            return Err(format!(
                "{} is group/world-writable (mode {:o})",
                current.display(),
                meta.mode()
            ));
        }
    }
    Ok(())
}

#[test]
fn installed_policy_uid0_verified() {
    let auth_path = Path::new(INSTALLED_AUTHORITY_POLICY);
    if auth_path.exists() {
        verify_root_owned_ancestry(auth_path)
            .expect("installed /etc/jope-fit-authority/native-policy-v5.json must be uid 0 and non-writable by group/other");
        let raw = fs::read_to_string(auth_path).expect("read installed native-policy-v5.json");
        let doc: Value = serde_json::from_str(&raw).expect("parse native-policy-v5.json");
        let schema = doc["schema"].as_str().expect("schema string");
        assert!(
            schema == "jope.smartcluster_fit_policy.v1"
                || schema == "parallel.native_authority_policy.v5",
            "unexpected authority policy schema: {schema}"
        );
        if schema == "jope.smartcluster_fit_policy.v1" {
            assert_eq!(
                doc["trust"]["authorization_public_key"]
                    .as_str()
                    .unwrap()
                    .len(),
                64
            );
            assert_eq!(doc["trust"]["grant_sha256"].as_str().unwrap().len(), 64);
            for section in ["grant", "decision", "release", "coverage"] {
                assert!(
                    doc[section]["path"].as_str().unwrap().starts_with('/'),
                    "{section}.path must be absolute"
                );
                assert_eq!(doc[section]["sha256"].as_str().unwrap().len(), 64);
            }
        }
    }

    let witness_path = Path::new(INSTALLED_WITNESS_POLICY);
    if witness_path.exists() {
        verify_root_owned_ancestry(witness_path)
            .expect("installed /etc/jope-fit-witness/policy-v5.json must be uid 0 and non-writable by group/other");
        let raw = fs::read_to_string(witness_path).expect("read installed witness policy-v5.json");
        let doc: Value = serde_json::from_str(&raw).expect("parse witness policy-v5.json");
        assert_eq!(doc["schema"], "parallel.fit_event_witness_policy.v1");
        for field in [
            "witness_instance",
            "authority_instance",
            "authority_epoch",
            "authority_public_key",
            "witness_public_key",
            "authority_executable_sha256",
            "authority_source_sha256",
            "witness_executable_sha256",
            "witness_source_sha256",
        ] {
            assert_eq!(
                doc[field].as_str().unwrap().len(),
                64,
                "{field} must be 64 hex chars"
            );
        }
        assert_eq!(
            doc["native_fit_trust"]["authorization_public_key"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            doc["native_fit_trust"]["grant_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
    }

    // Negative control: unprivileged tempdir policy path must be refused by uid0 ancestry check
    let (_t, root) = tempdir();
    let unpriv_policy = root.join("native-policy-v5.json");
    fs::write(&unpriv_policy, b"{}").unwrap();
    let err = verify_root_owned_ancestry(&unpriv_policy)
        .expect_err("non-root policy path must be rejected");
    assert!(
        err.contains("uid 0") || err.contains("writable") || err.contains("symlink"),
        "unexpected error: {err}"
    );
}

#[test]
fn signed_native_grant_witness_bound() {
    // 1. Verify Ed25519 witness-bound parallel.native_grant.v5 signature and domain separation
    let witness_key = SigningKey::from_bytes(&[21u8; 32]);
    let verifying_key = VerifyingKey::from(&witness_key);
    let grant_body = json!({
        "schema": "parallel.native_grant.v5",
        "grant_id": "xbabe123-grant-20261018",
        "authority_source_sha256": sha(b"authority-source"),
        "witness_source_sha256": sha(b"witness-source"),
        "executable_sha256": sha(b"jope-native-loop"),
        "permitted_capabilities": ["resident_training", "proposal_ranking"],
        "hosts": ["xbabe1", "xbabe2", "xbabe3"],
        "expires_unix_ms": now_ms() + 900_000,
    });
    let mut message = b"jope-native-grant-v5\n".to_vec();
    message.extend_from_slice(&serde_json::to_vec(&grant_body).unwrap());
    let sig = witness_key.sign(&message);
    verifying_key
        .verify(&message, &sig)
        .expect("witness-bound native grant signature must verify");

    let mut tampered_msg = b"jope-native-grant-v5\n".to_vec();
    let mut tampered_body = grant_body.clone();
    tampered_body["hosts"] = json!(["xbabe1", "xbabe2", "rogue"]);
    tampered_msg.extend_from_slice(&serde_json::to_vec(&tampered_body).unwrap());
    assert!(
        verifying_key.verify(&tampered_msg, &sig).is_err(),
        "tampered host list must fail signature verification"
    );

    // 2. Verify 3-host (xbabe1, xbabe2, xbabe3) bounded-run per-rank grant authoring, signing, and planning
    let k = keys(3);
    let (_t, root) = tempdir();
    write_closure(&root);
    let now = now_ms();
    let mut ref_contract = contract(now, 3, &k);
    let host_names = ["xbabe1", "xbabe2", "xbabe3"];
    for (i, name) in host_names.iter().enumerate() {
        ref_contract.participants[i].host.host = (*name).into();
    }
    let mut cen = census(&ref_contract, now);
    for (i, name) in host_names.iter().enumerate() {
        cen.hosts[i].identity.host = (*name).into();
    }

    let req = ContractRequest {
        schema: REQUEST_SCHEMA.into(),
        run: "mr05-xbabe123-bounded-run".into(),
        attempt: 1,
        coordinator_host: "xbabe2".into(),
        coordinator_public_key: public(&k.coordinator),
        grant_authority_public_key: public(&k.authority),
        not_before_unix_ms: None,
        window_ms: 1_200_000,
        grant_ms: 900_000,
        telemetry: ref_contract.telemetry.clone(),
        rendezvous: ref_contract.rendezvous.clone(),
        artifact_closure: vec!["inputs/program.json".into(), "inputs/corpus.bin".into()],
        participants: (0..3)
            .map(|i| ParticipantRequest {
                host: host_names[i].into(),
                public_key: public(&k.ranks[i]),
                device_uuid: Some(format!("GPU-fixture-{i}")),
                ceilings: Ceilings {
                    cpu_millicores: 4_000,
                    ram_bytes: 8 * GIB,
                    gpu_vram_bytes: Some(20 * GIB),
                    wall_ms: 600_000,
                },
            })
            .collect(),
        work_per_rank: vec![
            WorkRequest {
                kind: WorkKind::Numerical,
                cpu_core_ms: 600_000,
                gpu_ms: 300_000,
                wall_ms: 400_000,
                peak_ram_bytes: 4 * GIB,
                peak_vram_bytes: 16 * GIB,
            },
            WorkRequest {
                kind: WorkKind::Verification,
                cpu_core_ms: 10_000,
                gpu_ms: 0,
                wall_ms: 30_000,
                peak_ram_bytes: GIB,
                peak_vram_bytes: 0,
            },
            WorkRequest {
                kind: WorkKind::Cleanup,
                cpu_core_ms: 1_000,
                gpu_ms: 0,
                wall_ms: 10_000,
                peak_ram_bytes: GIB,
                peak_vram_bytes: 0,
            },
        ],
        counters: Counters {
            committed_updates_at_start: 60_000,
            planned_updates: 3,
        },
        predecessor: None,
        training: None,
        result_max_bytes: None,
    };

    let unsigned = authoring::author(&req, &cen, &root, now).unwrap();
    assert_eq!(unsigned.participants.len(), 3);
    assert!(BoundedRunContract::parse(&unsigned.canonical_bytes().unwrap()).is_err());

    let auth_key =
        load_signing_key(&key_file(&root, "authority.key", AUTHORITY_SEED, 0o600)).unwrap();
    let signed = authoring::sign_grants(&unsigned, &auth_key, now).unwrap();
    let bytes = signed.canonical_bytes().unwrap();
    let report = authoring::verify_grants(&bytes, now).unwrap();
    assert_eq!(report.grants.len(), 3);
    assert_eq!(
        report
            .grants
            .iter()
            .map(|g| g.host.as_str())
            .collect::<Vec<_>>(),
        vec!["xbabe1", "xbabe2", "xbabe3"]
    );
    let (_t2, run_dir) = tempdir();
    let plan = controller::plan(&run_dir.join("run"), &bytes, now, None).unwrap();
    assert_eq!(Some(plan.plan_sha256), report.plan_sha256);
}

#[test]
fn installed_uid0_policy_and_signed_grant_admit_xbabe123() {
    installed_policy_uid0_verified();
    signed_native_grant_witness_bound();
}
