//! W7-D6: repository verbs author a bounded-run contract and sign and verify its grants, so
//! nothing is hand-assembled. The authority key is read only from an owner-only (0600) file
//! that is not a symlink, never from arguments or the environment, and is never printed.
//! Synthetic identities (the fixture census); no host is contacted.
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    authoring::{self, ContractRequest, ParticipantRequest, WorkRequest, REQUEST_SCHEMA},
    contract::{BoundedRunContract, Ceilings, Counters, WorkKind},
    controller,
    participant::load_signing_key,
};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
};

/// PKCS#8 v1 Ed25519, as `openssl genpkey -algorithm ed25519` writes it (DER).
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
/// The fixture's grant-authority seed (`keys()` uses `[9; 32]`).
const AUTHORITY_SEED: [u8; 32] = [9; 32];

/// A request for the three fixture hosts of `census`, one GPU rank each.
fn request(k: &Keys) -> ContractRequest {
    let reference = contract(now_ms(), 3, k);
    ContractRequest {
        schema: REQUEST_SCHEMA.into(),
        run: "w7-authored-run".into(),
        attempt: 1,
        coordinator_host: "fixture-host-1".into(),
        coordinator_public_key: public(&k.coordinator),
        grant_authority_public_key: public(&k.authority),
        not_before_unix_ms: None,
        window_ms: 1_200_000,
        grant_ms: 900_000,
        telemetry: reference.telemetry.clone(),
        rendezvous: reference.rendezvous.clone(),
        artifact_closure: vec!["inputs/program.json".into(), "inputs/corpus.bin".into()],
        participants: (0..3)
            .map(|i| ParticipantRequest {
                host: format!("fixture-host-{i}"),
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
    }
}
/// The fixture census and a host input root holding the closure.
fn host(
    k: &Keys,
) -> (
    tempfile::TempDir,
    PathBuf,
    jope_parallel::native_loop::bounded::census::HostCensus,
) {
    let (t, root) = tempdir();
    write_closure(&root);
    let census = census(&contract(now_ms(), 3, k), now_ms());
    (t, root, census)
}

#[test]
fn authored_contracts_are_signed_and_verified_by_the_grant_authority() {
    let k = keys(3);
    let (_t, root, census) = host(&k);
    let now = now_ms();
    let unsigned = authoring::author(&request(&k), &census, &root, now).unwrap();
    // Identities, devices and closure digests come from the census and the files.
    assert_eq!(unsigned.participants.len(), 3);
    for (i, p) in unsigned.participants.iter().enumerate() {
        assert_eq!(p.host, census.hosts[i].identity);
        assert_eq!(p.device.as_ref().unwrap().uuid, format!("GPU-fixture-{i}"));
        assert_eq!(p.grant.body.expires_unix_ms, now + 900_000);
        assert!(p.grant.signature.is_empty());
    }
    let paths: Vec<_> = unsigned
        .artifact_closure
        .iter()
        .map(|e| e.path.clone())
        .collect();
    assert_eq!(
        paths,
        [
            PathBuf::from("inputs/corpus.bin"),
            PathBuf::from("inputs/program.json")
        ]
    );
    assert_eq!(
        unsigned.artifact_closure[0].sha256,
        sha(b"fixture corpus bytes")
    );
    assert_eq!(unsigned.work_census.len(), 9);
    assert_eq!(unsigned.original_deadline_unix_ms, now + 1_200_000);
    // Unsigned, it is not a plan.
    assert!(BoundedRunContract::parse(&unsigned.canonical_bytes().unwrap()).is_err());
    // Positive control: signed by the authority key file, it verifies and plans.
    let key = load_signing_key(&key_file(&root, "authority.key", AUTHORITY_SEED, 0o600)).unwrap();
    let signed = authoring::sign_grants(&unsigned, &key, now).unwrap();
    let bytes = signed.canonical_bytes().unwrap();
    let report = authoring::verify_grants(&bytes, now).unwrap();
    assert_eq!(report.plan_sha256, sha(&bytes));
    assert_eq!(report.grants.len(), 3);
    let (_t2, run) = tempdir();
    let plan = controller::plan(&run.join("run"), &bytes, now, None).unwrap();
    assert_eq!(plan.plan_sha256, report.plan_sha256);
    // The unsigned review names the same grant bodies the signature covers.
    let review = authoring::unsigned_report(&unsigned).unwrap();
    assert_eq!(review.grants, report.grants);
}

#[test]
fn a_wrong_key_a_tampered_grant_and_an_expired_window_are_refused() {
    let k = keys(3);
    let (_t, root, census) = host(&k);
    let now = now_ms();
    let unsigned = authoring::author(&request(&k), &census, &root, now).unwrap();
    // A key that is not the contract's grant authority signs nothing.
    let wrong = load_signing_key(&key_file(&root, "wrong.key", [8; 32], 0o600)).unwrap();
    let e = authoring::sign_grants(&unsigned, &wrong, now).unwrap_err();
    assert!(
        e.to_string().contains("not the contract's grant authority"),
        "{e}"
    );
    let key = load_signing_key(&key_file(&root, "authority.key", AUTHORITY_SEED, 0o600)).unwrap();
    let signed = authoring::sign_grants(&unsigned, &key, now).unwrap();
    // A grant altered after signing.
    let mut tampered = signed.clone();
    tampered.participants[1].grant.body.cpu_millicores += 1;
    let e = authoring::verify_grants(&serde_json::to_vec(&tampered).unwrap(), now).unwrap_err();
    assert!(e.to_string().contains("grant signature for rank 1"), "{e}");
    // A signature moved to another rank's grant.
    let mut swapped = signed.clone();
    swapped.participants[0].grant.signature = signed.participants[2].grant.signature.clone();
    assert!(authoring::verify_grants(&serde_json::to_vec(&swapped).unwrap(), now).is_err());
    // A window that has expired: verification after it, and signing after it.
    let later = signed.participants[0].grant.body.expires_unix_ms + 1;
    // After expiry the remaining window can no longer fund the ceilings either; Plan's
    // time-dependent validation refuses it.
    let e = authoring::verify_grants(&signed.canonical_bytes().unwrap(), later).unwrap_err();
    assert!(
        e.to_string().contains("already expired") || e.to_string().contains("cannot fund"),
        "{e}"
    );
    let e = authoring::sign_grants(&unsigned, &key, later).unwrap_err();
    assert!(e.to_string().contains("has already expired"), "{e}");
    // A contract that is already signed is not re-signed.
    let e = authoring::parse_unsigned(&signed.canonical_bytes().unwrap()).unwrap_err();
    assert!(e.to_string().contains("already signed"), "{e}");
}

#[test]
fn the_authority_key_file_must_be_owner_only_and_not_a_symlink() {
    let (_t, root) = tempdir();
    for mode in [0o640, 0o644, 0o604, 0o660] {
        let path = key_file(&root, &format!("key-{mode:o}"), AUTHORITY_SEED, mode);
        let e = load_signing_key(&path).unwrap_err();
        assert!(
            e.to_string().contains("readable by its owner only"),
            "{mode:o}: {e}"
        );
    }
    let key = key_file(&root, "authority.key", AUTHORITY_SEED, 0o600);
    load_signing_key(&key).unwrap();
    let link = root.join("link.key");
    std::os::unix::fs::symlink(&key, &link).unwrap();
    assert!(load_signing_key(&link).is_err());
    // Not an Ed25519 PKCS#8 key.
    let junk = root.join("junk.key");
    fs::write(&junk, b"not a key").unwrap();
    fs::set_permissions(&junk, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(load_signing_key(&junk).is_err());
}

#[test]
fn authoring_refuses_what_the_census_and_inputs_do_not_show() {
    let k = keys(3);
    let (_t, root, census) = host(&k);
    let now = now_ms();
    let refused = |edit: &dyn Fn(&mut ContractRequest), needle: &str| {
        let mut r = request(&k);
        edit(&mut r);
        let e = authoring::author(&r, &census, &root, now).unwrap_err();
        assert!(
            format!("{e:#}").contains(needle),
            "{e:#} (expected `{needle}`)"
        );
    };
    refused(
        &|r| r.participants[2].host = "fixture-host-9".into(),
        "is not in the census",
    );
    refused(
        &|r| r.participants[1].device_uuid = Some("GPU-elsewhere".into()),
        "is not in host `fixture-host-1`'s census",
    );
    refused(
        &|r| r.artifact_closure.push("inputs/absent.bin".into()),
        "inputs/absent.bin",
    );
    refused(
        &|r| r.artifact_closure.push("inputs/corpus.bin".into()),
        "listed once",
    );
    refused(
        &|r| r.artifact_closure.push("../outside.bin".into()),
        "must be relative",
    );
    refused(&|r| r.grant_ms = r.window_ms + 1, "within the run window");
    refused(
        &|r| r.schema = "jope.something_else.v1".into(),
        "unsupported contract request schema",
    );
    // The contract's own rules still apply: the coordinator key cannot be the authority.
    refused(
        &|r| r.coordinator_public_key = r.grant_authority_public_key.clone(),
        "coordinator cannot hold the grant authority key",
    );
}

#[test]
fn the_cli_authors_signs_and_verifies_without_printing_the_key() {
    let k = keys(3);
    let (_t, root, census) = host(&k);
    let bin = env!("CARGO_BIN_EXE_jope-native-loop");
    let work = root.join("work");
    fs::create_dir(&work).unwrap();
    fs::write(
        work.join("request.json"),
        serde_json::to_vec(&request(&k)).unwrap(),
    )
    .unwrap();
    fs::write(
        work.join("census.json"),
        serde_json::to_vec(&census).unwrap(),
    )
    .unwrap();
    let key = key_file(&work, "authority.key", AUTHORITY_SEED, 0o600);
    let run = |args: &[&str]| {
        let out = Command::new(bin).args(args).env_clear().output().unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        // Neither the seed nor the key file bytes are ever printed.
        assert!(!text.contains(&hex::encode(AUTHORITY_SEED)), "{text}");
        assert!(
            !text.contains(&hex::encode(pkcs8(AUTHORITY_SEED))),
            "{text}"
        );
        (out.status.success(), text)
    };
    let p = |p: &Path| p.display().to_string();
    let (ok, text) = run(&[
        "contract",
        "author",
        "--request",
        &p(&work.join("request.json")),
        "--census",
        &p(&work.join("census.json")),
        "--input-root",
        &p(&root),
        "--out",
        &p(&work.join("unsigned.json")),
    ]);
    assert!(ok, "{text}");
    let (ok, text) = run(&[
        "grant",
        "sign",
        "--contract",
        &p(&work.join("unsigned.json")),
        "--key",
        &p(&key),
        "--out",
        &p(&work.join("contract.json")),
    ]);
    assert!(ok, "{text}");
    let (ok, text) = run(&[
        "grant",
        "verify",
        "--contract",
        &p(&work.join("contract.json")),
    ]);
    assert!(ok, "{text}");
    let report: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        report["plan_sha256"],
        sha(&fs::read(work.join("contract.json")).unwrap())
    );
    // Outputs are never replaced.
    let (ok, text) = run(&[
        "grant",
        "sign",
        "--contract",
        &p(&work.join("unsigned.json")),
        "--key",
        &p(&key),
        "--out",
        &p(&work.join("contract.json")),
    ]);
    assert!(!ok && text.contains("exists"), "{text}");
    // A key with a bad mode is refused, and a verified contract is not re-signed.
    fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).unwrap();
    let (ok, text) = run(&[
        "grant",
        "sign",
        "--contract",
        &p(&work.join("unsigned.json")),
        "--key",
        &p(&key),
        "--out",
        &p(&work.join("again.json")),
    ]);
    assert!(!ok && text.contains("readable by its owner only"), "{text}");
    assert!(!work.join("again.json").exists());
    // The key cannot be given on the command line or in the environment: there is no such
    // argument.
    let (ok, _) = run(&[
        "grant",
        "sign",
        "--contract",
        &p(&work.join("unsigned.json")),
        "--key-hex",
        "00",
        "--out",
        &p(&work.join("x.json")),
    ]);
    assert!(!ok);
}
