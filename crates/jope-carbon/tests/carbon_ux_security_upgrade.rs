//! Integration tests for Carbon security split, passive download telemetry, cross-OS installers,
//! structured observability/redaction, tabular ergonomics, custom fitness (`aux` columns), and
//! zero-knowledge `explain` / `export` / `predict --out -` (`ADR-0007`).

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_ASN1_SIGNING};
use base64::Engine as _;
use flate2::write::GzEncoder;
use flate2::Compression;
use jope_carbon::observability::{
    redact_secrets, EXIT_AUTH_EXPIRED, EXIT_DATA_VALIDATION, EXIT_USAGE,
};
use jope_carbon::{
    load_delimited_table_with_options, parse_flexible_numeric_f64,
    prepare_encoded_training_set_with_options, CarbonFeatureTransform, CarbonTrainingPrepOptions,
};
use jope_prep::objective::ObjectiveSpec;

const SPKI_P256_PREFIX: [u8; 26] = [
    0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08, 0x2a,
    0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03, 0x42, 0x00,
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn generate_test_spki_keypair() -> (EcdsaKeyPair, String) {
    let rng = SystemRandom::new();
    let pkcs8 =
        EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, &rng).expect("generate p256");
    let kp =
        EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, pkcs8.as_ref()).expect("parse");
    let mut spki = Vec::with_capacity(91);
    spki.extend_from_slice(&SPKI_P256_PREFIX);
    spki.extend_from_slice(kp.public_key().as_ref());
    let b64 = base64::engine::general_purpose::STANDARD.encode(&spki);
    // Deliberately use CRLF line endings to verify both install.ps1 and install.sh normalize CRLF!
    let pem = format!(
        "-----BEGIN PUBLIC KEY-----\r\n{}\r\n{}\r\n-----END PUBLIC KEY-----\r\n",
        &b64[..64],
        &b64[64..]
    );
    (kp, pem)
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn build_stored_zip_single_file(entry_name: &str, payload: &[u8]) -> Vec<u8> {
    let name_bytes = entry_name.as_bytes();
    let crc = crc32_ieee(payload);
    let size = u32::try_from(payload.len()).expect("zip payload < 4 GiB");
    let name_len = u16::try_from(name_bytes.len()).expect("zip name len");

    let mut out = Vec::with_capacity(payload.len() + 256);
    // Local file header
    out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // stored
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(name_bytes);
    out.extend_from_slice(payload);

    let cd_offset = u32::try_from(out.len()).expect("cd offset");
    // Central directory header
    out.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0o1_007_550_000_u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(name_bytes);

    let cd_size = u32::try_from(out.len()).expect("end") - cd_offset;
    // End of central directory
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn build_single_file_targz(entry_name: &str, payload: &[u8]) -> Vec<u8> {
    let mut tar = Vec::with_capacity(512 + payload.len().div_ceil(512) * 512 + 1024);
    let mut hdr = [0u8; 512];
    let nb = entry_name.as_bytes();
    hdr[..nb.len()].copy_from_slice(nb);
    hdr[100..108].copy_from_slice(b"0000755\0");
    hdr[108..116].copy_from_slice(b"0000000\0");
    hdr[116..124].copy_from_slice(b"0000000\0");
    let size_octal = format!("{:011o}\0", payload.len());
    hdr[124..136].copy_from_slice(size_octal.as_bytes());
    hdr[136..148].copy_from_slice(b"14742000000\0");
    hdr[148..156].fill(b' ');
    hdr[156] = b'0';
    hdr[257..263].copy_from_slice(b"ustar\0");
    hdr[263..265].copy_from_slice(b"00");
    let csum: u32 = hdr.iter().map(|&b| u32::from(b)).sum();
    let csum_str = format!("{csum:06o}\0 ");
    hdr[148..156].copy_from_slice(csum_str.as_bytes());
    tar.extend_from_slice(&hdr);
    tar.extend_from_slice(payload);
    let rem = payload.len() % 512;
    if rem != 0 {
        tar.resize(tar.len() + (512 - rem), 0);
    }
    tar.resize(tar.len() + 1024, 0);

    let mut gz = GzEncoder::new(Vec::new(), Compression::fast());
    gz.write_all(&tar).expect("gzip tar");
    gz.finish().expect("finish gzip")
}

type MockHttpRoute = (u16, Vec<(String, String)>, Vec<u8>);

fn spawn_static_http_server(
    routes: BTreeMap<String, MockHttpRoute>,
    max_requests: usize,
) -> (String, Arc<AtomicUsize>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("local addr");
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_clone = Arc::clone(&hits);
    let handle = thread::spawn(move || {
        for _ in 0..max_requests {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut req_bytes = Vec::with_capacity(1024);
            let mut chunk = [0u8; 1024];
            loop {
                let n = stream.read(&mut chunk).unwrap_or(0);
                if n == 0 {
                    break;
                }
                req_bytes.extend_from_slice(&chunk[..n]);
                if req_bytes.windows(4).any(|w| w == b"\r\n\r\n") || req_bytes.len() >= 16_384 {
                    break;
                }
            }
            if req_bytes.is_empty() {
                continue;
            }
            let req = String::from_utf8_lossy(&req_bytes);
            let first_line = req.lines().next().unwrap_or("");
            let path = first_line.split_whitespace().nth(1).unwrap_or("/");
            if path == "/__stop" {
                break;
            }
            hits_clone.fetch_add(1, Ordering::SeqCst);
            let (status, extra_headers, body) = routes
                .get(path)
                .cloned()
                .unwrap_or_else(|| (404, Vec::new(), format!("not found: {path}").into_bytes()));
            let reason = match status {
                200 => "OK",
                302 => "Found",
                503 => "Service Unavailable",
                _ => "Not Found",
            };
            let mut resp = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
                body.len()
            );
            for (k, v) in extra_headers {
                resp.push_str(&format!("{k}: {v}\r\n"));
            }
            resp.push_str("\r\n");
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
            let _ = stream.shutdown(std::net::Shutdown::Write);
        }
    });
    (format!("http://{addr}"), hits, handle)
}

fn stop_http_server(base_url: &str, handle: thread::JoinHandle<()>) {
    if let Some(host_port) = base_url.strip_prefix("http://") {
        if let Ok(mut s) = std::net::TcpStream::connect(host_port) {
            let _ = s.write_all(b"GET /__stop HTTP/1.1\r\nHost: localhost\r\n\r\n");
        }
    }
    let _ = handle.join();
}

#[test]
fn thin_cli_observability_secret_redaction_fitness_and_error_json() {
    let fake_ctk = format!("ctk1.{}.{}", "eyJ0IjoiYSJ9", "cd".repeat(8));
    let fake_sig = "12".repeat(32);
    let fake_bearer = format!("{}_{}", "demo_bearer", "val_99");
    let fake_dek = "ef".repeat(32);
    let raw = format!(
        "hdr_tok={fake_ctk} url=https://s3/x?X-Amz-Signature={fake_sig}&ok=1 hdr=Authorization: Bearer {fake_bearer} dek_hex={fake_dek}"
    );
    let scrubbed = redact_secrets(&raw);
    assert!(
        !scrubbed.contains("eyJ0IjoiYSJ9"),
        "ctk1 leaked: {scrubbed}"
    );
    assert!(
        !scrubbed.contains(&fake_sig),
        "SigV4 signature leaked: {scrubbed}"
    );
    assert!(
        !scrubbed.contains(&fake_bearer),
        "Bearer token leaked: {scrubbed}"
    );
    assert!(
        !scrubbed.contains(&fake_dek),
        "64-hex secret leaked: {scrubbed}"
    );

    let bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    let tmp = tempfile::tempdir().expect("tempdir");

    // 1. Unknown verb returns EXIT_USAGE (2)
    let bad_verb = Command::new(&bin)
        .arg("nonexistent-subcommand")
        .output()
        .expect("run unknown verb");
    assert_eq!(bad_verb.status.code(), Some(i32::from(EXIT_USAGE)));

    // 2. Fail-closed CARBON_CA_BUNDLE validation
    let bad_tls = Command::new(&bin)
        .arg("version")
        .env("CARBON_CA_BUNDLE", tmp.path().join("missing-ca.pem"))
        .output()
        .expect("run bad tls");
    assert!(
        !bad_tls.status.success(),
        "missing CARBON_CA_BUNDLE must fail closed"
    );
    let stderr_tls = String::from_utf8_lossy(&bad_tls.stderr);
    assert!(
        stderr_tls.contains("CARBON_CA_BUNDLE"),
        "expected CARBON_CA_BUNDLE guidance, got: {stderr_tls}"
    );

    // 3. `carbon fitness list --json`, `validate`, `schema`, and `evaluate`
    let list_out = Command::new(&bin)
        .args(["fitness", "list", "--json"])
        .output()
        .expect("fitness list");
    assert!(list_out.status.success());
    let list_json: serde_json::Value =
        serde_json::from_slice(&list_out.stdout).expect("parse fitness list json");
    let count = list_json.as_array().map_or(0, Vec::len);
    assert!(count >= 45, "expected >=45 objective presets, got {count}");

    let schema_out = Command::new(&bin)
        .args(["fitness", "schema", "--json"])
        .output()
        .expect("fitness schema");
    assert!(schema_out.status.success());
    let schema_json: serde_json::Value =
        serde_json::from_slice(&schema_out.stdout).expect("parse schema json");
    assert!(schema_json.is_object());

    let spec_file = tmp.path().join("weighted.json");
    fs::write(
        &spec_file,
        r#"{"metric":"weighted","weights":{"op":"column","name":"sample_w"},"fitness":{"metric":"rmse"}}"#,
    )
    .expect("write spec");
    let val_out = Command::new(&bin)
        .args([
            "fitness",
            "validate",
            &format!("@{}", spec_file.display()),
            "--json",
        ])
        .output()
        .expect("fitness validate");
    assert!(val_out.status.success());
    let val_json: serde_json::Value =
        serde_json::from_slice(&val_out.stdout).expect("parse validate json");
    assert_eq!(val_json["valid"], true);
    assert_eq!(val_json["aux_columns"], serde_json::json!(["sample_w"]));

    let eval_csv = tmp.path().join("eval.csv");
    fs::write(
        &eval_csv,
        "y,yhat,sample_w\n10.0,11.0,2.0\n20.0,19.0,1.0\n30.0,30.5,3.0\n",
    )
    .expect("write eval csv");
    let eval_out = Command::new(&bin)
        .args([
            "fitness",
            "evaluate",
            "--spec",
            &format!("@{}", spec_file.display()),
            "--csv",
            eval_csv.to_str().expect("utf8"),
            "--target",
            "y",
            "--pred",
            "yhat",
            "--json",
        ])
        .output()
        .expect("fitness evaluate");
    assert!(
        eval_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_json: serde_json::Value =
        serde_json::from_slice(&eval_out.stdout).expect("parse eval json");
    assert!(eval_json["loss"].as_f64().is_some_and(f64::is_finite));

    // 4. Structured `carbon.error.v1` JSON stdout envelope + semantic exit code on `--json` failure
    let missing_fit = Command::new(&bin)
        .args([
            "fit",
            tmp.path().join("missing.csv").to_str().expect("utf8"),
            "--target",
            "y",
            "--dry-run",
            "--json",
        ])
        .output()
        .expect("missing fit json");
    assert_eq!(
        missing_fit.status.code(),
        Some(i32::from(EXIT_DATA_VALIDATION))
    );
    let err_env: serde_json::Value =
        serde_json::from_slice(&missing_fit.stdout).expect("parse carbon.error.v1 stdout");
    assert_eq!(err_env["schema"], "carbon.error.v1");
    assert_eq!(
        err_env["error"]["exit_code"],
        u64::from(EXIT_DATA_VALIDATION)
    );

    // 5. Expired token on `carbon whoami` returns EXIT_AUTH_EXPIRED (9)
    let home_dir = tmp.path().join("home_expired");
    fs::create_dir_all(&home_dir).expect("mkdir home");
    let local_manifest = tmp.path().join("local-clouds.json");
    fs::write(
        &local_manifest,
        r#"{"schema":"jope.carbon.clouds.v1","clouds":{}}"#,
    )
    .expect("write local manifest");
    let expired_payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(r#"{"tenant_id":"t_exp","tier":"free","exp_epoch_secs":1700000000,"admin":false}"#);
    let expired_token = format!("ctk1.{expired_payload}.{}", "ab".repeat(32));
    let whoami_exp = Command::new(&bin)
        .args([
            "whoami",
            "--clouds-manifest",
            local_manifest.to_str().expect("utf8"),
            "--token",
            &expired_token,
            "--json",
        ])
        .env("HOME", &home_dir)
        .env("USERPROFILE", &home_dir)
        .env("CARBON_CONFIG_DIR", home_dir.join(".config/jopedime"))
        .output()
        .expect("whoami expired");
    assert_eq!(whoami_exp.status.code(), Some(i32::from(EXIT_AUTH_EXPIRED)));
}

#[test]
fn tabular_compressed_utf16_jsonl_currency_dates_inf_and_offline_dry_run() {
    assert_eq!(parse_flexible_numeric_f64("$1,200.50"), Some(1200.5));
    assert_eq!(parse_flexible_numeric_f64("($450.00)"), Some(-450.0));
    assert_eq!(parse_flexible_numeric_f64("12.5%"), Some(0.125));

    let tmp = tempfile::tempdir().expect("tempdir");

    // 1. UTF-16LE BOM CSV
    let csv_text = "price,pct,dt,w,y\n\"$1,000.00\",10%,2026-01-05,1.0,10.5\n($200.00),25%,2026-01-06,2.0,20.5\n";
    let mut utf16_bytes = vec![0xFFu8, 0xFEu8];
    for unit in csv_text.encode_utf16() {
        utf16_bytes.extend_from_slice(&unit.to_le_bytes());
    }
    let utf16_path = tmp.path().join("utf16.csv");
    fs::write(&utf16_path, &utf16_bytes).expect("write utf16");
    let tbl_utf16 = load_delimited_table_with_options(&utf16_path, None, true, &[], false)
        .expect("load utf16le bom csv");
    assert_eq!(tbl_utf16.rows.len(), 2);
    assert_eq!(tbl_utf16.headers, vec!["price", "pct", "dt", "w", "y"]);

    // 2. JSONL / NDJSON
    let jsonl_path = tmp.path().join("rows.jsonl");
    let mut jsonl = String::new();
    for i in 0..24 {
        let day = (i % 28) + 1;
        let y = f64::from(i) * 2.5 + 1.0;
        jsonl.push_str(&format!(
            r#"{{"price":"${}.00","pct":"{}%","dt":"2026-03-{day:02}","w":{},"y":{y}}}"#,
            100 + i * 10,
            5 + (i % 10),
            1 + (i % 3)
        ));
        jsonl.push('\n');
    }
    fs::write(&jsonl_path, &jsonl).expect("write jsonl");
    let tbl_jsonl =
        load_delimited_table_with_options(&jsonl_path, None, true, &[], false).expect("load jsonl");
    assert_eq!(tbl_jsonl.rows.len(), 24);

    // 3. Fail-closed `Inf` on feature cells unless `--inf-as-nan` is enabled
    let inf_csv = tmp.path().join("has_inf.csv");
    let mut inf_content = String::from("x1,x2,y\n");
    for i in 0..20 {
        let x1 = if i == 3 {
            "Inf".to_owned()
        } else {
            format!("{i}.0")
        };
        inf_content.push_str(&format!("{x1},1.5,{}.0\n", i * 2 + 1));
    }
    fs::write(&inf_csv, &inf_content).expect("write inf csv");
    let err_inf = load_delimited_table_with_options(&inf_csv, None, true, &[], false)
        .and_then(|tbl| {
            prepare_encoded_training_set_with_options(
                &tbl,
                &CarbonTrainingPrepOptions {
                    requested_target: Some("y"),
                    seed: 42,
                    holdout_fraction: 0.2,
                    max_encoded_features: 256,
                    objective_spec: None,
                    split_mode: None,
                    calibration_mode: None,
                    reject_constant_target: true,
                },
            )
        })
        .expect_err("Inf must fail closed by default");
    assert!(
        err_inf.contains("--inf-as-nan") || err_inf.to_lowercase().contains("inf"),
        "expected --inf-as-nan guidance: {err_inf}"
    );
    let inf_tbl_ok = load_delimited_table_with_options(&inf_csv, None, true, &[], true)
        .expect("load with inf_as_nan=true");
    let ok_inf = prepare_encoded_training_set_with_options(
        &inf_tbl_ok,
        &CarbonTrainingPrepOptions {
            requested_target: Some("y"),
            seed: 42,
            holdout_fraction: 0.2,
            max_encoded_features: 256,
            objective_spec: None,
            split_mode: None,
            calibration_mode: None,
            reject_constant_target: true,
        },
    )
    .expect("inf_as_nan opt-in succeeds");
    assert_eq!(ok_inf.prepared.feature_names.len(), 2);

    // 4. Constant-`y` pre-upload refusal
    let const_csv = tmp.path().join("const_y.csv");
    let mut const_body = String::from("x1,y\n");
    for i in 0..20 {
        const_body.push_str(&format!("{i}.0,42.0\n"));
    }
    fs::write(&const_csv, &const_body).expect("write const csv");
    let const_tbl =
        load_delimited_table_with_options(&const_csv, None, true, &[], false).expect("load const");
    let const_err = prepare_encoded_training_set_with_options(
        &const_tbl,
        &CarbonTrainingPrepOptions {
            requested_target: Some("y"),
            seed: 42,
            holdout_fraction: 0.2,
            max_encoded_features: 256,
            objective_spec: None,
            split_mode: None,
            calibration_mode: None,
            reject_constant_target: true,
        },
    )
    .expect_err("constant target y must fail closed before upload");
    assert!(const_err.to_lowercase().contains("constant"));

    // 5. `.csv.gz` + ISO-8601 date expansion + aux column exclusion + offline `carbon fit --dry-run`
    let gz_path = tmp.path().join("train.csv.gz");
    let mut raw_csv = String::from("price,pct,dt,w,y\n");
    for i in 0..30 {
        let day = (i % 28) + 1;
        let price = 100 + i * 15;
        let pct = 10 + (i % 7);
        let w = 1 + (i % 4);
        let y = f64::from(price) * 0.4 + f64::from(pct) * 1.2 + f64::from(i);
        raw_csv.push_str(&format!(
            "\"${price}.00\",{pct}%,2026-04-{day:02},{w}.0,{y:.4}\n"
        ));
    }
    {
        let mut gz = GzEncoder::new(Vec::new(), Compression::fast());
        gz.write_all(raw_csv.as_bytes()).expect("gz write");
        fs::write(&gz_path, gz.finish().expect("gz finish")).expect("save gz");
    }
    let gz_tbl =
        load_delimited_table_with_options(&gz_path, None, true, &[], false).expect("load .csv.gz");
    let weighted_json =
        r#"{"metric":"weighted","weights":{"op":"column","name":"w"},"fitness":{"metric":"rmse"}}"#;
    let weighted_spec = ObjectiveSpec::parse(weighted_json).expect("spec");
    let encoded = prepare_encoded_training_set_with_options(
        &gz_tbl,
        &CarbonTrainingPrepOptions {
            requested_target: Some("y"),
            seed: 42,
            holdout_fraction: 0.2,
            max_encoded_features: 256,
            objective_spec: Some(&weighted_spec),
            split_mode: Some("forward"),
            calibration_mode: Some("affine"),
            reject_constant_target: true,
        },
    )
    .expect("encode with date + aux + forward split");
    assert_eq!(encoded.prepared.aux_column_names, vec!["w".to_owned()]);
    assert!(!encoded.prepared.feature_names.iter().any(|n| n == "w"));
    assert!(
        encoded
            .encoder
            .transforms
            .iter()
            .any(|t| matches!(t, CarbonFeatureTransform::DateIso { .. })),
        "ISO-8601 column dt must expand via FIT-safe DateIso transform"
    );

    // Offline CLI `--dry-run` with unreachable endpoint (`127.0.0.1:9`) must succeed with zero network!
    let spec_file = tmp.path().join("w_obj.json");
    fs::write(&spec_file, weighted_json).expect("write spec");
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    let dry_out = Command::new(&bin)
        .args([
            "fit",
            gz_path.to_str().expect("utf8"),
            "--target",
            "y",
            "--fitness",
            &format!("@{}", spec_file.display()),
            "--split",
            "forward",
            "--calibration",
            "affine",
            "--dry-run",
            "--json",
        ])
        .env("CARBON_ENDPOINT", "http://127.0.0.1:9")
        .output()
        .expect("run offline dry-run");
    assert!(
        dry_out.status.success(),
        "offline --dry-run failed: {}",
        String::from_utf8_lossy(&dry_out.stderr)
    );
    let dry_json: serde_json::Value =
        serde_json::from_slice(&dry_out.stdout).expect("parse dry-run json");
    assert_eq!(dry_json["selected_endpoint"], "dry-run://local-preflight");
    assert_eq!(dry_json["run_id"], "run_dry_run_preflight");
    assert_eq!(dry_json["neural_guidance_mode"], "dry_run_validated");
    assert_eq!(dry_json["total_rows"], 30);
}

#[test]
fn cross_os_installer_signed_redirect_tamper_and_uninstall() {
    let root = repo_root();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    let bin_bytes = fs::read(&bin).expect("read compiled carbon binary");
    let tmp = tempfile::tempdir().expect("tempdir");

    let (kp, crlf_pubkey_pem) = generate_test_spki_keypair();
    let pubkey_path = tmp.path().join("test-signing-crlf.pub");
    fs::write(&pubkey_path, crlf_pubkey_pem.as_bytes()).expect("write crlf pubkey");

    let clouds_bytes = br#"{"schema":"jope.carbon.clouds.v1","regions":[]}"#.to_vec();

    // Build all cross-OS archives containing the real compiled host `carbon` executable so both
    // `install.sh` (Linux/macOS) and `install.ps1` (Windows / pwsh) can install & smoke-run it!
    let win_x64_zip = build_stored_zip_single_file("carbon.exe", &bin_bytes);
    let win_arm_zip = build_stored_zip_single_file("carbon.exe", &bin_bytes);
    let mac_arm_tgz = build_single_file_targz("carbon", &bin_bytes);
    let mac_x64_tgz = build_single_file_targz("carbon", &bin_bytes);
    let lin_x64_musl_tgz = build_single_file_targz("carbon", &bin_bytes);
    let lin_x64_gnu_tgz = build_single_file_targz("carbon", &bin_bytes);
    let lin_arm_musl_tgz = build_single_file_targz("carbon", &bin_bytes);
    let lin_arm_gnu_tgz = build_single_file_targz("carbon", &bin_bytes);

    let assets: Vec<(&str, Vec<u8>)> = vec![
        ("carbon-clouds.json", clouds_bytes),
        ("carbon-windows-x86_64-msvc.zip", win_x64_zip),
        ("carbon-windows-aarch64-msvc.zip", win_arm_zip),
        ("carbon-darwin-arm64.tar.gz", mac_arm_tgz),
        ("carbon-darwin-x86_64.tar.gz", mac_x64_tgz),
        ("carbon-linux-x86_64-musl.tar.gz", lin_x64_musl_tgz),
        ("carbon-linux-x86_64-gnu.tar.gz", lin_x64_gnu_tgz),
        ("carbon-linux-aarch64-musl.tar.gz", lin_arm_musl_tgz),
        ("carbon-linux-aarch64-gnu.tar.gz", lin_arm_gnu_tgz),
    ];

    let mut sums_text = String::new();
    for (name, bytes) in &assets {
        let digest = jope_carbon::sha256_lower_hex(bytes);
        sums_text.push_str(&format!("{digest}  {name}\n"));
    }
    let rng = SystemRandom::new();
    let sig = kp
        .sign(&rng, sums_text.as_bytes())
        .expect("sign SHA256SUMS.txt");
    let sig_bytes = format!(
        "{}\n",
        base64::engine::general_purpose::STANDARD.encode(sig.as_ref())
    )
    .into_bytes();

    // Mirror server hosts the actual release binaries; Primary server returns `302 Found`
    // telemetry redirects to the Mirror server (matching `downloads.neverhuman.org` -> GitHub Releases!).
    let mut mirror_routes = BTreeMap::new();
    let tag = "carbon-cli-v1.1.0";
    for (name, bytes) in &assets {
        mirror_routes.insert(
            format!("/releases/{tag}/{name}"),
            (200, Vec::new(), bytes.clone()),
        );
    }
    mirror_routes.insert(
        format!("/releases/{tag}/SHA256SUMS.txt"),
        (200, Vec::new(), sums_text.as_bytes().to_vec()),
    );
    mirror_routes.insert(
        format!("/releases/{tag}/SHA256SUMS.txt.sig"),
        (200, Vec::new(), sig_bytes.clone()),
    );
    let (mirror_url, mirror_hits, mirror_handle) = spawn_static_http_server(mirror_routes, 64);

    let mut primary_routes = BTreeMap::new();
    for (name, _) in &assets {
        primary_routes.insert(
            format!("/carbon/releases/{tag}/{name}"),
            (
                302,
                vec![(
                    "Location".to_owned(),
                    format!("{mirror_url}/releases/{tag}/{name}"),
                )],
                Vec::new(),
            ),
        );
    }
    for meta in ["SHA256SUMS.txt", "SHA256SUMS.txt.sig"] {
        primary_routes.insert(
            format!("/carbon/releases/{tag}/{meta}"),
            (
                302,
                vec![(
                    "Location".to_owned(),
                    format!("{mirror_url}/releases/{tag}/{meta}"),
                )],
                Vec::new(),
            ),
        );
    }
    let (primary_url, primary_hits, primary_handle) = spawn_static_http_server(primary_routes, 64);

    #[cfg(unix)]
    {
        let install_sh = root.join("deploy/carbon/install.sh");
        let home_dir = tmp.path().join("posix_home");
        let bin_dir = home_dir.join(".local/bin");
        let cfg_dir = home_dir.join(".config/jopedime");
        fs::create_dir_all(&home_dir).expect("mkdir posix_home");

        let ok = Command::new("sh")
            .arg(&install_sh)
            .args([
                "--version",
                tag,
                "--base-url",
                &format!("{primary_url}/carbon/releases/{tag}"),
                "--bin-dir",
                bin_dir.to_str().expect("utf8"),
                "--config-dir",
                cfg_dir.to_str().expect("utf8"),
                "--allow-test-pubkey",
                pubkey_path.to_str().expect("utf8"),
                "--modify-path",
                "--skip-ping",
            ])
            .env("HOME", &home_dir)
            .env("SHELL", "/bin/zsh")
            .env("CARBON_INSTALL_STAGE_ROOT", tmp.path())
            .output()
            .expect("run install.sh");
        assert!(
            ok.status.success(),
            "install.sh failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&ok.stdout),
            String::from_utf8_lossy(&ok.stderr)
        );
        assert!(
            bin_dir.join("carbon").is_file(),
            "installed carbon binary missing"
        );
        assert!(cfg_dir.join("carbon-clouds.json").is_file());
        assert!(
            home_dir.join(".zshrc").is_file(),
            "--modify-path with SHELL=/bin/zsh must create .zshrc entry"
        );
        assert!(primary_hits.load(Ordering::SeqCst) >= 3);
        assert!(mirror_hits.load(Ordering::SeqCst) >= 3);

        // Tamper rejection: corrupt signature in a local source dir must fail closed!
        let tamper_dir = tmp.path().join("tamper_src");
        fs::create_dir_all(&tamper_dir).expect("mkdir tamper");
        for (name, bytes) in &assets {
            fs::write(tamper_dir.join(name), bytes).expect("write asset");
        }
        fs::write(tamper_dir.join("SHA256SUMS.txt"), sums_text.as_bytes()).expect("write sums");
        let mut bad_sig_der = sig.as_ref().to_vec();
        let last = bad_sig_der.len() - 1;
        bad_sig_der[last] ^= 0xFF;
        let bad_sig = format!(
            "{}\n",
            base64::engine::general_purpose::STANDARD.encode(&bad_sig_der)
        )
        .into_bytes();
        fs::write(tamper_dir.join("SHA256SUMS.txt.sig"), &bad_sig).expect("write bad sig");

        let bad_run = Command::new("sh")
            .arg(&install_sh)
            .args([
                "--source-dir",
                tamper_dir.to_str().expect("utf8"),
                "--bin-dir",
                tmp.path().join("bad_bin").to_str().expect("utf8"),
                "--config-dir",
                tmp.path().join("bad_cfg").to_str().expect("utf8"),
                "--allow-test-pubkey",
                pubkey_path.to_str().expect("utf8"),
                "--skip-ping",
            ])
            .env("HOME", &home_dir)
            .env("CARBON_INSTALL_STAGE_ROOT", tmp.path())
            .output()
            .expect("run tampered install.sh");
        assert!(
            !bad_run.status.success(),
            "tampered signature must fail install.sh"
        );

        // Uninstall removes binary and release manifests cleanly
        let uninst = Command::new("sh")
            .arg(&install_sh)
            .args([
                "--bin-dir",
                bin_dir.to_str().expect("utf8"),
                "--config-dir",
                cfg_dir.to_str().expect("utf8"),
                "--uninstall",
            ])
            .env("HOME", &home_dir)
            .output()
            .expect("run install.sh --uninstall");
        assert!(uninst.status.success());
        assert!(!bin_dir.join("carbon").exists());
    }

    // Exercise `install.ps1` on Windows (`powershell.exe` + `pwsh`) AND on Linux/macOS whenever `pwsh` is installed!
    let ps_shells: Vec<&str> = if cfg!(windows) {
        vec!["powershell.exe", "pwsh"]
    } else {
        vec!["pwsh"]
    };
    let install_ps1 = root.join("deploy/carbon/install.ps1");
    for shell in ps_shells {
        let probe = Command::new(shell)
            .args(["-NoProfile", "-Command", "$PSVersionTable.PSVersion.Major"])
            .output();
        let Ok(probe_out) = probe else {
            if cfg!(windows) && shell == "powershell.exe" {
                panic!("powershell.exe must be available on Windows runner");
            }
            continue;
        };
        if !probe_out.status.success() {
            continue;
        }

        // Only run full binary execution on Windows where `carbon.exe` is a real Win32 PE binary;
        // on Unix `pwsh`, still unit-test `Convert-Asn1DerEcdsaToP1363` on the exact `$rLen == 1`
        // DER edge case + `-Uninstall`!
        if cfg!(windows) {
            let shell_slug = shell.trim_end_matches(".exe");
            let ps_bin_dir = tmp.path().join(format!("ps_bin_{shell_slug}"));
            let ps_cfg_dir = tmp.path().join(format!("ps_cfg_{shell_slug}"));
            let ok = Command::new(shell)
                .args([
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    install_ps1.to_str().expect("utf8"),
                    "-Version",
                    tag,
                    "-BaseUrl",
                    &format!("{primary_url}/carbon/releases/{tag}"),
                    "-BinDir",
                    ps_bin_dir.to_str().expect("utf8"),
                    "-ConfigDir",
                    ps_cfg_dir.to_str().expect("utf8"),
                    "-AllowTestPubkey",
                    pubkey_path.to_str().expect("utf8"),
                    "-NoModifyPath",
                    "-SkipPing",
                ])
                .env("CARBON_INSTALL_STAGE_ROOT", tmp.path())
                .output()
                .expect("run install.ps1");
            assert!(
                ok.status.success(),
                "install.ps1 ({shell}) failed:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&ok.stdout),
                String::from_utf8_lossy(&ok.stderr)
            );
            assert!(ps_bin_dir.join("carbon.exe").is_file());
            assert!(ps_cfg_dir.join("carbon-clouds.json").is_file());
            assert!(primary_hits.load(Ordering::SeqCst) >= 3);
            assert!(mirror_hits.load(Ordering::SeqCst) >= 3);

            // Tampered signature must fail closed on Windows PowerShell!
            let tamper_ps = tmp.path().join(format!("ps_tamper_{shell_slug}"));
            fs::create_dir_all(&tamper_ps).expect("mkdir tamper_ps");
            for (name, bytes) in &assets {
                fs::write(tamper_ps.join(name), bytes).expect("write asset");
            }
            fs::write(tamper_ps.join("SHA256SUMS.txt"), sums_text.as_bytes()).expect("sums");
            let mut bad_sig_der = sig.as_ref().to_vec();
            let last = bad_sig_der.len() - 1;
            bad_sig_der[last] ^= 0xFF;
            let bad_sig = format!(
                "{}\n",
                base64::engine::general_purpose::STANDARD.encode(&bad_sig_der)
            )
            .into_bytes();
            fs::write(tamper_ps.join("SHA256SUMS.txt.sig"), &bad_sig).expect("bad sig");

            let bad_ps = Command::new(shell)
                .args([
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    install_ps1.to_str().expect("utf8"),
                    "-SourceDir",
                    tamper_ps.to_str().expect("utf8"),
                    "-BinDir",
                    tmp.path()
                        .join(format!("ps_bad_bin_{shell_slug}"))
                        .to_str()
                        .expect("utf8"),
                    "-ConfigDir",
                    tmp.path()
                        .join(format!("ps_bad_cfg_{shell_slug}"))
                        .to_str()
                        .expect("utf8"),
                    "-AllowTestPubkey",
                    pubkey_path.to_str().expect("utf8"),
                    "-NoModifyPath",
                    "-SkipPing",
                ])
                .env("CARBON_INSTALL_STAGE_ROOT", tmp.path())
                .output()
                .expect("run tampered install.ps1");
            assert!(
                !bad_ps.status.success(),
                "tampered sig must fail install.ps1 ({shell})"
            );

            // `-Uninstall` removes `carbon.exe` and manifests cleanly
            let uninst_ps = Command::new(shell)
                .args([
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    install_ps1.to_str().expect("utf8"),
                    "-BinDir",
                    ps_bin_dir.to_str().expect("utf8"),
                    "-ConfigDir",
                    ps_cfg_dir.to_str().expect("utf8"),
                    "-NoModifyPath",
                    "-Uninstall",
                ])
                .output()
                .expect("run install.ps1 -Uninstall");
            assert!(uninst_ps.status.success());
            assert!(!ps_bin_dir.join("carbon.exe").exists());
        }

        // Verify `Convert-Asn1DerEcdsaToP1363` from `install.ps1` directly on synthetic `$rLen == 1`
        // and `$sLen == 33` (leading `0x00` pad) DER integer edge cases!
        let der_script = format!(
            r#"$src = Get-Content -LiteralPath '{}' -Raw
$m = [regex]::Match($src, '(?s)function Convert-Asn1DerEcdsaToP1363\b.*?\r?\n\}}')
if (-not $m.Success) {{ throw 'Convert-Asn1DerEcdsaToP1363 not found' }}
Invoke-Expression $m.Value
$sBytes = New-Object byte[] 33
$sBytes[0] = 0x00
$sBytes[1] = 0x80
$sBytes[32] = 0x7F
$der = New-Object System.Collections.Generic.List[byte]
$der.Add(0x30)
$der.Add(38)
$der.Add(0x02); $der.Add(1); $der.Add(0x05)
$der.Add(0x02); $der.Add(33); $der.AddRange([byte[]]$sBytes)
$p1363 = Convert-Asn1DerEcdsaToP1363 ([byte[]]$der.ToArray())
if ($p1363.GetType().Name -ne 'Byte[]') {{ throw ('expected Byte[], got ' + $p1363.GetType().Name) }}
if ($p1363.Length -ne 64) {{ throw ('expected 64 bytes, got ' + $p1363.Length) }}
if ($p1363[31] -ne 5 -or $p1363[32] -ne 0x80 -or $p1363[63] -ne 0x7F) {{ throw 'unexpected P1363 byte layout' }}
Write-Output 'P1363_OK'"#,
            install_ps1.display().to_string().replace('\'', "''")
        );
        let der_check = Command::new(shell)
            .args(["-NoProfile", "-Command", &der_script])
            .output()
            .expect("run DER P1363 check");
        assert!(
            der_check.status.success()
                && String::from_utf8_lossy(&der_check.stdout).contains("P1363_OK"),
            "DER P1363 edge-case check failed in {shell}:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&der_check.stdout),
            String::from_utf8_lossy(&der_check.stderr)
        );
    }

    stop_http_server(&primary_url, primary_handle);
    stop_http_server(&mirror_url, mirror_handle);
}

#[cfg(all(feature = "cli", feature = "engine", feature = "emu"))]
mod full_stack {
    use super::*;
    use jope_api::carbon::{CarbonRegionCatalog, CarbonTierWire};
    use jope_carbon::cells::engine::verify_cold_start_r882_digest_pin;
    use jope_carbon::storage::cas::CarbonFsCas;
    use jope_carbon::{
        execute_self_upgrade_with_options, parse_secret_bundle_json_for_role,
        persist_model_handle_with_options, synthesize_cell_template_for_release,
        verify_static_linux_elf, CarbonEdgeConfig, CarbonEdgeRouter, CarbonEmuConfig,
        CarbonGatewayRequest, CarbonLocalLambdaEmulator, CarbonPresigner, CarbonSavedModelHandle,
        CarbonTabularEncoder,
    };

    #[test]
    fn deploy_security_split_secrets_elf_guard_digest_pin_and_passive_download_telemetry() {
        // 1. Split `edge-secrets` vs `cell-secrets` CloudFormation synthesis (`DEP-SEC-01`)
        let cf_raw =
            synthesize_cell_template_for_release("01234567abcdef89").expect("synthesize CFN");
        let cf: serde_json::Value = serde_json::from_str(&cf_raw).expect("parse CFN");
        assert!(
            cf["Parameters"]["EdgeSecretsArn"].is_object(),
            "EdgeSecretsArn parameter must exist"
        );
        assert!(
            cf["Parameters"]["CorsAllowedOrigins"].is_object(),
            "CorsAllowedOrigins parameter must exist"
        );
        assert!(
            cf["Conditions"]["HasDedicatedEdgeSecrets"].is_object(),
            "HasDedicatedEdgeSecrets condition must exist"
        );
        let resources = &cf["Resources"];
        let edge_role_str = serde_json::to_string(&resources["EdgeRole"]).expect("EdgeRole json");
        assert!(
            edge_role_str.contains("HasDedicatedEdgeSecrets")
                && edge_role_str.contains("EdgeSecretsArn"),
            "EdgeRole must gate GetSecretValue on EdgeSecretsArn: {edge_role_str}"
        );
        assert!(
            resources["PassiveDownloadsInsightsQuery"].is_object(),
            "PassiveDownloadsInsightsQuery CloudWatch Logs Insights resource must exist"
        );
        let engine_env = &resources["EngineFunction"]["Properties"]["Environment"]["Variables"]
            ["CARBON_R882_DIGEST"];
        assert_eq!(
            engine_env,
            &serde_json::json!({"Ref": "R882BundleDigest"}),
            "CARBON_R882_DIGEST must be pinned in CFN"
        );

        // 2. Defense-in-depth role scrub (`parse_secret_bundle_json_for_role(..., true)` for `edge`)
        let raw_secret_doc = serde_json::json!({
            "seal_secret_hex": "11".repeat(32),
            "presign_secret_hex": "22".repeat(32),
            "auth_secret_hex": "33".repeat(32),
            "engine_kek_hex": "44".repeat(32),
            "engine_kek_pubkey_b64": "BPubKeyBase64",
        })
        .to_string();
        let edge_scrubbed = parse_secret_bundle_json_for_role(&raw_secret_doc, true)
            .expect("parse edge-scrubbed secret bundle");
        assert!(edge_scrubbed.engine_kek_hex.is_none());
        assert!(edge_scrubbed.seal_secret_hex.is_none());
        assert!(edge_scrubbed.auth_secret_hex.is_some());
        assert!(edge_scrubbed.presign_secret_hex.is_some());

        let engine_full = parse_secret_bundle_json_for_role(&raw_secret_doc, false)
            .expect("parse full engine secret bundle");
        assert!(engine_full.engine_kek_hex.is_some());
        assert!(engine_full.seal_secret_hex.is_some());

        // 3. Static Linux ELF `PT_INTERP` rejection & cold-start `CARBON_R882_DIGEST` pin (`DEP-SEC-03`)
        let mut dyn_elf = vec![0u8; 128];
        dyn_elf[0..4].copy_from_slice(b"\x7FELF");
        dyn_elf[4] = 2; // 64-bit
        dyn_elf[5] = 1; // little-endian
        dyn_elf[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
        dyn_elf[32..40].copy_from_slice(&64u64.to_le_bytes()); // e_phoff = 64
        dyn_elf[54..56].copy_from_slice(&56u16.to_le_bytes()); // e_phentsize = 56
        dyn_elf[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum = 1
        dyn_elf[64..68].copy_from_slice(&3u32.to_le_bytes()); // PT_INTERP = 3
        let elf_err = verify_static_linux_elf(&dyn_elf).expect_err("PT_INTERP must fail");
        assert!(elf_err.contains("PT_INTERP"), "got: {elf_err}");

        verify_cold_start_r882_digest_pin(Some(&"a".repeat(64)), Some(&"a".repeat(64)), true)
            .expect("matching digest pin passes");
        let pin_err =
            verify_cold_start_r882_digest_pin(Some(&"a".repeat(64)), Some(&"b".repeat(64)), true)
                .expect_err("mismatched digest pin fails");
        assert!(pin_err.contains("CARBON_R882_DIGEST"));

        // 4. Operator build self-upgrade guard (`--force` required when `engine` compiled in)
        let (_, pub_pem) = generate_test_spki_keypair();
        let up_err = execute_self_upgrade_with_options(
            Some("http://127.0.0.1:9"),
            Some("carbon-cli-v1.1.0"),
            &pub_pem,
            false,
            false,
        )
        .expect_err("operator build must refuse accidental thin-CLI self-upgrade without --force");
        assert!(up_err.contains("--force"), "got: {up_err}");

        // 5. Passive download redirect telemetry & security headers (`INS-06` + `DEP-SEC-04`)
        let presigner = CarbonPresigner::from_secret([0x33u8; 32]);
        let day1 = presigner.daily_ip_pseudonym_hex(1_763_164_800, "203.0.113.42");
        let day2 = presigner.daily_ip_pseudonym_hex(1_763_164_800 + 86_400, "203.0.113.42");
        assert_ne!(
            day1, day2,
            "daily IP HMAC must rotate across UTC calendar days"
        );
        assert_eq!(day1.len(), 16);

        let mut hdrs = BTreeMap::new();
        hdrs.insert("user-agent".to_owned(), "curl/8.7.1".to_owned());
        hdrs.insert("accept".to_owned(), "*/*".to_owned());
        hdrs.insert(
            "x-carbon-trusted-source-ip".to_owned(),
            "198.51.100.7".to_owned(),
        );

        let tmp = tempfile::tempdir().expect("tempdir");
        let cas = Arc::new(CarbonFsCas::open(tmp.path().join("edge_cas")).expect("open fs cas"));
        let router = CarbonEdgeRouter::new(
            CarbonEdgeConfig {
                region: "us-east-2".to_owned(),
                public_base_url: "https://us-east-2.carbon.neverhuman.org".to_owned(),
                enterprise_self_hosted: false,
                catalog: CarbonRegionCatalog {
                    catalog_version: 1,
                    regions: Vec::new(),
                },
                presigner,
                auth_verifier: None,
                engine_kek_pubkey_b64: Some("BPubKey".to_owned()),
            },
            cas,
            Arc::new(NoopInvoker),
        );

        let dl_resp = router.handle(&CarbonGatewayRequest {
            method: "GET".to_owned(),
            path: "/carbon/releases/carbon-cli-v1.1.0/carbon-linux-x86_64-musl.tar.zst".to_owned(),
            query: BTreeMap::new(),
            headers: hdrs,
            body_b64: None,
            raw_body: Vec::new(),
        });
        assert_eq!(dl_resp.status, 302);
        assert_eq!(
            dl_resp.headers.get("location").map(String::as_str),
            Some("https://github.com/neverhuman/JopeDime/releases/download/carbon-cli-v1.1.0/carbon-linux-x86_64-musl.tar.zst")
        );
        assert_eq!(
            dl_resp.headers.get("cache-control").map(String::as_str),
            Some("private, no-store")
        );
        assert!(dl_resp.headers.contains_key("strict-transport-security"));
        assert_eq!(
            dl_resp
                .headers
                .get("x-content-type-options")
                .map(String::as_str),
            Some("nosniff")
        );
        assert!(dl_resp.headers.contains_key("x-carbon-request-id"));

        // Path traversal / unallowlisted asset must return 400!
        for bad_path in [
            "/carbon/releases/carbon-cli-v1.1.0/../etc/passwd",
            "/carbon/releases/%2e%2e/SHA256SUMS.txt",
            "/carbon/releases/carbon-cli-v1.1.0/evil.exe",
        ] {
            let bad_resp = router.handle(&CarbonGatewayRequest {
                method: "GET".to_owned(),
                path: bad_path.to_owned(),
                query: BTreeMap::new(),
                headers: BTreeMap::new(),
                body_b64: None,
                raw_body: Vec::new(),
            });
            assert_eq!(bad_resp.status, 400, "expected 400 on {bad_path}");
        }

        // 6. `--save` strips `capability_token` by default & `--embed-token` preserves it when requested!
        let cfg_dir = tmp.path().join("jopedime_cfg");
        fs::create_dir_all(&cfg_dir).expect("mkdir cfg_dir");
        std::env::set_var("CARBON_CONFIG_DIR", &cfg_dir);

        let save_file = tmp.path().join("model.carbon.json");
        let handle = CarbonSavedModelHandle {
            schema: jope_carbon::CARBON_MODEL_HANDLE_SCHEMA.to_owned(),
            created_epoch_ms: 1_760_000_000_000,
            run_id: "run_test_strip_01".to_owned(),
            dataset_id: "ds_test_strip_01".to_owned(),
            tenant_id: "tenant_strip".to_owned(),
            tier: CarbonTierWire::Pro,
            cloud: None,
            endpoint: "http://127.0.0.1:9".to_owned(),
            region: "us-east-2".to_owned(),
            engine_kek_pubkey_b64: Some("BPub".to_owned()),
            encrypted_cuen: true,
            u_dek_b64: Some("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=".to_owned()),
            capability_token: Some("ctk1.secret_payload.secret_sig".to_owned()),
            encoder: CarbonTabularEncoder {
                raw_feature_columns: vec!["x1".to_owned()],
                dropped_columns: Vec::new(),
                transforms: Vec::new(),
                encoded_feature_names: vec!["x1".to_owned()],
                target_column: "y".to_owned(),
                delimiter: ',',
                had_header: true,
            },
            fit_rows: 100,
            holdout_rows: 25,
            holdout_rmse: Some(0.25),
            holdout_mae: Some(0.20),
            holdout_r2: Some(0.95),
            feature_importances: Vec::new(),
        };
        persist_model_handle_with_options(&handle, Some(&save_file), false)
            .expect("persist stripped");
        let saved_raw = fs::read_to_string(&save_file).expect("read saved handle");
        assert!(
            !saved_raw.contains("ctk1.secret_payload"),
            "--save must strip capability_token by default: {saved_raw}"
        );
        // Registry handle under CARBON_CONFIG_DIR/runs/ still preserves capability_token locally!
        let reg_raw = fs::read_to_string(cfg_dir.join("runs/run_test_strip_01.json"))
            .expect("read registry handle");
        assert!(reg_raw.contains("ctk1.secret_payload"));

        let save_embedded = tmp.path().join("model_embedded.carbon.json");
        persist_model_handle_with_options(&handle, Some(&save_embedded), true)
            .expect("persist embedded");
        assert!(fs::read_to_string(&save_embedded)
            .expect("read embedded")
            .contains("ctk1.secret_payload"));
        std::env::remove_var("CARBON_CONFIG_DIR");
    }

    struct NoopInvoker;
    impl jope_carbon::CarbonEngineInvoker for NoopInvoker {
        fn invoke_engine(
            &self,
            _command: &jope_carbon::CarbonEngineWireCommand,
        ) -> Result<Vec<u8>, String> {
            Err("noop".to_owned())
        }
        fn invoke_engine_async(
            &self,
            _command: &jope_carbon::CarbonEngineWireCommand,
        ) -> Result<(), String> {
            Err("noop".to_owned())
        }
    }

    #[test]
    fn e2e_custom_fitness_aux_column_explain_export_predict_stdout_and_shred() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
        let carbon_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));

        let mut emu_cfg = CarbonEmuConfig::for_dir(tmp.path().join("cell_e2e"));
        emu_cfg.region = "us-east-2".to_owned();
        emu_cfg.force_hybrid_fanout = false;
        emu_cfg.eval_worker_threads = 2;
        emu_cfg.edge_binary_path = Some(edge_bin);
        let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start emu");
        let pro_token = emu
            .mint_capability_token("tenant_fit_e2e", CarbonTierWire::Pro, false)
            .expect("mint pro token");

        let home_dir = tmp.path().join("cli_home");
        fs::create_dir_all(&home_dir).expect("mkdir home");

        // Training CSV includes `w` (aux weight column, excluded from X!)
        let train_csv = tmp.path().join("train_weighted.csv");
        let mut csv_body = String::from("x1,x2,w,y\n");
        for i in 0..64 {
            let x1 = (i as f64) * 0.2 - 5.0;
            let x2 = ((i % 9) as f64) * 0.5 - 2.0;
            let w = if i % 4 == 0 { 3.0 } else { 1.0 };
            let y = 2.5 * x1 - 1.2 * x2 + 0.75;
            csv_body.push_str(&format!("{x1:.4},{x2:.4},{w:.1},{y:.6}\n"));
        }
        fs::write(&train_csv, &csv_body).expect("write train csv");

        let spec_path = tmp.path().join("weighted_obj.json");
        fs::write(
            &spec_path,
            r#"{"metric":"weighted","weights":{"op":"column","name":"w"},"fitness":{"metric":"rmse"}}"#,
        )
        .expect("write objective spec");

        let handle_path = tmp.path().join("champion.carbon.json");

        // 1. `carbon fit` with `--fitness @weighted_obj.json`, `--split forward`, `--calibration affine`, `--save` (strips token!)
        let fit_out = Command::new(&carbon_bin)
            .args([
                "fit",
                train_csv.to_str().expect("utf8"),
                "--target",
                "y",
                "--effort",
                "fast",
                "--fitness",
                &format!("@{}", spec_path.display()),
                "--split",
                "forward",
                "--calibration",
                "affine",
                "--endpoint",
                &emu.base_url(),
                "--token",
                &pro_token,
                "--save",
                handle_path.to_str().expect("utf8"),
                "--json",
            ])
            .env("HOME", &home_dir)
            .env("USERPROFILE", &home_dir)
            .env("CARBON_CONFIG_DIR", home_dir.join(".config/jopedime"))
            .output()
            .expect("run carbon fit");
        assert!(
            fit_out.status.success(),
            "carbon fit failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&fit_out.stdout),
            String::from_utf8_lossy(&fit_out.stderr)
        );
        let fit_json: serde_json::Value =
            serde_json::from_slice(&fit_out.stdout).expect("parse fit receipt");
        assert!(
            fit_json["outer_holdout_score"]
                .as_f64()
                .is_some_and(f64::is_finite),
            "fit_json missing finite outer_holdout_score: {fit_json}"
        );

        // Saved handle has empty capability_token, so store token via `carbon login` to verify registry fallback!
        let saved_handle_str = fs::read_to_string(&handle_path).expect("read handle");
        assert!(
            !saved_handle_str.contains(&pro_token),
            "token must be stripped from --save file"
        );

        let login_out = Command::new(&carbon_bin)
            .args([
                "login",
                "--endpoint",
                &emu.base_url(),
                "--token",
                &pro_token,
            ])
            .env("HOME", &home_dir)
            .env("USERPROFILE", &home_dir)
            .env("CARBON_CONFIG_DIR", home_dir.join(".config/jopedime"))
            .output()
            .expect("carbon login");
        assert!(
            login_out.status.success(),
            "carbon login failed: {}",
            String::from_utf8_lossy(&login_out.stderr)
        );

        // 2. `carbon explain` decrypts exact holdout permutation importances via `CUEN\x01`!
        let explain_out = Command::new(&carbon_bin)
            .args([
                "explain",
                "--model",
                handle_path.to_str().expect("utf8"),
                "--json",
            ])
            .env("HOME", &home_dir)
            .env("USERPROFILE", &home_dir)
            .env("CARBON_CONFIG_DIR", home_dir.join(".config/jopedime"))
            .output()
            .expect("carbon explain");
        assert!(
            explain_out.status.success(),
            "carbon explain failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&explain_out.stdout),
            String::from_utf8_lossy(&explain_out.stderr)
        );
        let explain_json: serde_json::Value =
            serde_json::from_slice(&explain_out.stdout).expect("parse explain json");
        let imps = explain_json["feature_importances"]
            .as_array()
            .expect("importances array");
        assert_eq!(
            imps.len(),
            2,
            "must report exact importances for x1 and x2 (not aux w)"
        );

        // 3. `carbon export --lang python` & `--lang rust` decrypt real standalone source (not placeholder!)
        let py_export = tmp.path().join("exported_model.py");
        let exp_py_out = Command::new(&carbon_bin)
            .args([
                "export",
                "--model",
                handle_path.to_str().expect("utf8"),
                "--lang",
                "python",
                "--out",
                py_export.to_str().expect("utf8"),
            ])
            .env("HOME", &home_dir)
            .env("USERPROFILE", &home_dir)
            .env("CARBON_CONFIG_DIR", home_dir.join(".config/jopedime"))
            .output()
            .expect("carbon export python");
        assert!(
            exp_py_out.status.success(),
            "carbon export python failed: {}",
            String::from_utf8_lossy(&exp_py_out.stderr)
        );
        let py_code = fs::read_to_string(&py_export).expect("read exported python");
        assert!(
            py_code.contains("def predict"),
            "exported Python must contain standalone def predict, got:\n{py_code}"
        );
        assert!(!py_code.contains("TODO") && !py_code.contains("placeholder"));

        // 4. `carbon predict --out -` on unlabeled CSV WITHOUT aux column `w` streams pure CSV to stdout!
        let test_unlabeled = tmp.path().join("unlabeled_no_w.csv");
        fs::write(&test_unlabeled, "x1,x2\n1.0,-0.5\n2.0,1.0\n-1.5,0.25\n")
            .expect("write unlabeled csv without w");
        let pred_out = Command::new(&carbon_bin)
            .args([
                "predict",
                test_unlabeled.to_str().expect("utf8"),
                "--model",
                handle_path.to_str().expect("utf8"),
                "--out",
                "-",
            ])
            .env("HOME", &home_dir)
            .env("USERPROFILE", &home_dir)
            .env("CARBON_CONFIG_DIR", home_dir.join(".config/jopedime"))
            .output()
            .expect("carbon predict --out -");
        assert!(
            pred_out.status.success(),
            "carbon predict --out - failed: {}",
            String::from_utf8_lossy(&pred_out.stderr)
        );
        let pred_stdout = String::from_utf8_lossy(&pred_out.stdout);
        let pred_lines: Vec<&str> = pred_stdout.lines().collect();
        assert_eq!(
            pred_lines.len(),
            4,
            "stdout must have exact header + 3 prediction rows with zero trailing receipt noise: {pred_stdout}"
        );
        assert!(pred_lines[0].contains("prediction"));

        // 5. `carbon shred` destroys server run state and local key material
        let shred_out = Command::new(&carbon_bin)
            .args([
                "shred",
                "--model",
                handle_path.to_str().expect("utf8"),
                "--json",
            ])
            .env("HOME", &home_dir)
            .env("USERPROFILE", &home_dir)
            .env("CARBON_CONFIG_DIR", home_dir.join(".config/jopedime"))
            .output()
            .expect("carbon shred");
        assert!(
            shred_out.status.success(),
            "carbon shred failed: {}",
            String::from_utf8_lossy(&shred_out.stderr)
        );

        emu.shutdown();
    }
}
