//! Section 14 coverage closure for `src/cli/spec_report.rs` (`run_ci_quickstart_qualification`,
//! `execute_carbon_spec_report` with `--code-coverage-export`, `execute_carbon_spec_verify` `Ok` &
//! tamper refusals, `CLI-43` `carbon spec-verify`), `src/cells/engine_invoker.rs`, and
//! `src/cells/eval.rs`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use axum::{
    body::Body,
    http::{Response, StatusCode},
    routing::post,
    Router,
};
use base64::Engine as _;

use jope_api::{carbon::CarbonExplainRequest, ids::RunId};
use jope_carbon::{
    cli::spec_report::{
        run_ci_quickstart_qualification, SpecEvidenceEntry, SpecPdfReceipt, SpecScreenshotEntry,
        REQUIRED_SPEC_SCREENSHOTS,
    },
    derive_cuen_slot_subkey, encrypt_cuen_payload, execute_carbon_spec_report,
    execute_carbon_spec_verify, generate_random_u_dek, open_engine_invoker, open_eval_invoker,
    sha256_lower_hex, verify_pdf_bytes, verify_png_bytes, CarbonCasStore, CarbonEngineInvoker,
    CarbonEngineWireCommand, CarbonEvalDecryptionGrant, CarbonEvalInvoker, CarbonEvalWireRequest,
    CarbonEvalWorker, CarbonFsCas, CarbonHttpEngineInvoker, CarbonHttpEvalInvoker,
    CarbonSpecReportOptions, CarbonSpecVerifyOptions, CarbonThinClient, CAS_BLOB_CUEN_SCOPE,
};
use jope_prep::cuen::CuenPayloadKind;
use jope_service::worker::{Attempt, Limits, Request as WorkerRequest};

use super::{spawn_loopback_axum, ScopedProcessEnv};

fn ieee_crc32_test(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg() & 0xEDB8_8320;
            crc = (crc >> 1) ^ mask;
        }
    }
    !crc
}

fn push_png_chunk(buf: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let crc_start = buf.len();
    buf.extend_from_slice(kind);
    buf.extend_from_slice(data);
    let crc = ieee_crc32_test(&buf[crc_start..]);
    buf.extend_from_slice(&crc.to_be_bytes());
}

fn build_valid_test_png(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(96);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8; // bit depth
    ihdr[9] = 6; // RGBA
    push_png_chunk(&mut out, b"IHDR", &ihdr);
    // Minimal valid zlib stream (`0x78 0x9C 0x03 0x00 0x00 0x00 0x00 0x01`).
    push_png_chunk(
        &mut out,
        b"IDAT",
        &[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01],
    );
    push_png_chunk(&mut out, b"IEND", &[]);
    out
}

fn build_valid_test_pdf(page_count: usize, total_min_bytes: usize) -> Vec<u8> {
    let mut s = String::with_capacity(total_min_bytes + 1024);
    s.push_str("%PDF-1.4\n1 0 obj << /Type /Pages /Count ");
    s.push_str(&page_count.to_string());
    s.push_str(" >> endobj\n");
    for i in 0..page_count {
        s.push_str(&format!(
            "{} 0 obj << /Type /Page /Parent 1 0 R >> endobj\n",
            i + 2
        ));
    }
    while s.len() + 16 < total_min_bytes {
        s.push_str("% carbon-spec-padding-line-0123456789abcdef0123456789abcdef\n");
    }
    s.push_str("%%EOF\n");
    s.into_bytes()
}

#[test]
fn spec_report_ci_quickstart_qualification_in_process() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let carbon_exe = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sha = "0ff23cc6ac4ac5ba783e67af27e80d0adc116038";
    let qual_scratch = tmp.path().join("qs_scratch");
    fs::create_dir_all(&qual_scratch).expect("mkdir qs_scratch");
    let ci_qual = run_ci_quickstart_qualification(&carbon_exe, &repo_root, &qual_scratch, sha)
        .expect("run_ci_quickstart_qualification");
    for s in &ci_qual.steps {
        eprintln!(
            "step {}: {:.1} ms (passed={})",
            s.test_id, s.wall_ms, s.passed
        );
    }
    assert!(ci_qual.all_passed && ci_qual.total_tests >= 8);
}

#[test]
fn spec_report_full_ci_qualification_and_spec_verify_bundle_roundtrip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let carbon_exe = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sha = "0ff23cc6ac4ac5ba783e67af27e80d0adc116038";

    // 1. Direct unit checks on `verify_png_bytes` and `verify_pdf_bytes` (valid + all error branches).
    let good_png = build_valid_test_png(1600, 320);
    assert_eq!(
        verify_png_bytes("good.png", &good_png).expect("valid png"),
        (1600, 320)
    );
    assert!(verify_png_bytes("short.png", b"short").is_err());
    let small_png = build_valid_test_png(800, 100);
    assert!(verify_png_bytes("small.png", &small_png).is_err());
    let mut bad_crc_png = good_png.clone();
    let last_idx = bad_crc_png.len() - 1;
    bad_crc_png[last_idx] ^= 0x55;
    assert!(verify_png_bytes("crc.png", &bad_crc_png).is_err());

    let good_pdf = build_valid_test_pdf(16, 84_000);
    assert_eq!(
        verify_pdf_bytes("good.pdf", &good_pdf).expect("valid pdf"),
        16
    );
    assert!(verify_pdf_bytes("tiny.pdf", b"%PDF-1.4\n%%EOF\n").is_err());
    assert!(verify_pdf_bytes("hdr.pdf", &vec![b'X'; 85_000]).is_err());
    let mut no_eof = good_pdf.clone();
    let n = no_eof.len();
    no_eof[n - 6..].copy_from_slice(b"XXXXXX");
    assert!(verify_pdf_bytes("noeof.pdf", &no_eof).is_err());
    let few_pages = build_valid_test_pdf(3, 84_000);
    assert!(verify_pdf_bytes("few.pdf", &few_pages).is_err());

    // 2. Synthesize verified `ci-qual-full.json` receipt for fast deterministic bundle generation & `spec-verify` roundtrip.
    let ci_qual_synth = serde_json::json!({
        "schema": "jopedime.carbon.ci-quickstart-qualification.v1",
        "timestamp_epoch_ms": 1_763_000_000_000u64,
        "source_sha": sha,
        "binary_version": jope_carbon::CARBON_PKG_VERSION,
        "release_tag": jope_carbon::CARBON_RELEASE_TAG,
        "emulator_regions": ["prod:us-east-2", "prod:eu-central-1", "dev:us-east-2"],
        "emulator_edge_invocations": 40,
        "emulator_engine_invocations": 12,
        "emulator_eval_invocations": 24,
        "residual_cas_blobs_after_shred": 0,
        "total_tests": 8,
        "passed_tests": 8,
        "all_passed": true,
        "total_wall_ms": 1200.0,
        "steps": [{
            "test_id": "CI-QS-01",
            "title": "Signed install.sh verification",
            "command": "sh deploy/carbon/install.sh",
            "exit_code": 0,
            "passed": true,
            "wall_ms": 150.0,
            "highlights": "ok",
            "transcript_excerpt": "$ carbon --version"
        }],
        "redteam_vectors": [{
            "vector_id": "V-01",
            "finding_ref": "RT-02",
            "title": "Unauthenticated upload prepare rejected",
            "passed": true,
            "observed_http_status": 401,
            "wall_ms": 5.0,
            "evidence": "HTTP 401"
        }]
    });
    let ci_qual_path = tmp.path().join("ci-qual-full.json");
    fs::write(
        &ci_qual_path,
        serde_json::to_vec_pretty(&ci_qual_synth).expect("ser ci_qual"),
    )
    .expect("write ci_qual");

    // 3. Build synthetic `llvm-cov` export JSON for all `crates/jope-carbon/src/**/*.rs` files and run `execute_carbon_spec_report`.
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut src_files = Vec::new();
    let mut stack = vec![src_root];
    while let Some(dir) = stack.pop() {
        for ent in fs::read_dir(&dir).expect("read_dir") {
            let p = ent.expect("ent").path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                src_files.push(p);
            }
        }
    }
    let llvm_files: Vec<serde_json::Value> = src_files
        .iter()
        .map(|p| {
            serde_json::json!({
                "filename": p.display().to_string(),
                "summary": {
                    "lines": { "count": 1000, "covered": 950 },
                    "regions": { "count": 1500, "covered": 1425 },
                    "functions": { "count": 100, "covered": 95 }
                },
                "segments": [
                    [1, 1, 9, true, true, false],
                    [10000, 1, 0, false, false, false]
                ]
            })
        })
        .collect();
    let llvm_export_path = tmp.path().join("llvm-cov-synth.json");
    fs::write(
        &llvm_export_path,
        serde_json::to_vec(&serde_json::json!({ "data": [{ "files": llvm_files }] })).expect("ser"),
    )
    .expect("write llvm_export_path");

    // Missing qualification option must refuse cleanly.
    assert!(execute_carbon_spec_report(&CarbonSpecReportOptions {
        evidence_dir: repo_root.join("docs/carbon/evidence"),
        out_dir: tmp.path().join("err_out"),
        run_ci_qualification: false,
        ci_qualification_in: None,
        live_ci_receipt_in: None,
        code_coverage_in: None,
        source_sha: Some(sha.to_owned()),
        repo_root: repo_root.clone(),
    })
    .is_err());

    let spec_dir = tmp.path().join("spec_full_bundle");
    let html_path = execute_carbon_spec_report(&CarbonSpecReportOptions {
        evidence_dir: repo_root.join("docs/carbon/evidence"),
        out_dir: spec_dir.clone(),
        run_ci_qualification: false,
        ci_qualification_in: Some(ci_qual_path),
        live_ci_receipt_in: None,
        code_coverage_in: Some(llvm_export_path),
        source_sha: Some(sha.to_owned()),
        repo_root: repo_root.clone(),
    })
    .expect("execute_carbon_spec_report with code coverage");
    assert!(html_path.is_file());

    // 4. Populate rendered PDF + 8 `2x` screenshots + `spec-pdf-receipt.json`, then verify BOTH in-process
    //    `execute_carbon_spec_verify` AND `carbon spec-verify` CLI subprocess (`CLI-43` `Ok` path!).
    let pdf_leaf = "carbon-engineering-testing-spec.pdf";
    fs::write(spec_dir.join(pdf_leaf), &good_pdf).expect("write pdf");
    let pdf_sha = sha256_lower_hex(&good_pdf);
    let html_sha = sha256_lower_hex(&fs::read(&html_path).expect("read html"));
    let ci_q_sha = sha256_lower_hex(
        &fs::read(spec_dir.join("ci-quickstart-qualification.json")).expect("read ci_q"),
    );
    let cov_sha =
        sha256_lower_hex(&fs::read(spec_dir.join("code-coverage-summary.json")).expect("read cov"));

    let mut shot_entries = Vec::with_capacity(REQUIRED_SPEC_SCREENSHOTS.len());
    for &(id, fname, cap) in &REQUIRED_SPEC_SCREENSHOTS {
        let p = spec_dir.join("screenshots").join(fname);
        fs::write(&p, &good_png).expect("write screenshot");
        shot_entries.push(SpecScreenshotEntry {
            id: id.to_owned(),
            file_name: fname.to_owned(),
            caption: cap.to_owned(),
            width: 1600,
            height: 320,
            bytes: good_png.len() as u64,
            sha256: sha256_lower_hex(&good_png),
        });
    }

    let receipt = SpecPdfReceipt {
        schema: "jopedime.carbon.spec-pdf-receipt.v1".to_owned(),
        rendered_epoch_ms: 1_763_000_000_000,
        source_sha: sha.to_owned(),
        binary_version: jope_carbon::CARBON_PKG_VERSION.to_owned(),
        release_tag: jope_carbon::CARBON_RELEASE_TAG.to_owned(),
        all_ci_tests_passed: true,
        all_prod_evidence_verified: true,
        pdf_file: pdf_leaf.to_owned(),
        pdf_bytes: good_pdf.len() as u64,
        pdf_sha256: pdf_sha,
        pdf_page_count: 16,
        html_sha256: html_sha,
        ci_qualification_sha256: ci_q_sha,
        live_ci_qualification_file: None,
        live_ci_qualification_sha256: None,
        code_coverage_file: Some("code-coverage-summary.json".to_owned()),
        code_coverage_sha256: Some(cov_sha),
        horizontal_overflow_px: 0,
        screenshots: shot_entries,
        evidence_ledger: Vec::<SpecEvidenceEntry>::new(),
    };
    let receipt_path = spec_dir.join("spec-pdf-receipt.json");
    fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("ser receipt"),
    )
    .expect("write spec-pdf-receipt.json");

    let verified = execute_carbon_spec_verify(&CarbonSpecVerifyOptions {
        dir: spec_dir.clone(),
        expected_source_sha: Some(sha.to_owned()),
    })
    .expect("in-process execute_carbon_spec_verify ok");
    assert_eq!(verified.pdf_page_count, 16);

    let cli_sv = Command::new(&carbon_exe)
        .args([
            "spec-verify",
            "--dir",
            spec_dir.to_str().expect("utf8"),
            "--expected-source-sha",
            sha,
        ])
        .output()
        .expect("spawn carbon spec-verify");
    assert!(
        cli_sv.status.success(),
        "carbon spec-verify failed: {}",
        String::from_utf8_lossy(&cli_sv.stderr)
    );
    assert!(String::from_utf8_lossy(&cli_sv.stdout).contains("\"verified\""));

    // Negative checks: wrong `expected_source_sha` & `horizontal_overflow_px > 1`.
    assert!(execute_carbon_spec_verify(&CarbonSpecVerifyOptions {
        dir: spec_dir.clone(),
        expected_source_sha: Some("1111111111111111111111111111111111111111".to_owned()),
    })
    .is_err());

    let mut overflow_rcpt = receipt;
    overflow_rcpt.horizontal_overflow_px = 5;
    fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&overflow_rcpt).expect("ser overflow"),
    )
    .expect("write overflow receipt");
    assert!(execute_carbon_spec_verify(&CarbonSpecVerifyOptions {
        dir: spec_dir,
        expected_source_sha: Some(sha.to_owned()),
    })
    .is_err());
}

#[test]
fn engine_invoker_and_eval_worker_all_error_and_cache_branches() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // 1. `CarbonHttpEngineInvoker` + `CarbonHttpEvalInvoker`: HTTP 502, bad JSON, and connection refused.
    let err_app = Router::new()
        .route(
            "/fail502",
            post(|| async {
                Response::builder()
                    .status(StatusCode::BAD_GATEWAY)
                    .body(Body::from("upstream error"))
                    .expect("502")
            }),
        )
        .route(
            "/badjson",
            post(|| async {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from("not-valid-json"))
                    .expect("200")
            }),
        );
    let (mock_addr, mock_tx, mock_jh) = spawn_loopback_axum("mock-invoker-err", err_app);

    let cmd = CarbonEngineWireCommand::Explain {
        run_id: "run_01JCCARB0NINVOKER000000001".to_owned(),
        explain: CarbonExplainRequest {
            tenant_id: "t1".to_owned(),
            top_k: Some(3),
            client_response_pubkey_b64: None,
            wrapped_u_dek_b64: None,
        },
    };
    let eng_502 = CarbonHttpEngineInvoker::new(format!("http://{mock_addr}/fail502"));
    assert!(eng_502.invoke_engine(&cmd).is_err());
    assert!(eng_502.invoke_engine_async(&cmd).is_err());

    let eng_dead = CarbonHttpEngineInvoker::new("http://127.0.0.1:9/invoke");
    assert!(eng_dead.invoke_engine(&cmd).is_err());
    assert!(eng_dead.invoke_engine_async(&cmd).is_err());

    // Build a valid `WorkerRequest` + FIT JMAT blob for `CarbonEvalWorker` & `CarbonHttpEvalInvoker`.
    let headers = vec!["x1".to_owned(), "x2".to_owned(), "y".to_owned()];
    let rows: Vec<Vec<f64>> = (0..24)
        .map(|i| {
            let x1 = i as f64;
            let x2 = (i % 4) as f64;
            vec![x1, x2, 2.0 * x1 + x2]
        })
        .collect();
    let prep = CarbonThinClient::prepare_numeric_dataset(headers, &rows, "y", 11, 0.25)
        .expect("prep numeric");
    let fit_pt_sha = sha256_lower_hex(&prep.artifacts.fit_jmat);

    let run_id = RunId::new();
    let worker_req = WorkerRequest {
        attempt: Attempt {
            run: run_id,
            round: 0,
            slot: 0,
            attempt: 1,
        },
        matrix_path: PathBuf::from("fit.jmat"),
        fold_spec: serde_json::json!({"kind": "k_fold", "folds": 3}),
        dsl: "x1".to_owned(),
        objective: serde_json::json!("rmse"),
        seed: 11,
        limits: Limits {
            wall_ms: 10_000,
            rss_bytes: 256 * 1_048_576,
        },
    };
    let eval_wire = CarbonEvalWireRequest {
        fit_jmat_sha256: fit_pt_sha.clone(),
        blob_scope: None,
        request: worker_req,
        fit_champion: false,
        decryption_grant: None,
    };

    let eval_502 = CarbonHttpEvalInvoker::new(format!("http://{mock_addr}/fail502"));
    assert!(eval_502.invoke_eval(&eval_wire).is_err());
    let eval_bad = CarbonHttpEvalInvoker::new(format!("http://{mock_addr}/badjson"));
    assert!(eval_bad.invoke_eval(&eval_wire).is_err());
    let eval_dead = CarbonHttpEvalInvoker::new("http://127.0.0.1:9/eval");
    assert!(eval_dead.invoke_eval(&eval_wire).is_err());

    let _ = mock_tx.send(());
    let _ = mock_jh.join();

    // 2. `open_engine_invoker` & `open_eval_invoker` `lambda://` dispatch + missing AWS creds fast refusal!
    {
        let _no_aws = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("AWS_SESSION_TOKEN", None),
            ("AWS_PROFILE", Some("nonexistent-carbon-unit-test-profile")),
            ("AWS_REGION", Some("us-west-2")),
        ]);
        let lam_eng = open_engine_invoker("lambda://carbon-prod-engine:live", "us-east-2");
        assert!(lam_eng.invoke_engine(&cmd).is_err());
        assert!(lam_eng.invoke_engine_async(&cmd).is_err());

        let lam_eval = open_eval_invoker("lambda://carbon-prod-eval:live", "us-east-2");
        assert!(lam_eval.invoke_eval(&eval_wire).is_err());
    }

    // 3. `CarbonEvalDecryptionGrant` Debug redaction + `CarbonEvalWorker` grant/cache error & eviction paths!
    let cas = Arc::new(CarbonFsCas::open(tmp.path().join("eval_cas")).expect("open cas"));
    let eval_worker = CarbonEvalWorker::new(cas.clone(), tmp.path().join("eval_scratch"))
        .expect("new CarbonEvalWorker");

    assert!(eval_worker.handle_wire_bytes(b"{not_json").is_err());

    let u_dek = generate_random_u_dek().expect("u_dek");
    let fit_subkey = derive_cuen_slot_subkey(
        &u_dek,
        CuenPayloadKind::FitJmat,
        "tenant_ev",
        CAS_BLOB_CUEN_SCOPE,
    )
    .expect("subkey");
    let subkey_b64 = base64::engine::general_purpose::STANDARD.encode(*fit_subkey);
    let enc_fit = encrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::FitJmat,
        "tenant_ev",
        CAS_BLOB_CUEN_SCOPE,
        &prep.artifacts.fit_jmat,
    )
    .expect("enc_fit");
    let enc_sha = sha256_lower_hex(&enc_fit);
    cas.put_scoped_blob_verified(None, &enc_sha, &enc_fit)
        .expect("put enc_fit");

    let valid_grant = CarbonEvalDecryptionGrant {
        tenant_id: "tenant_ev".to_owned(),
        fit_slot_subkey_b64: subkey_b64.clone(),
        fit_plaintext_sha256: fit_pt_sha.clone(),
        exp_epoch_secs: 4_102_444_800,
    };
    let grant_dbg = format!("{valid_grant:?}");
    assert!(grant_dbg.contains("[REDACTED]") && !grant_dbg.contains(&subkey_b64));

    // Expired grant (`exp_epoch_secs = 1`).
    let mut exp_wire = eval_wire.clone();
    exp_wire.fit_jmat_sha256 = enc_sha.clone();
    exp_wire.decryption_grant = Some({
        let mut g = valid_grant.clone();
        g.exp_epoch_secs = 1;
        g
    });
    assert!(eval_worker.evaluate_shard(&exp_wire).is_err());

    // Invalid base64 subkey.
    exp_wire.decryption_grant = Some({
        let mut g = valid_grant.clone();
        g.fit_slot_subkey_b64 = "%%%not_b64%%%".to_owned();
        g
    });
    assert!(eval_worker.evaluate_shard(&exp_wire).is_err());

    // Wrong subkey byte length (8 bytes instead of 32).
    exp_wire.decryption_grant = Some({
        let mut g = valid_grant.clone();
        g.fit_slot_subkey_b64 = base64::engine::general_purpose::STANDARD.encode([7u8; 8]);
        g
    });
    assert!(eval_worker.evaluate_shard(&exp_wire).is_err());

    // Decrypted plaintext SHA-256 mismatch before cache populate.
    exp_wire.decryption_grant = Some({
        let mut g = valid_grant.clone();
        g.fit_plaintext_sha256 = "0".repeat(64);
        g
    });
    assert!(eval_worker.evaluate_shard(&exp_wire).is_err());

    // Populate valid cache entry #1, then trigger cache-hit plaintext SHA-256 mismatch!
    exp_wire.decryption_grant = Some(valid_grant.clone());
    let _ = eval_worker.evaluate_shard(&exp_wire);
    exp_wire.decryption_grant = Some({
        let mut g = valid_grant;
        g.fit_plaintext_sha256 = "f".repeat(64);
        g
    });
    let hit_mismatch = eval_worker
        .evaluate_shard(&exp_wire)
        .expect_err("cache hit plaintext sha mismatch");
    assert!(hit_mismatch.contains("mismatch"));

    // Populate plaintext cache entries #2 and #3 (`blob_scope` distinct) to trigger `>= EVAL_CACHE_MAX_ENTRIES` eviction!
    cas.put_scoped_blob_verified(
        Some("tenant_ev/upl_a"),
        &fit_pt_sha,
        &prep.artifacts.fit_jmat,
    )
    .expect("put scope_a");
    cas.put_scoped_blob_verified(
        Some("tenant_ev/upl_b"),
        &fit_pt_sha,
        &prep.artifacts.fit_jmat,
    )
    .expect("put scope_b");
    let mut w_a = eval_wire.clone();
    w_a.blob_scope = Some("tenant_ev/upl_a".to_owned());
    let _ = eval_worker.invoke_eval(&w_a);
    let mut w_b = eval_wire;
    w_b.blob_scope = Some("tenant_ev/upl_b".to_owned());
    let _ = eval_worker.invoke_eval(&w_b);
}
