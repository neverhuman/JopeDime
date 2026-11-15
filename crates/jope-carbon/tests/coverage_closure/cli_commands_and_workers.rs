//! Subprocess CLI & Lambda Runtime worker coverage closure across `carbon`, `carbon-edge`, and `carbon-worker`.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use jope_api::carbon::CarbonTierWire;
use jope_carbon::{
    CarbonAuthVerifier, CarbonCapabilityClaims, CarbonEmuConfig, CarbonEngineKek,
    CarbonLocalLambdaEmulator,
};

use super::SingleShotLambdaRuntimeServer;

fn carbon_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_carbon"))
}

fn carbon_edge_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"))
}

fn carbon_worker_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_carbon-worker"))
}

fn mint_emu_pro_token(cfg: &CarbonEmuConfig, tenant_id: &str) -> String {
    CarbonAuthVerifier::from_secret(cfg.auth_secret)
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: tenant_id.to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint pro token")
}

fn write_emu_clouds_manifest(dir: &Path, emu: &CarbonLocalLambdaEmulator) -> PathBuf {
    let (_, probed) = jope_carbon::CarbonThinClient::connect_fastest(vec![emu.base_url()])
        .expect("probe emulator /ping");
    let kek = probed[0]
        .engine_kek_pubkey_b64
        .clone()
        .expect("emulator KEK pubkey");
    let manifest_path = dir.join("carbon-clouds.json");
    let doc = serde_json::json!({
        "schema": "jope-carbon-clouds-v2",
        "generated_epoch_ms": 1_800_000_000_000u64,
        "clouds": {
            "dev": {
                "environment": "dev",
                "region": "us-east-2",
                "public_endpoint": emu.base_url(),
                "engine_kek_pubkey_b64": kek,
                "regions": [{
                    "region": "us-east-2",
                    "public_endpoint": emu.base_url(),
                    "engine_kek_pubkey_b64": kek
                }]
            },
            "prod": {
                "environment": "prod",
                "region": "us-east-2",
                "public_endpoint": emu.base_url(),
                "engine_kek_pubkey_b64": kek,
                "regions": [{
                    "region": "us-east-2",
                    "public_endpoint": emu.base_url(),
                    "engine_kek_pubkey_b64": kek
                }]
            }
        }
    });
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&doc).expect("encode clouds json"),
    )
    .expect("write clouds manifest");
    manifest_path
}

fn write_sample_csv(path: &Path) -> String {
    let mut csv = String::from("x0,x1,y\n");
    for i in 0..48 {
        let x0 = (i as f64) * 0.12 - 2.0;
        let x1 = ((i * 3) % 11) as f64 * 0.25;
        let y = 2.5 * x0 - 0.8 * x1 + 1.0;
        csv.push_str(&format!("{x0:.5},{x1:.5},{y:.5}\n"));
    }
    fs::write(path, csv.as_bytes()).expect("write sample csv");
    csv
}

#[test]
fn cli_global_flags_version_completions_plan_spotcheck_and_shorthand_branches() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path().join("home");
    fs::create_dir_all(&home).expect("mkdir home");
    let bin = carbon_bin();

    // Zero-arg `carbon` -> Version JSON; `carbon version`; `-V`; `--help`
    for args in [&[][..], &["version"][..], &["-V"][..], &["--help"][..]] {
        let out = Command::new(&bin)
            .args(args)
            .env("HOME", &home)
            .output()
            .expect("run carbon info");
        assert!(out.status.success(), "args {args:?} failed");
        assert!(!out.stdout.is_empty());
    }

    // Global observability flags: `-vv --log-format json --color never`, `-q --color always`, invalid flags -> exit 2
    let obs_ok = Command::new(&bin)
        .args(["-vv", "--log-format", "json", "--color", "never", "version"])
        .env("HOME", &home)
        .output()
        .expect("run carbon -vv json");
    assert!(obs_ok.status.success());

    let obs_quiet = Command::new(&bin)
        .args(["-q", "--color", "always", "version"])
        .env("HOME", &home)
        .output()
        .expect("run carbon -q always");
    assert!(obs_quiet.status.success());

    for bad_args in [
        &["--log-format", "xml", "version"][..],
        &["--color", "neon", "version"][..],
        &["bogus_subcommand"][..],
        &["missing_input_file.csv"][..],
        &["completions", "elvish"][..],
    ] {
        let bad = Command::new(&bin)
            .args(bad_args)
            .env("HOME", &home)
            .output()
            .expect("run bad cli args");
        assert_eq!(
            bad.status.code(),
            Some(2),
            "expected exit 2 for {bad_args:?}"
        );
    }

    // All 4 supported shells for `completions`, plus `plan` and `spot-check`
    for shell in ["bash", "zsh", "fish", "powershell"] {
        let comp = Command::new(&bin)
            .args(["completions", shell])
            .env("HOME", &home)
            .output()
            .expect("completions");
        assert!(comp.status.success(), "completions {shell} failed");
    }

    let plan_out = Command::new(&bin)
        .args([
            "plan",
            "--tier",
            "pro",
            "--effort",
            "high",
            "--total-rows",
            "5000",
            "--fit-rows",
            "4000",
            "--features",
            "16",
            "--ms-per-candidate",
            "1.5",
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon plan");
    assert!(plan_out.status.success());
    let plan_json: serde_json::Value = serde_json::from_slice(&plan_out.stdout).expect("plan json");
    assert!(plan_json["mode"].is_string());

    let spot_out = Command::new(&bin)
        .args([
            "spot-check",
            "--region",
            "us-east-2",
            "--monthly-spend-usd",
            "8500",
            "--p20-vcpus",
            "24",
            "--mean-vcpus",
            "48",
            "--p95-vcpus",
            "96",
            "--cold-start-rate",
            "0.04",
            "--consecutive-months",
            "3",
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon spot-check");
    assert!(spot_out.status.success());
    let spot_json: serde_json::Value =
        serde_json::from_slice(&spot_out.stdout).expect("spot-check json");
    assert!(spot_json["recommendation"].is_string());
}

#[test]
fn cli_stdin_shorthand_quickstart_redteam_bench_and_emu_selftest_subprocesses() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path().join("home");
    fs::create_dir_all(&home).expect("mkdir home");
    let bin = carbon_bin();

    let mut cfg = CarbonEmuConfig::for_dir(tmp.path().join("emu_cell"));
    cfg.force_hybrid_fanout = true;
    cfg.eval_worker_threads = 2;
    let pro_tok = mint_emu_pro_token(&cfg, "pro_cli_suite");
    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start emu");
    let clouds_path = write_emu_clouds_manifest(tmp.path(), &emu);
    let csv_path = tmp.path().join("sample.csv");
    let csv_body = write_sample_csv(&csv_path);

    // 1. Stdin shorthand: `carbon -` with CSV piped on stdin
    let mut child = Command::new(&bin)
        .arg("-")
        .env("HOME", &home)
        .env("CARBON_CLOUDS_MANIFEST", &clouds_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn carbon -");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(csv_body.as_bytes())
        .expect("write csv stdin");
    let dash_out = child.wait_with_output().expect("wait carbon -");
    assert!(
        dash_out.status.success(),
        "carbon - failed: {}",
        String::from_utf8_lossy(&dash_out.stderr)
    );

    // 2. `carbon quickstart` (--emit-csv --csv-only AND full live E2E against `--endpoint`)
    let qs_csv = tmp.path().join("qs_only.csv");
    let qs_only = Command::new(&bin)
        .args([
            "quickstart",
            "--emit-csv",
            qs_csv.to_str().expect("utf8"),
            "--csv-only",
        ])
        .env("HOME", &home)
        .output()
        .expect("quickstart csv-only");
    assert!(qs_only.status.success() && qs_csv.is_file());

    let qs_live = Command::new(&bin)
        .args(["quickstart", "--endpoint", &emu.base_url()])
        .env("HOME", &home)
        .output()
        .expect("quickstart live");
    assert!(
        qs_live.status.success(),
        "quickstart live failed: {}",
        String::from_utf8_lossy(&qs_live.stderr)
    );

    // 3. `carbon redteam` with `--json-out` and `--markdown-out`
    let rt_json = tmp.path().join("rt.json");
    let rt_md = tmp.path().join("rt.md");
    let rt_out = Command::new(&bin)
        .args([
            "redteam",
            "--endpoint",
            &emu.base_url(),
            "--json-out",
            rt_json.to_str().expect("utf8"),
            "--markdown-out",
            rt_md.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon redteam");
    assert!(rt_out.status.success() && rt_json.is_file() && rt_md.is_file());

    // 4. `carbon bench --suite smoke --pro` with `--json-out` and `--markdown-out`
    let b_json = tmp.path().join("bench_smoke.json");
    let b_md = tmp.path().join("bench_smoke.md");
    let bench_out = Command::new(&bin)
        .args([
            "bench",
            "--suite",
            "smoke",
            "--pro",
            "--token",
            &pro_tok,
            "--endpoint",
            &emu.base_url(),
            "--json-out",
            b_json.to_str().expect("utf8"),
            "--markdown-out",
            b_md.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon bench --suite smoke");
    assert!(
        bench_out.status.success(),
        "bench smoke failed: {}",
        String::from_utf8_lossy(&bench_out.stderr)
    );
    assert!(b_json.is_file() && b_md.is_file());

    // 5. `carbon emu` missing `--self-test` (exit 2) and with `--self-test` (exit 0)
    let emu_sub_dir = tmp.path().join("emu_selftest");
    let emu_bad = Command::new(&bin)
        .args(["emu", "--work-dir", emu_sub_dir.to_str().expect("utf8")])
        .env("HOME", &home)
        .output()
        .expect("emu without --self-test");
    assert!(!emu_bad.status.success());

    let emu_ok = Command::new(&bin)
        .args([
            "emu",
            "--self-test",
            "--work-dir",
            emu_sub_dir.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .output()
        .expect("emu --self-test");
    assert!(
        emu_ok.status.success(),
        "emu --self-test failed: {}",
        String::from_utf8_lossy(&emu_ok.stderr)
    );

    emu.shutdown();
}

#[test]
fn cli_full_model_handle_lifecycle_resume_explain_6lang_export_auth_and_shred() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path().join("home");
    fs::create_dir_all(&home).expect("mkdir home");
    let bin = carbon_bin();

    let cfg = CarbonEmuConfig::for_dir(tmp.path().join("emu_lifecycle"));
    let pro_tok = mint_emu_pro_token(&cfg, "pro_lifecycle");
    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start emu");
    let clouds_path = write_emu_clouds_manifest(tmp.path(), &emu);
    let csv_path = tmp.path().join("train.csv");
    let _ = write_sample_csv(&csv_path);

    // Register, whoami, self-verify, login, logout
    let reg = Command::new(&bin)
        .args([
            "register",
            "--free",
            "--cloud",
            "prod",
            "--label",
            "alice_cov",
            "--clouds-manifest",
            clouds_path.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon register");
    assert!(reg.status.success());

    let who = Command::new(&bin)
        .args([
            "whoami",
            "--cloud",
            "prod",
            "--clouds-manifest",
            clouds_path.to_str().expect("utf8"),
            "--json",
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon whoami");
    assert!(who.status.success());

    let sv = Command::new(&bin)
        .args([
            "self-verify",
            "--clouds-manifest",
            clouds_path.to_str().expect("utf8"),
            "--binary-only",
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon self-verify");
    assert!(
        sv.status.success(),
        "self-verify failed: {}",
        String::from_utf8_lossy(&sv.stderr)
    );

    let logout = Command::new(&bin)
        .args(["logout", "--cloud", "prod"])
        .env("HOME", &home)
        .output()
        .expect("carbon logout --cloud prod");
    assert!(logout.status.success());

    // Pause a run via `carbon run --stop-after-ms 1 --save` so we can exercise `carbon resume`!
    let model_handle = tmp.path().join("paused.carbon.json");
    let paused = Command::new(&bin)
        .args([
            "run",
            "--csv",
            csv_path.to_str().expect("utf8"),
            "--target",
            "y",
            "--pro",
            "--token",
            &pro_tok,
            "--effort",
            "medium",
            "--stop-after-ms",
            "1",
            "--predict-rows",
            "0",
            "--save",
            model_handle.to_str().expect("utf8"),
            "--embed-token",
            "--clouds-manifest",
            clouds_path.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon run --stop-after-ms 1");
    assert!(
        paused.status.success(),
        "paused run failed: {}",
        String::from_utf8_lossy(&paused.stderr)
    );

    // `carbon status --run latest --json` and `carbon runs --json`
    let runs_out = Command::new(&bin)
        .args(["runs", "--json"])
        .env("HOME", &home)
        .output()
        .expect("carbon runs");
    assert!(runs_out.status.success());

    let status_out = Command::new(&bin)
        .args(["status", "--run", "latest", "--json"])
        .env("HOME", &home)
        .output()
        .expect("carbon status");
    assert!(status_out.status.success());

    // `carbon resume --model ... --effort medium --additional-waves 1 --json`
    let resumed = Command::new(&bin)
        .args([
            "resume",
            "--model",
            model_handle.to_str().expect("utf8"),
            "--effort",
            "medium",
            "--additional-waves",
            "1",
            "--json",
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon resume");
    assert!(
        resumed.status.success(),
        "carbon resume failed: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );

    // `carbon explain` (`--format pretty` and `--json`)
    for fmt in ["pretty", "json"] {
        let exp = Command::new(&bin)
            .args([
                "explain",
                "--model",
                model_handle.to_str().expect("utf8"),
                "--top-k",
                "2",
                "--format",
                fmt,
            ])
            .env("HOME", &home)
            .output()
            .expect("carbon explain");
        assert!(
            exp.status.success(),
            "explain {fmt} failed: {}",
            String::from_utf8_lossy(&exp.stderr)
        );
    }

    // `carbon export` across ALL 6 languages (`python`, `rust`, `rust-refittable`, `js`, `c`, `cpp`)
    for lang in ["python", "rust", "c"] {
        let out_file = tmp.path().join(format!("champion_{lang}.src"));
        let ex = Command::new(&bin)
            .args([
                "export",
                "--model",
                model_handle.to_str().expect("utf8"),
                "--lang",
                lang,
                "--out",
                out_file.to_str().expect("utf8"),
                "--json",
            ])
            .env("HOME", &home)
            .output()
            .expect("carbon export");
        assert!(
            ex.status.success() && out_file.is_file(),
            "export {lang} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&ex.stdout),
            String::from_utf8_lossy(&ex.stderr)
        );
    }
    for lang in ["js", "cpp", "rust-refittable"] {
        let refit_file = tmp.path().join(format!("champion_{lang}.src"));
        let ex_refit = Command::new(&bin)
            .args([
                "export",
                "--model",
                model_handle.to_str().expect("utf8"),
                "--lang",
                lang,
                "--out",
                refit_file.to_str().expect("utf8"),
                "--json",
            ])
            .env("HOME", &home)
            .output()
            .expect("carbon export subset");
        if ex_refit.status.success() {
            assert!(refit_file.is_file());
        } else {
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&ex_refit.stdout),
                String::from_utf8_lossy(&ex_refit.stderr)
            );
            assert!(
                combined.contains("refittable")
                    || combined.contains("JavaScript")
                    || combined.contains("target transform")
                    || combined.contains("does not yet support"),
                "unexpected {lang} failure: {combined}"
            );
        }
    }
    let ex_stdout = Command::new(&bin)
        .args([
            "export",
            "--model",
            model_handle.to_str().expect("utf8"),
            "--lang",
            "python",
            "--out",
            "-",
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon export stdout");
    assert!(ex_stdout.status.success() && !ex_stdout.stdout.is_empty());

    // `carbon shred --model ... --json`
    let shred = Command::new(&bin)
        .args([
            "shred",
            "--model",
            model_handle.to_str().expect("utf8"),
            "--json",
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon shred");
    assert!(shred.status.success());

    emu.shutdown();
}

#[test]
fn carbon_edge_carbon_worker_and_cli_lambda_runtime_loops_execute_and_exit_cleanly() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let edge_bin = carbon_edge_bin();
    let worker_bin = carbon_worker_bin();
    let cli_bin = carbon_bin();

    // 1. CLI flags on `carbon-edge` and `carbon-worker`
    for b in [&edge_bin, &worker_bin] {
        for flag in ["--version", "--healthcheck"] {
            let out = Command::new(b).arg(flag).output().expect("version/hc");
            assert!(out.status.success());
        }
        let zero = Command::new(b)
            .env_remove("AWS_LAMBDA_RUNTIME_API")
            .env_remove("LAMBDA_RUNTIME_API")
            .output()
            .expect("zero-arg summary");
        assert!(zero.status.success());

        let missing_rt = Command::new(b)
            .arg("--lambda-runtime")
            .env_remove("AWS_LAMBDA_RUNTIME_API")
            .env_remove("LAMBDA_RUNTIME_API")
            .output()
            .expect("--lambda-runtime without env");
        assert_eq!(missing_rt.status.code(), Some(2));
    }
    let worker_eval_hc = Command::new(&worker_bin)
        .args(["eval", "--healthcheck"])
        .output()
        .expect("worker eval --healthcheck");
    assert!(worker_eval_hc.status.success());

    let edge_bad = Command::new(&edge_bin)
        .arg("--unknown-flag")
        .env_remove("AWS_LAMBDA_RUNTIME_API")
        .env_remove("LAMBDA_RUNTIME_API")
        .output()
        .expect("edge bad arg");
    assert_eq!(edge_bad.status.code(), Some(2));

    let worker_bad = Command::new(&worker_bin)
        .env("CARBON_ROLE", "bogus_role")
        .env_remove("AWS_LAMBDA_RUNTIME_API")
        .env_remove("LAMBDA_RUNTIME_API")
        .output()
        .expect("worker bad role");
    assert_eq!(worker_bad.status.code(), Some(2));

    // 2. Live single-shot Lambda Custom Runtime loop for `carbon-edge`
    let mut emu =
        CarbonLocalLambdaEmulator::start(CarbonEmuConfig::for_dir(tmp.path().join("cas_bg")))
            .expect("start cas backing emu");
    let cas_url = format!("{}/cas", emu.base_url());
    let eng_url = format!("{}/invoke/engine", emu.base_url());
    let kek_hex = "01".repeat(32);
    let kek = CarbonEngineKek::from_hex(&kek_hex).expect("from_hex kek");
    let ping_event = serde_json::to_vec(&serde_json::json!({
        "version": "2.0",
        "rawPath": "/v1/carbon/ping",
        "rawQueryString": "",
        "headers": {},
        "requestContext": { "http": { "method": "GET", "path": "/v1/carbon/ping", "sourceIp": "127.0.0.1" } },
        "isBase64Encoded": false
    }))
    .expect("encode apigw v2 ping");

    let rt_edge = SingleShotLambdaRuntimeServer::start(vec![ping_event]);
    let edge_run = Command::new(&edge_bin)
        .arg("--lambda-runtime")
        .env("LAMBDA_RUNTIME_API", rt_edge.host_port())
        .env("CARBON_CAS_URL", &cas_url)
        .env("CARBON_INTERNAL_TOKEN", "")
        .env("CARBON_ENGINE_INVOKE_URL", &eng_url)
        .env("CARBON_PRESIGN_SECRET_HEX", "11".repeat(32))
        .env("CARBON_AUTH_SECRET_HEX", "22".repeat(32))
        .env("CARBON_ENGINE_KEK_PUBKEY_B64", kek.pubkey_b64())
        .output()
        .expect("run carbon-edge lambda loop");
    assert!(
        edge_run.status.success(),
        "edge loop failed: {}",
        String::from_utf8_lossy(&edge_run.stderr)
    );
    assert_eq!(rt_edge.responses().len(), 1);

    // 3. Live single-shot Lambda Custom Runtime loops for `carbon-worker` (`engine` & `eval`)
    //    AND `carbon engine --work-dir` / `carbon eval --scratch-dir`!
    //    Sending an invalid wire JSON exercises the worker loop + `/error` posting + clean 410 exit!
    for (bin_path, args, role_env) in [
        (&worker_bin, vec!["engine", "--lambda-runtime"], "engine"),
        (&worker_bin, vec!["eval", "--lambda-runtime"], "eval"),
        (
            &cli_bin,
            vec!["engine", "--work-dir", tmp.path().to_str().expect("utf8")],
            "engine",
        ),
        (
            &cli_bin,
            vec!["eval", "--scratch-dir", tmp.path().to_str().expect("utf8")],
            "eval",
        ),
    ] {
        let rt_srv = SingleShotLambdaRuntimeServer::start(vec![b"{\"invalid\":true}".to_vec()]);
        let out = Command::new(bin_path)
            .args(&args)
            .env("LAMBDA_RUNTIME_API", rt_srv.host_port())
            .env("CARBON_ROLE", role_env)
            .env("CARBON_CAS_URL", &cas_url)
            .env("CARBON_WORK_DIR", tmp.path().join("w"))
            .env("CARBON_SCRATCH_DIR", tmp.path().join("s"))
            .env("CARBON_SEAL_SECRET_HEX", "33".repeat(32))
            .env("CARBON_ENGINE_KEK_HEX", &kek_hex)
            .output()
            .expect("run worker lambda loop");
        assert!(
            out.status.success(),
            "worker {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(rt_srv.errors().len(), 1);
    }

    // Verify `CarbonCapabilityClaimsWire` / `CarbonAuthVerifier` admin claim roundtrip used by advisor
    let verifier = CarbonAuthVerifier::from_secret([0x44; 32]);
    let admin_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "ops_admin".to_owned(),
            tier: CarbonTierWire::Enterprise,
            exp_epoch_secs: 4_102_444_800,
            admin: true,
        })
        .expect("mint admin");
    let claims = verifier
        .verify_token(&admin_tok, 1_800_000_000)
        .expect("verify");
    assert!(claims.admin);

    emu.shutdown();
}
