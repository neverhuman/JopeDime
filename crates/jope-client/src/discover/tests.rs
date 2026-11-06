use super::*;
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const PROBE: &str = "discover::tests::native_client_probe";

struct Home(PathBuf);
impl Home {
    fn new() -> Self {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/client-discovery")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        private_home(&path).unwrap();
        Self(path)
    }
    fn options(&self) -> Options {
        Options {
            home: self.0.clone(),
            executable: self.0.join("missing-engine"),
            endpoint: None,
            mode: Mode::Interactive,
            engine_available: true,
        }
    }
}
impl Drop for Home {
    fn drop(&mut self) {
        // A remaining engine descriptor is retained for diagnosis, never removed underneath it.
        if !self.0.join("daemon.json").exists() {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}
fn write_record(home: &Path, url: &str, pid: u32) {
    let bytes = serde_json::to_vec(&serde_json::json!({"url":url,"pid":pid,"token":TOKEN,
            "version":"test","api_version":"1.0","started_at":"test-epoch"}))
    .unwrap();
    let mut file = private_options()
        .write(true)
        .create(true)
        .truncate(true)
        .open(home.join("daemon.json"))
        .unwrap();
    file.write_all(&bytes).unwrap();
}
/// Discovery without the version check: these fixtures report version "test".
fn discover_any(
    options: &Options,
    environment: Option<std::ffi::OsString>,
    timeout: Duration,
    start: impl FnOnce(&Options) -> std::io::Result<std::process::Child>,
) -> Result<Engine, Error> {
    discover_versioned(options, environment, timeout, start, None)
}
fn select(options: &Options) -> Result<Engine, Error> {
    discover_any(options, None, Duration::from_millis(250), |_| {
        panic!("unexpected spawn")
    })
}
fn server(status: u16, delay: Duration) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut byte = [0];
        while !bytes.ends_with(b"\r\n\r\n") {
            assert_eq!(stream.read(&mut byte).unwrap(), 1);
            bytes.push(byte[0]);
        }
        let request = String::from_utf8(bytes).unwrap().to_ascii_lowercase();
        assert!(request.starts_with("get /v1/system http/1.1\r\n"));
        assert!(request.contains(&format!("authorization: bearer {TOKEN}\r\n")));
        thread::sleep(delay);
        let body = r#"{"version":"test","api_version":"1.0","health":"ok","local_paths":true,"capabilities":{"tasks":[],"languages":[],"devices":[],"learned_guidance":false},"warnings":[],"telemetry":[]}"#;
        let _ = write!(stream, "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nJope-Api-Version: 1.0\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    });
    (url, handle)
}

#[test]
fn selection_precedence_and_batch_lite_contract() {
    let home = Home::new();
    let mut options = home.options();
    options.endpoint = Some("https://example.test:443".into());
    assert!(
        matches!(discover_any(&options, Some("bad".into()), STARTUP_TIMEOUT, |_| panic!()),
            Ok(Engine::Endpoint(url)) if url == "https://example.test:443")
    );
    options.endpoint = None;
    assert!(matches!(
        discover_any(
            &options,
            Some("http://localhost:9000".into()),
            STARTUP_TIMEOUT,
            |_| panic!()
        ),
        Ok(Engine::Endpoint(_))
    ));
    for url in [
        "",
        "file:///home",
        "http://a:0",
        "http://a/path",
        "http://token@a",
        "http://a?token=secret",
        "http://a#secret",
    ] {
        options.endpoint = Some(url.into());
        assert!(
            matches!(select(&options), Err(Error::InvalidEndpoint)),
            "{url}"
        );
    }
    options.endpoint = None;
    options.mode = Mode::Batch;
    assert!(matches!(select(&options), Ok(Engine::InProcess)));
    options.engine_available = false;
    let error = select(&options).unwrap_err();
    assert_eq!(error.to_string(), NO_ENGINE);
    assert_eq!(error.exit_code(), 9);
    assert!(!home.0.join(".discover.lock").exists());
}

#[test]
fn authenticated_live_metadata_precedes_batch_and_lite() {
    let home = Home::new();
    let (url, handle) = server(200, Duration::ZERO);
    write_record(&home.0, &url, std::process::id());
    let mut options = home.options();
    options.engine_available = false;
    options.mode = Mode::Batch;
    let engine = select(&options).unwrap();
    assert!(
        matches!(&engine, Engine::Background(background) if background.pid == std::process::id())
    );
    assert!(!format!("{engine:?}").contains(TOKEN));
    handle.join().unwrap();
    fs::remove_file(home.0.join("daemon.json")).unwrap();
}

#[test]
fn stale_malformed_and_refused_metadata_never_attach() {
    let home = Home::new();
    let mut options = home.options();
    options.engine_available = false;
    write_record(&home.0, "http://127.0.0.1:1", u32::MAX);
    assert!(matches!(select(&options), Err(Error::NoEngine)));
    fs::write(home.0.join("daemon.json"), b"{unfinished").unwrap();
    assert!(matches!(select(&options), Err(Error::NoEngine)));
    let (url, handle) = server(401, Duration::ZERO);
    write_record(&home.0, &url, std::process::id());
    assert!(matches!(select(&options), Err(Error::NoEngine)));
    handle.join().unwrap();
    fs::write(
        home.0.join("daemon.json"),
        vec![b' '; MAX_METADATA as usize + 1],
    )
    .unwrap();
    assert!(matches!(select(&options), Err(Error::InvalidMetadata)));
    fs::remove_file(home.0.join("daemon.json")).unwrap();
}

#[test]
fn health_probe_obeys_deadline() {
    let home = Home::new();
    let (url, handle) = server(200, Duration::from_millis(400));
    write_record(&home.0, &url, std::process::id());
    let mut options = home.options();
    options.engine_available = false;
    let before = Instant::now();
    assert!(matches!(
        discover_any(&options, None, Duration::from_millis(80), |_| panic!()),
        Err(Error::NoEngine)
    ));
    assert!(before.elapsed() < Duration::from_millis(350));
    handle.join().unwrap();
    fs::remove_file(home.0.join("daemon.json")).unwrap();
}

#[cfg(unix)]
#[test]
fn public_metadata_and_symlinks_are_refused() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let home = Home::new();
    write_record(&home.0, "http://127.0.0.1:1", std::process::id());
    fs::set_permissions(
        home.0.join("daemon.json"),
        fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    assert!(matches!(select(&home.options()), Err(Error::Io(_))));
    fs::rename(home.0.join("daemon.json"), home.0.join("other")).unwrap();
    symlink(home.0.join("other"), home.0.join("daemon.json")).unwrap();
    assert!(matches!(select(&home.options()), Err(Error::Io(_))));
    fs::remove_file(home.0.join("daemon.json")).unwrap();
}

fn probe(home: &Home, engine: &Path, lite: bool) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            PROBE,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env_remove("JOPEDIME_ENDPOINT")
        .env("JOPE_CLT02_HOME", &home.0)
        .env("JOPE_CLT02_ENGINE", engine)
        .env("JOPE_CLT02_LITE", if lite { "1" } else { "0" })
        .env("TOKIO_WORKER_THREADS", "2")
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1");
    command
}
#[test]
fn lite_process_exits_nine_with_documented_message() {
    let home = Home::new();
    let output = probe(&home, &home.options().executable, true)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(9));
    assert_eq!(String::from_utf8(output.stderr).unwrap().trim(), NO_ENGINE);
}

#[test]
fn invalid_child_exit_code_reports_configuration_error() {
    let home = Home::new();
    let mut command = probe(&home, &home.options().executable, true);
    command.env("JOPE_CLT02_EXIT", "not-an-integer");
    let output = command.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "invalid JOPE_CLT02_EXIT unexpectedly succeeded"
    );
    assert!(
        stderr.contains("JOPE_CLT02_EXIT must be a valid i32 exit code"),
        "expected the invalid exit-code diagnostic, got: {stderr}"
    );
}

#[test]
#[ignore = "subprocess fixture, invoked by supervised tests"]
fn native_client_probe() {
    let terminate: fn(i32) -> ! = std::process::exit;
    if let Ok(code) = std::env::var("JOPE_CLT02_EXIT") {
        let exit_code = code
            .parse::<i32>()
            .expect("JOPE_CLT02_EXIT must be a valid i32 exit code");
        terminate(exit_code);
    }
    let home = PathBuf::from(std::env::var_os("JOPE_CLT02_HOME").expect("supervised fixture home"));
    if let Some(gate) = std::env::var_os("JOPE_CLT02_GATE") {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !Path::new(&gate).exists() {
            assert!(Instant::now() < deadline, "parent never released clients");
            thread::sleep(POLL);
        }
    }
    let options = Options {
        home,
        executable: PathBuf::from(std::env::var_os("JOPE_CLT02_ENGINE").unwrap()),
        endpoint: None,
        mode: Mode::Interactive,
        engine_available: std::env::var("JOPE_CLT02_LITE").unwrap() != "1",
    };
    match discover(&options) {
        Ok(Engine::Background(background)) => println!(
            "DISCOVERY_RESULT {}",
            serde_json::json!({"pid":background.pid,"url":background.url})
        ),
        Ok(other) => panic!("unexpected selection: {other:?}"),
        Err(error) => {
            eprintln!("{error}");
            terminate(error.exit_code().into());
        }
    }
}

#[test]
fn losing_child_exit_is_not_readiness_and_retries_shared_metadata() {
    for exit in [0, 9] {
        let home = Home::new();
        let (url, handle) = server(200, Duration::ZERO);
        let directory = home.0.clone();
        let publisher = thread::spawn(move || {
            thread::sleep(Duration::from_millis(120));
            write_record(&directory, &url, std::process::id());
        });
        let result = discover_any(&home.options(), None, Duration::from_secs(2), |_| {
            // Both successful and unsuccessful child exits precede the winner's publication.
            let mut command = probe(&home, &home.options().executable, true);
            command.env("JOPE_CLT02_EXIT", exit.to_string());
            detached(&mut command)?;
            command.spawn()
        });
        assert!(matches!(result, Ok(Engine::Background(_))));
        publisher.join().unwrap();
        handle.join().unwrap();
        fs::remove_file(home.0.join("daemon.json")).unwrap();
    }
}

#[test]
fn held_store_lock_reports_busy_without_modifying_it() {
    let home = Home::new();
    let mut file = private_options()
        .write(true)
        .create_new(true)
        .open(home.0.join("lock"))
        .unwrap();
    let payload = format!("{}\nbatch command\n", std::process::id());
    file.write_all(payload.as_bytes()).unwrap();
    file.lock().unwrap();
    let error = discover_any(&home.options(), None, Duration::from_millis(80), |_| {
        let mut command = probe(&home, &home.options().executable, true);
        command.env("JOPE_CLT02_EXIT", "9");
        detached(&mut command)?;
        command.spawn()
    })
    .unwrap_err();
    assert!(matches!(error, Error::Busy { .. }));
    assert_eq!(error.exit_code(), 9);
    drop(file);
    assert_eq!(fs::read_to_string(home.0.join("lock")).unwrap(), payload);
}

#[test]
#[ignore = "requires JOPE_CLT02_ENGINE built from this source; native platform qualification"]
fn ten_native_clients_share_one_detached_engine() {
    let executable = PathBuf::from(
        std::env::var_os("JOPE_CLT02_ENGINE").expect("real full engine binary required"),
    );
    assert!(executable.is_file());
    let home = Home::new();
    let gate = home.0.join("release-clients");
    let mut children: Vec<_> = (0..10)
        .map(|_| {
            probe(&home, &executable, false)
                .env("JOPE_CLT02_GATE", &gate)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    fs::write(&gate, b"go").unwrap();
    let deadline = Instant::now() + Duration::from_secs(12);
    let mut results: Vec<serde_json::Value> = Vec::new();
    for child in &mut children {
        while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
            thread::sleep(POLL);
        }
    }
    let mut failures = Vec::new();
    for mut child in children {
        if child.try_wait().unwrap().is_none() {
            child.kill().unwrap();
        }
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        if !output.status.success() || !stdout.contains("1 passed; 0 failed") {
            failures.push(format!(
                "status {} stdout {stdout} stderr {stderr}",
                output.status
            ));
            continue;
        }
        let line = stdout
            .lines()
            .find_map(|line| line.split_once("DISCOVERY_RESULT ").map(|(_, json)| json));
        if let Some(result) = line.and_then(|line| serde_json::from_str(line).ok()) {
            results.push(result);
        } else {
            failures.push(format!("missing attachment inventory: {stdout}"));
        }
    }
    // Stop our fixture even when an assertion below will fail.
    let descriptor = record(&home.0).unwrap();
    let mut detached_session = true;
    if let Some(record) = &descriptor {
        assert!(
            alive(record.pid),
            "engine must outlive all client processes"
        );
        #[cfg(unix)]
        {
            detached_session = session(record.pid) == record.pid as i32;
        }
    }
    let stop = Command::new(&executable)
        .args(["serve", "--stop"])
        .env("JOPEDIME_HOME", &home.0)
        .env("TOKIO_WORKER_THREADS", "2")
        .output()
        .unwrap();
    let stop_deadline = Instant::now() + Duration::from_secs(5);
    while home.0.join("daemon.json").exists() && Instant::now() < stop_deadline {
        thread::sleep(POLL);
    }
    assert!(
        stop.status.success(),
        "fixture shutdown: {}",
        String::from_utf8_lossy(&stop.stderr)
    );
    assert!(
        !home.0.join("daemon.json").exists(),
        "fixture engine did not clean up"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(results.len(), 10);
    assert!(results.iter().all(|result| result == &results[0]));
    assert_eq!(
        Some(results[0]["pid"].as_u64().unwrap()),
        descriptor.map(|r| r.pid as u64)
    );
    assert!(detached_session, "Unix engine must lead its own session");
    println!(
        "NATIVE_DISCOVERY {}",
        serde_json::json!({"platform":std::env::consts::OS,"clients":results,"detached_session":detached_session,"shutdown":true})
    );
}
#[cfg(unix)]
#[allow(unsafe_code)]
fn session(pid: u32) -> i32 {
    unsafe extern "C" {
        fn getsid(pid: i32) -> i32;
    }
    // SAFETY: read-only session query for our own fixture's positive PID.
    unsafe { getsid(pid as i32) }
}

const SNAPSHOT: &str = r#"{"run_id":"run_01JBD8Q6N4V2T5Y8K3M7P9R1S0","revision":"1","state":"STATE","phase":"searching","pending":null,"effort":"fast","budget_used":{"evaluations":"0","wall_seconds":"0.0"},"budget_total":{"evaluations":"10","wall_seconds":"60"},"best":null,"event_watermark":"1","warnings":[],"completion_reason":null}"#;

/// A fake engine answering `/v1/system`, `/v1/runs` and `/__engine/stop` (which removes its
/// descriptor, as a real engine does on exit). Returns its URL and the paths it was asked for.
fn engine(
    home: PathBuf,
    run_state: Option<&'static str>,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = seen.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut byte = [0];
            while !bytes.ends_with(b"\r\n\r\n") {
                if stream.read(&mut byte).unwrap_or(0) != 1 {
                    break;
                }
                bytes.push(byte[0]);
            }
            let request = String::from_utf8_lossy(&bytes).to_string();
            let path = request.split_whitespace().nth(1).unwrap_or("").to_owned();
            log.lock().unwrap().push(path.clone());
            let (status, body) = if path.starts_with("/v1/system") {
                (200, r#"{"version":"0.9.9","api_version":"1.0","health":"ok","local_paths":true,"capabilities":{"tasks":[],"languages":[],"devices":[],"learned_guidance":false},"warnings":[],"telemetry":[]}"#.to_owned())
            } else if path.starts_with("/v1/runs") {
                let runs = run_state
                    .map(|s| SNAPSHOT.replace("STATE", s))
                    .unwrap_or_default();
                (200, format!(r#"{{"runs":[{runs}],"next_after":null}}"#))
            } else if path == "/__engine/stop" {
                let _ = fs::remove_file(home.join("daemon.json"));
                (204, String::new())
            } else {
                (404, String::new())
            };
            let _ = write!(stream, "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nJope-Api-Version: 1.0\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        }
    });
    (url, seen)
}

fn write_versioned(home: &Path, url: &str, version: &str) {
    write_built(home, url, version, None);
}

fn write_built(home: &Path, url: &str, version: &str, build: Option<&str>) {
    let mut record = serde_json::json!({"url":url,"pid":std::process::id(),"token":TOKEN,
            "version":version,"api_version":"1.0","started_at":"test-epoch"});
    if let Some(build) = build {
        record["build"] = build.into();
    }
    let bytes = serde_json::to_vec(&record).unwrap();
    let mut file = private_options()
        .write(true)
        .create(true)
        .truncate(true)
        .open(home.join("daemon.json"))
        .unwrap();
    file.write_all(&bytes).unwrap();
}

#[test]
fn a_matching_engine_version_is_reused() {
    let home = Home::new();
    let (url, seen) = engine(home.0.clone(), None);
    write_versioned(&home.0, &url, "7.7.7");
    let selected = discover_versioned(
        &home.options(),
        None,
        Duration::from_secs(2),
        |_| panic!("no spawn"),
        Some(Expected {
            version: "7.7.7",
            build: None,
        }),
    );
    assert!(matches!(selected, Ok(Engine::Background(b)) if b.version == "7.7.7"));
    assert!(!seen.lock().unwrap().iter().any(|p| p == "/__engine/stop"));
    let _ = fs::remove_file(home.0.join("daemon.json"));
}

#[test]
fn an_idle_engine_of_another_version_is_stopped_and_replaced() {
    let home = Home::new();
    let (url, seen) = engine(home.0.clone(), Some("completed"));
    write_versioned(&home.0, &url, "1.0.1");
    let mut options = home.options();
    options.mode = Mode::Batch;
    let selected = discover_versioned(
        &options,
        None,
        Duration::from_secs(5),
        |_| panic!("batch never spawns"),
        Some(Expected {
            version: "1.0.0",
            build: None,
        }),
    );
    assert!(
        matches!(selected, Ok(Engine::InProcess)),
        "{:?}",
        selected.err()
    );
    assert!(seen.lock().unwrap().iter().any(|p| p == "/__engine/stop"));
    assert!(!home.0.join("daemon.json").exists());
    let notice = take_notice().unwrap();
    assert_eq!(
        notice,
        "Restarted the background engine: it was jAIn 1.0.1, now 1.0.0"
    );
}

#[test]
fn a_busy_engine_of_another_version_is_left_running_and_explained() {
    let home = Home::new();
    let (url, seen) = engine(home.0.clone(), Some("running"));
    write_versioned(&home.0, &url, "1.0.1");
    let selected = discover_versioned(
        &home.options(),
        None,
        Duration::from_secs(2),
        |_| panic!("no spawn"),
        Some(Expected {
            version: "1.0.0",
            build: None,
        }),
    );
    let error = selected.err().unwrap();
    assert!(
        matches!(&error, Error::OutdatedBusy { version, running } if version == "1.0.1" && running.len() == 1)
    );
    assert_eq!(
        error.to_string(),
        "An older jAIn engine (1.0.1) is still running a job (run_01JBD8Q6N4V2T5Y8K3M7P9R1S0). Wait for it or run `jain serve --stop --force`."
    );
    assert!(!seen.lock().unwrap().iter().any(|p| p == "/__engine/stop"));
    assert!(
        home.0.join("daemon.json").exists(),
        "the busy engine keeps its descriptor"
    );
    let _ = fs::remove_file(home.0.join("daemon.json"));
}

const BUILD_A: &str = "51c0c1a4e1eef9a16aaa24d33804fca6c298a2a3";
const BUILD_B: &str = "2ab7f551c9b1aaaabbbbccccddddeeeeffff0000";

#[test]
fn an_engine_of_the_same_build_is_reused() {
    let home = Home::new();
    let (url, seen) = engine(home.0.clone(), None);
    write_built(&home.0, &url, "1.0.0", Some(BUILD_A));
    let selected = discover_versioned(
        &home.options(),
        None,
        Duration::from_secs(2),
        |_| panic!("no spawn"),
        Some(Expected {
            version: "1.0.0",
            build: Some(BUILD_A),
        }),
    );
    assert!(
        matches!(&selected, Ok(Engine::Background(b)) if b.build.as_deref() == Some(BUILD_A)),
        "{:?}",
        selected.err()
    );
    assert!(!seen.lock().unwrap().iter().any(|p| p == "/__engine/stop"));
    let _ = fs::remove_file(home.0.join("daemon.json"));
}

/// #2141: rc.1 and rc.2 both report 1.0.0; the build (source commit) tells them apart.
#[test]
fn an_idle_engine_of_another_build_with_the_same_version_is_replaced() {
    for (theirs, shown) in [
        (Some(BUILD_B), "1.0.0 (build 2ab7f55)"),
        // A descriptor from an engine that predates build identities.
        (None, "1.0.0 (another build)"),
    ] {
        let home = Home::new();
        let (url, seen) = engine(home.0.clone(), Some("completed"));
        write_built(&home.0, &url, "1.0.0", theirs);
        let mut options = home.options();
        options.mode = Mode::Batch;
        let selected = discover_versioned(
            &options,
            None,
            Duration::from_secs(5),
            |_| panic!("batch never spawns"),
            Some(Expected {
                version: "1.0.0",
                build: Some(BUILD_A),
            }),
        );
        assert!(
            matches!(selected, Ok(Engine::InProcess)),
            "{:?}",
            selected.err()
        );
        assert!(seen.lock().unwrap().iter().any(|p| p == "/__engine/stop"));
        assert_eq!(
            take_notice().unwrap(),
            format!(
                "Restarted the background engine: it was jAIn {shown}, now 1.0.0 (build 51c0c1a)"
            )
        );
    }
}

#[test]
fn a_busy_engine_of_another_build_is_refused_like_another_version() {
    let home = Home::new();
    let (url, seen) = engine(home.0.clone(), Some("running"));
    write_built(&home.0, &url, "1.0.0", Some(BUILD_B));
    let selected = discover_versioned(
        &home.options(),
        None,
        Duration::from_secs(2),
        |_| panic!("no spawn"),
        Some(Expected {
            version: "1.0.0",
            build: Some(BUILD_A),
        }),
    );
    assert!(
        matches!(selected.err(), Some(Error::OutdatedBusy { version, .. }) if version == "1.0.0 (build 2ab7f55)")
    );
    assert!(!seen.lock().unwrap().iter().any(|p| p == "/__engine/stop"));
    let _ = fs::remove_file(home.0.join("daemon.json"));
}

#[test]
fn a_development_client_without_a_build_compares_versions_only() {
    let home = Home::new();
    let (url, seen) = engine(home.0.clone(), None);
    write_built(&home.0, &url, "1.0.0", Some(BUILD_B));
    let selected = discover_versioned(
        &home.options(),
        None,
        Duration::from_secs(2),
        |_| panic!("no spawn"),
        Some(Expected {
            version: "1.0.0",
            build: None,
        }),
    );
    assert!(matches!(selected, Ok(Engine::Background(_))));
    assert!(!seen.lock().unwrap().iter().any(|p| p == "/__engine/stop"));
    let _ = fs::remove_file(home.0.join("daemon.json"));
    assert_eq!(valid_build(Some("abc")), None);
    assert_eq!(valid_build(Some(BUILD_A)), Some(BUILD_A));
}
