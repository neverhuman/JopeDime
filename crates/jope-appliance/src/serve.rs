//! Process integration for the existing clap command and engine lifecycle adapter.

use super::{Cli, Command};
use clap::Parser;
use std::{
    path::{Path, PathBuf},
    process::{Child, Command as ProcessCommand, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

static CHILD_ID: AtomicU64 = AtomicU64::new(0);

struct ServeChild {
    child: Option<Child>,
    started: PathBuf,
}

impl ServeChild {
    fn spawn(engine_home: &Path, args: &[&str]) -> Self {
        let helper = concat!(module_path!(), "::serve_child");
        let helper = helper.split_once("::").unwrap().1;
        let started = engine_home.parent().unwrap().join(format!(
            "serve-child-{}.started",
            CHILD_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let child = ProcessCommand::new(std::env::current_exe().unwrap())
            .args([helper, "--exact", "--ignored", "--nocapture", "--quiet"])
            .arg("--test-threads=1")
            .env_clear()
            .env("JOPEDIME_HOME", engine_home)
            .env("JOPE_SERVE_TEST_ARGS", serde_json::to_string(args).unwrap())
            .env("JOPE_SERVE_TEST_STARTED", &started)
            .env("TOKIO_WORKER_THREADS", "2")
            .env("RAYON_NUM_THREADS", "2")
            .env("OMP_NUM_THREADS", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        Self {
            child: Some(child),
            started,
        }
    }

    fn id(&self) -> u32 {
        self.child.as_ref().unwrap().id()
    }

    fn finish(mut self) -> Output {
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.child.as_mut().unwrap().try_wait().unwrap().is_none() {
            assert!(Instant::now() < deadline, "serve child did not terminate");
            std::thread::sleep(Duration::from_millis(25));
        }
        let output = self.child.take().unwrap().wait_with_output().unwrap();
        assert!(
            self.started.is_file(),
            "the child fixture must execute; a zero-test exit is insufficient"
        );
        output
    }
}

impl Drop for ServeChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

fn command(engine_home: &Path, args: &[&str]) -> Output {
    ServeChild::spawn(engine_home, args).finish()
}

fn assert_code(output: &Output, expected: i32) {
    assert_eq!(
        output.status.code(),
        Some(expected),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn wait_ready(server: &mut ServeChild, engine_home: &Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(
            server.child.as_mut().unwrap().try_wait().unwrap().is_none(),
            "the engine must stay running until readiness"
        );
        if engine_home.join("daemon.json").is_file() {
            let status = command(engine_home, &["serve", "--status"]);
            assert_code(&status, 0);
            if String::from_utf8_lossy(&status.stdout)
                .contains(&format!("running (pid {})", server.id()))
            {
                return;
            }
        }
        assert!(Instant::now() < deadline, "engine readiness timed out");
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn stop(server: ServeChild, engine_home: &Path) {
    assert_code(&command(engine_home, &["serve", "--stop"]), 0);
    assert_code(&server.finish(), 0);
    assert!(!engine_home.join("daemon.json").exists());
}

#[test]
fn serve_starts_reports_status_and_stops() {
    let dir = tempfile::tempdir().unwrap();
    let engine_home = dir.path().join("engine");
    let mut server = ServeChild::spawn(&engine_home, &["serve", "--idle-exit", "30s"]);
    wait_ready(&mut server, &engine_home);
    let status = command(&engine_home, &["serve", "--status"]);
    assert_code(&status, 0);
    assert!(
        String::from_utf8_lossy(&status.stdout).contains(&format!("running (pid {})", server.id()))
    );
    stop(server, &engine_home);
    let stopped = command(&engine_home, &["serve", "--status"]);
    assert_code(&stopped, 0);
    assert!(String::from_utf8_lossy(&stopped.stdout)
        .lines()
        .any(|line| line == "stopped"));
}

#[test]
fn serve_idle_exit_removes_descriptor_and_allows_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let engine_home = dir.path().join("engine");
    let started = Instant::now();
    let mut server = ServeChild::spawn(&engine_home, &["serve", "--idle-exit", "2s"]);
    wait_ready(&mut server, &engine_home);
    assert_code(&server.finish(), 0);
    assert!(started.elapsed() >= Duration::from_secs(2));
    assert!(!engine_home.join("daemon.json").exists());
    let mut reopened = ServeChild::spawn(&engine_home, &["serve", "--idle-exit", "30s"]);
    wait_ready(&mut reopened, &engine_home);
    stop(reopened, &engine_home);
}

#[test]
fn second_serve_reports_the_live_lock_holder() {
    let dir = tempfile::tempdir().unwrap();
    let engine_home = dir.path().join("engine");
    let mut server = ServeChild::spawn(&engine_home, &["serve", "--idle-exit", "30s"]);
    wait_ready(&mut server, &engine_home);
    let busy = command(&engine_home, &["serve", "--idle-exit", "30s"]);
    assert_code(&busy, 9);
    let stderr = String::from_utf8_lossy(&busy.stderr);
    assert!(stderr.contains("ENGINE_BUSY"));
    assert!(stderr.contains(&format!("pid={}", server.id())));
    stop(server, &engine_home);
}

#[test]
#[ignore = "lifecycle child; parent integration tests invoke it with an execution marker"]
fn serve_child() {
    let started = std::env::var_os("JOPE_SERVE_TEST_STARTED").expect("supervisor execution marker");
    std::fs::write(PathBuf::from(started), b"started").unwrap();
    let args: Vec<String> =
        serde_json::from_str(&std::env::var("JOPE_SERVE_TEST_ARGS").unwrap()).unwrap();
    let cli = Cli::try_parse_from(std::iter::once("jopedime".to_owned()).chain(args))
        .unwrap_or_else(|error| error.exit());
    let Command::Serve {
        status,
        stop,
        idle_exit,
    } = cli.command
    else {
        panic!("the lifecycle fixture accepts only serve commands");
    };
    std::process::exit(i32::from(crate::cmd::serve::run(
        status,
        stop,
        idle_exit.as_deref(),
    )));
}
