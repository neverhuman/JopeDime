use std::{fs, os::unix::process::CommandExt, process::{Child, Command}, thread, time::{Duration, Instant}};

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        unsafe { libc::kill(-(self.0.id() as i32), libc::SIGKILL); }
        let _ = self.0.wait();
    }
}

#[test]
fn comparison_cli_receives_termination_without_leaving_a_controller_child() {
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir(directory.path().join("tools")).unwrap();
    fs::write(directory.path().join("tools/seed_experiment.py"), r#"
import json, os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--root') + 1])
def stop(sig, frame):
    (root / 'terminated').write_text(str(sig))
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(root / 'identity.json').write_text(json.dumps({'pid': os.getpid(), 'args': sys.argv[1:]}))
while True:
    time.sleep(.01)
"#).unwrap();
    let mut process = OwnedChild(Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root").arg(directory.path()).args(["compare-seeds", "--manifest", "frozen.json", "--workers", "16", "--prefix", "64", "--seconds", "7", "--downstream"])
        .process_group(0).spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(5);
    let identity = loop {
        if let Ok(bytes) = fs::read(directory.path().join("identity.json")) {
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) { break value; }
        }
        assert!(Instant::now() < deadline, "comparison controller did not start");
        thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(identity["pid"].as_u64().unwrap(), process.0.id() as u64);
    assert!(identity["args"].as_array().unwrap().iter().any(|v| v == "--downstream"));
    assert_eq!(unsafe { libc::kill(process.0.id() as i32, libc::SIGTERM) }, 0);
    loop {
        if let Some(status) = process.0.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        assert!(Instant::now() < deadline, "termination did not reach the comparison controller");
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(fs::read_to_string(directory.path().join("terminated")).unwrap(), libc::SIGTERM.to_string());
}
