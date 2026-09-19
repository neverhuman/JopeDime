#![allow(dead_code)]
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub struct Run {
    pub status: i32,
    pub stdout: Vec<u8>,
    pub stderr: String,
}

impl Run {
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.stdout).unwrap_or_else(|e| {
            panic!(
                "stdout is not JSON: {e}\n{}\n{}",
                String::from_utf8_lossy(&self.stdout),
                self.stderr
            )
        })
    }
    pub fn ok(self) -> Self {
        assert_eq!(
            self.status,
            0,
            "exit {}\nstdout: {}\nstderr: {}",
            self.status,
            String::from_utf8_lossy(&self.stdout),
            self.stderr
        );
        self
    }
}

/// Run the `jope-learning` binary against `state` with no mirror configured.
pub fn jl(state: &Path, args: &[&str]) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_jope-learning"))
        .env_remove("JOPE_LEARNING_URL")
        .env("JOPE_STATE", state)
        .args(args)
        .output()
        .expect("spawn jope-learning");
    Run {
        status: out.status.code().unwrap_or(-1),
        stdout: out.stdout,
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}
