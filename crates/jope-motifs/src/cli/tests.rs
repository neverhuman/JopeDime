use super::*;
use crate::export::OWNER_BANNED;
use crate::fixture::{self, FakeState};
use crate::mine::InputManifest;
use crate::source::TaskLabel;
use crate::teaching::{no_reader, EligibleState};
use clap::CommandFactory;
use feat_math::transpile::lint_shipped_text;
use serde_json::Value;

fn env_with(policy: Policy, open_state: StateOpener) -> Env {
    Env { policy, open_state }
}

/// An opener that serves a copy of `state` for any root.
fn opener(state: FakeState) -> StateOpener {
    Box::new(move |_: Option<&Path>| -> Result<Box<dyn EligibleState>> {
        Ok(Box::new(state.clone()))
    })
}

/// An opener that must never run: every refusal it guards happens before the State is read.
fn unopened() -> StateOpener {
    Box::new(|_: Option<&Path>| -> Result<Box<dyn EligibleState>> {
        panic!("the State is read only after every path and parameter check")
    })
}

/// The test policy with a State in memory.
fn state_env(state: FakeState) -> Env {
    env_with(Policy::for_test(Vec::new(), false), opener(state))
}

fn invoke(arguments: &[&str]) -> Run {
    let env = env_with(Policy::for_test(Vec::new(), false), Box::new(no_reader));
    invoke_in(&env, arguments)
}

fn invoke_in(env: &Env, arguments: &[&str]) -> Run {
    let cli = Cli::try_parse_from(arguments.iter().copied()).expect("arguments parse");
    let mut out = Vec::new();
    let mut err = Vec::new();
    let status = run_with(cli, env, &mut out, &mut err).expect("streams");
    Run {
        status,
        out: String::from_utf8(out).expect("utf-8"),
        err: String::from_utf8(err).expect("utf-8"),
    }
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("utf-8 path")
}

fn written(directory: &Path, name: &str, table: &MotifTable) -> String {
    let path = directory.join(name);
    std::fs::write(&path, table.to_json().expect("json")).expect("write table");
    path.to_str().expect("utf-8 path").to_owned()
}

fn read_table(directory: &Path) -> MotifTable {
    let bytes = std::fs::read(directory.join(TABLE_FILE)).expect("table written");
    MotifTable::load(&bytes).expect("table loads")
}

fn read_inputs(directory: &Path) -> InputManifest {
    let bytes = std::fs::read(directory.join(INPUTS_FILE)).expect("inputs written");
    serde_json::from_slice(&bytes).expect("inputs parse")
}

fn assert_clean(text: &str, table: &MotifTable) {
    assert!(lint_shipped_text(text).is_empty(), "{text}");
    let lower = text.to_lowercase();
    for word in OWNER_BANNED {
        assert!(!lower.contains(word), "{word}");
    }
    for sentinel in fixture::sentinels(table) {
        assert!(!lower.contains(&sentinel.to_lowercase()), "{sentinel}");
    }
    assert!(!text.contains("/home/") && !text.contains("/tmp/"));
}

#[path = "mining.rs"]
mod mining;
#[path = "output.rs"]
mod output;
