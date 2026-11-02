use super::*;
use std::fs;
use std::sync::{Mutex, MutexGuard};

/// The process environment is shared: every test that mutates it holds
/// this lock and restores what it touched.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const VARS: [&str; 11] = [
    var::JOPE_HOME,
    var::JOPE_STATE,
    var::JOPE_REGISTRY,
    var::SOTARENA_ROOT,
    var::SOTARENA_REPORTER_ROOT,
    var::JOPE_SCQ_ROOT,
    "JOPEDIME_HOME",
    "XDG_STATE_HOME",
    "XDG_DATA_HOME",
    "HOME",
    "LOCALAPPDATA",
];

struct EnvScope {
    _guard: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    cwd: PathBuf,
}

impl EnvScope {
    fn new() -> Self {
        let guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let saved = VARS.iter().map(|name| (*name, env::var_os(name))).collect();
        let cwd = env::current_dir().expect("cwd");
        for name in VARS {
            env::remove_var(name);
        }
        Self {
            _guard: guard,
            saved,
            cwd,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        env::set_current_dir(&self.cwd).expect("restore cwd");
        for (name, value) in self.saved.drain(..) {
            match value {
                Some(value) => env::set_var(name, value),
                None => env::remove_var(name),
            }
        }
    }
}

fn temp_dir(label: &str) -> PathBuf {
    let unique = format!(
        "jope-paths-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let dir = env::temp_dir().join(unique);
    fs::create_dir_all(&dir).expect("temp dir");
    dir.canonicalize().expect("canonical temp dir")
}

#[test]
fn home_prefers_the_environment() {
    let _scope = EnvScope::new();
    env::set_var(var::JOPE_HOME, "/opt/jope/checkout");
    assert_eq!(jope_home(), PathBuf::from("/opt/jope/checkout"));
    env::set_var(var::JOPE_HOME, "");
    assert_ne!(jope_home(), PathBuf::from(""), "an empty value is unset");
}

#[test]
fn jopedime_home_uses_environment_and_os_data_directories() {
    let _scope = EnvScope::new();

    env::set_var("JOPEDIME_HOME", "/configured/jopedime");
    assert_eq!(jopedime_home(), PathBuf::from("/configured/jopedime"));
    env::remove_var("JOPEDIME_HOME");

    assert_eq!(
        default_jopedime_home(
            "linux",
            None,
            Some(std::ffi::OsStr::new("/xdg/data")),
            Some(std::ffi::OsStr::new("/users/alice")),
            None,
        ),
        PathBuf::from("/xdg/data/jopedime")
    );
    assert_eq!(
        default_jopedime_home(
            "linux",
            None,
            None,
            Some(std::ffi::OsStr::new("/users/alice")),
            None,
        ),
        PathBuf::from("/users/alice/.local/share/jopedime")
    );
    assert_eq!(
        default_jopedime_home(
            "macos",
            None,
            None,
            Some(std::ffi::OsStr::new("/Users/alice")),
            None,
        ),
        PathBuf::from("/Users/alice/Library/Application Support/jopedime")
    );
    assert_eq!(
        default_jopedime_home(
            "windows",
            None,
            None,
            None,
            Some(std::ffi::OsStr::new(r"C:\Users\Alice\AppData\Local")),
        ),
        PathBuf::from(r"C:\Users\Alice\AppData\Local").join("jopedime")
    );
}

#[test]
fn home_walks_up_from_the_current_directory_to_the_marker() {
    let _scope = EnvScope::new();
    let root = temp_dir("marker");
    fs::write(root.join(HOME_MARKER), "").unwrap();
    let nested = root.join("a/b/c");
    fs::create_dir_all(&nested).unwrap();
    env::set_current_dir(&nested).unwrap();
    assert_eq!(jope_home().canonicalize().unwrap(), root);
    fs::remove_dir_all(&root).ok();
}

#[test]
fn home_resolution_is_independent_of_the_build_directory() {
    let root = temp_dir("resolution");
    let cwd_root = root.join("checkout");
    let exe_root = root.join("installed");
    let nested = cwd_root.join("nested");
    let exe = exe_root.join("bin/jope");
    let stray = root.join("no-marker");
    for dir in [&nested, exe.parent().unwrap(), &stray] {
        fs::create_dir_all(dir).unwrap();
    }
    fs::write(cwd_root.join(HOME_MARKER), "").unwrap();
    fs::write(exe_root.join(HOME_MARKER), "").unwrap();
    assert!(
        find_marker(&stray).is_none(),
        "fixture must have no ancestor marker"
    );

    // Environment overrides remain verbatim, including relative paths.
    assert_eq!(
        select_home(Some(Path::new("relative")), Some(&nested), Some(&exe)),
        PathBuf::from("relative")
    );
    assert_eq!(
        select_home(Some(Path::new("")), Some(&nested), Some(&exe)),
        cwd_root
    );
    assert_eq!(select_home(None, Some(&nested), Some(&exe)), cwd_root);
    assert_eq!(select_home(None, Some(&stray), Some(&exe)), exe_root);
    assert_eq!(select_home(None, None, Some(&exe)), exe_root);
    assert_eq!(
        select_home(None, Some(&stray), Some(&stray.join("bin"))),
        stray
    );
    assert_eq!(select_home(None, Some(&stray), None), stray);
    assert_eq!(select_home(None, None, None), PathBuf::from("."));
    assert_eq!(
        select_home(None, None, Some(&stray.join("bin"))),
        PathBuf::from(".")
    );

    // The nearest marker wins within each ancestry chain.
    fs::write(nested.join(HOME_MARKER), "").unwrap();
    assert_eq!(select_home(None, Some(&nested), Some(&exe)), nested);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn state_resolution_order() {
    let _scope = EnvScope::new();
    env::set_var(var::JOPE_STATE, "/var/lib/jope");
    assert_eq!(jope_state(), PathBuf::from("/var/lib/jope"));
    env::remove_var(var::JOPE_STATE);

    env::set_var("XDG_STATE_HOME", "/xdg/state");
    env::set_var("HOME", "/users/someone");
    assert_eq!(jope_state(), PathBuf::from("/xdg/state/jope"));

    env::set_var("XDG_STATE_HOME", "relative/ignored");
    assert_eq!(
        jope_state(),
        PathBuf::from("/users/someone/.local/state/jope")
    );

    env::remove_var("XDG_STATE_HOME");
    env::remove_var("HOME");
    env::set_var(var::JOPE_HOME, "/opt/jope/checkout");
    assert_eq!(jope_state(), PathBuf::from("/opt/jope/checkout/state"));
}

#[test]
fn registry_and_arena_roots_follow_home_and_state() {
    let _scope = EnvScope::new();
    env::set_var(var::JOPE_HOME, "/opt/jope/checkout");
    env::set_var(var::JOPE_STATE, "/var/lib/jope");
    assert_eq!(
        jope_registry(),
        PathBuf::from("/opt/jope/checkout/datasets/by-id")
    );
    assert_eq!(quant_root(), PathBuf::from("/opt/jope/checkout/datasets"));
    assert_eq!(
        sotarena_root(),
        PathBuf::from("/opt/jope/checkout/benchmarks/sotarena")
    );
    assert_eq!(
        sotarena_reporter_root(),
        PathBuf::from("/var/lib/jope/arena")
    );
    assert_eq!(scq_root(), PathBuf::from("/var/lib/jope/scq"));
    assert_eq!(legacy_root(), PathBuf::from("/var/lib/jope/legacy"));

    env::set_var(var::JOPE_REGISTRY, "/data/quant/regression");
    env::set_var(var::SOTARENA_ROOT, "/data/sotarena");
    env::set_var(var::SOTARENA_REPORTER_ROOT, "/data/arena");
    env::set_var(var::JOPE_SCQ_ROOT, "/data/smart_cluster");
    assert_eq!(jope_registry(), PathBuf::from("/data/quant/regression"));
    assert_eq!(quant_root(), PathBuf::from("/data/quant"));
    assert_eq!(sotarena_root(), PathBuf::from("/data/sotarena"));
    assert_eq!(sotarena_reporter_root(), PathBuf::from("/data/arena"));
    assert_eq!(scq_root(), PathBuf::from("/data/smart_cluster"));
}

#[test]
fn quant_root_of_a_bare_registry_is_itself() {
    let _scope = EnvScope::new();
    env::set_var(var::JOPE_REGISTRY, "registry");
    assert_eq!(quant_root(), PathBuf::from("registry"));
}

#[test]
fn string_form() {
    assert_eq!(to_string(Path::new("/x/y")), "/x/y");
}

once_str!(fn cached_home_child => jope_home().join("child"));

#[test]
fn once_str_resolves_once() {
    let first = cached_home_child();
    let second = cached_home_child();
    assert!(first.ends_with("/child"), "{first}");
    assert!(std::ptr::eq(first, second), "cached per process");
}
