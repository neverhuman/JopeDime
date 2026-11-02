use super::*;

#[test]
fn login_input_is_not_persisted_in_history() {
    assert!(!super::remember("/login accidentally-pasted-credential"));
    assert!(!super::remember("\\login https://example.test"));
    assert!(super::remember("/open data.csv"));
}

fn create_history(path: &Path, entries: &[String]) {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    owner_mode(&mut options);
    let mut file = options.open(path).unwrap();
    for e in entries {
        writeln!(file, "{}", serde_json::to_string(e).unwrap()).unwrap();
    }
}
#[test]
fn history_retention_permissions_multiline_and_json_privacy_roundtrip() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("history");
    create_history(
        &path,
        &(0..1005).map(|i| format!("entry{i}")).collect::<Vec<_>>(),
    );
    let mut h = HistoryFile::open(&path).unwrap();
    assert_eq!(h.entries().len(), 1000);
    assert_eq!(h.entries()[0], "entry5");
    h.push("entry1004").unwrap();
    assert_eq!(h.entries().len(), 1000);
    h.push("\\open \"line one\nline two\"").unwrap();
    let small = format!("{{\"x\":\"{}\"}}", "a".repeat(1016));
    assert_eq!(small.len(), 1024);
    h.push(&format!("\\objective {small}")).unwrap();
    let count = h.entries().len();
    let last = h.entries().last().unwrap().clone();
    let big = format!("{{\"x\":\"{}\"}}", "a".repeat(1017));
    for command in ["\\objective", "/objective", "\\FITNESS"] {
        h.push(&format!("{command} {big}")).unwrap();
        assert_eq!(h.entries().len(), count);
        assert_eq!(h.entries().last().unwrap(), &last);
    }
    let reopened = HistoryFile::open(&path).unwrap();
    assert_eq!(reopened.entries(), h.entries());
    assert!(reopened
        .entries()
        .iter()
        .any(|s| s.contains("line one\nline two")));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
#[test]
fn history_navigation_and_reverse_search_restore_the_original_draft() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("history");
    let mut h = HistoryFile::open(&path).unwrap();
    for entry in ["\\open alpha.csv", "\\open beta.csv", "\\run"] {
        h.push(entry).unwrap();
    }
    let mut c = Composer::with_history(&path).unwrap();
    c.set_text("my draft").unwrap();
    key(&mut c, KeyCode::Left, KeyModifiers::NONE);
    let cursor = c.cursor_byte();
    key(&mut c, KeyCode::Up, KeyModifiers::NONE);
    assert_eq!(c.text(), "\\run");
    key(&mut c, KeyCode::Down, KeyModifiers::NONE);
    assert_eq!(c.text(), "my draft");
    assert_eq!(c.cursor_byte(), cursor);
    key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    for ch in "open".chars() {
        key(&mut c, KeyCode::Char(ch), KeyModifiers::NONE);
    }
    assert_eq!(c.text(), "\\open beta.csv");
    key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    assert_eq!(c.text(), "\\open alpha.csv");
    key(&mut c, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(c.text(), "my draft");
    assert_eq!(c.cursor_byte(), cursor);
    key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    assert_eq!(
        key(&mut c, KeyCode::Enter, KeyModifiers::NONE),
        ComposerAction::Submit("\\run".into())
    );
    assert_eq!(c.text(), "");
}
#[cfg(unix)]
#[test]
fn insecure_nonregular_symlink_and_busy_history_refuse_without_overwriting() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("history");
    fs::write(&path, "private").unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(HistoryFile::open(&path).is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), "private");
    fs::remove_file(&path).unwrap();
    let target = directory.path().join("target");
    create_history(&target, &["outside".into()]);
    symlink(&target, &path).unwrap();
    assert!(HistoryFile::open(&path).is_err());
    assert!(fs::read_to_string(&target).unwrap().contains("outside"));
    fs::remove_file(&path).unwrap();
    let mut h = HistoryFile::open(&path).unwrap();
    h.push("before").unwrap();
    let lock = regular_options()
        .write(true)
        .open(path.with_extension("lock"))
        .unwrap();
    lock.try_lock().unwrap();
    assert!(h.push("blocked").is_err());
    drop(lock);
    assert_eq!(HistoryFile::open(&path).unwrap().entries(), ["before"]);
    #[cfg(target_os = "linux")]
    {
        let fifo = directory.path().join("fifo");
        assert!(std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        let start = Instant::now();
        assert!(HistoryFile::open(&fifo).is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}
