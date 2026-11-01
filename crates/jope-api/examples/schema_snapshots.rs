#[path = "../tests/support/mod.rs"]
mod support;

fn main() {
    let snapshot = serde_json::to_string_pretty(&support::schemas()).unwrap();
    if let Some(path) = std::env::var_os("JOPE_SCHEMA_SNAPSHOT_OUTPUT") {
        std::fs::write(path, snapshot).unwrap();
    } else {
        println!("{snapshot}");
    }
}
