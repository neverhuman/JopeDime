//! Regenerate with `cargo run -q -p jope-pipeline --example receipt_schema`.
fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&jope_pipeline::receipt::derived_schema()).unwrap()
    );
}
