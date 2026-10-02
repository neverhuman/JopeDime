fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&jope_pipeline::kernel2::receipt::derived_schema()).unwrap()
    );
}
