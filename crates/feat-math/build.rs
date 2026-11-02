#[path = "tests/go_support/generate.rs"]
mod go_golden;

#[allow(dead_code)]
#[path = "registry/codegen.rs"]
mod registry_codegen;

fn main() {
    go_golden::generate();
    println!("cargo:rerun-if-changed=registry/operators.toml");
    println!("cargo:rerun-if-changed=registry/codegen.rs");
    println!("cargo:rerun-if-changed=src/kernel/registry_v2/generated.rs");
    // Validate and compare without writing source. Staleness is fatal in the
    // explicit --check/test gate; a warning here keeps --write repair reachable.
    let input = std::fs::read_to_string("registry/operators.toml").expect("registry data");
    let expected = registry_codegen::generate(&input).expect("valid registry metadata");
    let actual = std::fs::read_to_string("src/kernel/registry_v2/generated.rs").unwrap_or_default();
    if registry_codegen::check(&expected, &actual).is_err() {
        println!(
            "cargo:warning=stale generated registry; run registry_codegen --write then --check"
        );
    }
}
