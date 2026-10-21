use std::path::PathBuf;
use std::process::Command;

#[allow(dead_code)]
#[path = "registry/codegen.rs"]
mod registry_codegen;

fn main() {
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
    println!("cargo:rerun-if-changed=cpp/fitted_runtime.cpp");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=JOPE_CXX_MARCH");
    // The fitted C++ runtime is compiled only with the `fitted-cpp` feature (default-on). The
    // `jopedime` appliance turns it off, so that build neither needs g++ nor embeds C++.
    if std::env::var_os("CARGO_FEATURE_FITTED_CPP").is_none() {
        return;
    }
    // ADR-0002 §5: the fitted runtime is compiled for a portable baseline by default so hosted CI
    // runners and the RTX box produce the same bits; `-ffp-contract=off -fno-fast-math` are what
    // make the result deterministic, the target ISA only changes speed. JOPE_CXX_MARCH opts in to
    // a native ISA (the historical value was znver2).
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "g++".to_owned());
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let march = match std::env::var("JOPE_CXX_MARCH") {
        Ok(m) if !m.is_empty() => Some(format!("-march={m}")),
        _ if target_arch == "x86_64" => Some("-march=x86-64-v2".to_owned()),
        _ => None,
    };
    let output = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set"))
        .join("libjope_fitted_runtime.so");
    let mut cmd = Command::new(&cxx);
    cmd.args(["-std=c++17", "-O3"]);
    if let Some(ref flag) = march {
        cmd.arg(flag);
    }
    cmd.args(["-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC"]);
    if target_os == "linux" {
        cmd.arg("-Wl,--build-id=none");
    }
    cmd.args(["cpp/fitted_runtime.cpp", "-ldl", "-o"])
        .arg(&output);
    let result = cmd.output().expect("launch g++ for fitted C++ runtime");
    if !result.status.success() {
        panic!(
            "compile fitted C++ runtime: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
