use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cpp/fitted_runtime.cpp");
    let output = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set"))
        .join("libjope_fitted_runtime.so");
    let result = Command::new("g++")
        .args([
            "-std=c++17",
            "-O3",
            "-march=znver2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-shared",
            "-fPIC",
            "-Wl,--build-id=none",
            "cpp/fitted_runtime.cpp",
            "-ldl",
            "-o",
        ])
        .arg(&output)
        .output()
        .expect("launch g++ for fitted C++ runtime");
    if !result.status.success() {
        panic!(
            "compile fitted C++ runtime: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
