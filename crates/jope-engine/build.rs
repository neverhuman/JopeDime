fn main() {
    println!("cargo:rerun-if-changed=native/runtime.cpp");
    if std::env::var_os("CARGO_FEATURE_NEURAL").is_none() {
        return;
    }
    for key in [
        "LIBTORCH",
        "LIBTORCH_CXX11_ABI",
        "CUDA_HOME",
        "JOPE_NCCL_INCLUDE",
        "JOPE_CUDA_LIBRARY_PATH",
        "JOPE_CUDART_SONAME",
        "JOPE_NCCL_SONAME",
        "CXX",
        "PYTHON",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let (torch, abi) = if let Ok(torch) = std::env::var("LIBTORCH") {
        (
            torch,
            std::env::var("LIBTORCH_CXX11_ABI").expect("explicit SDK requires LIBTORCH_CXX11_ABI"),
        )
    } else {
        let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
        let output = std::process::Command::new(python).args(["-c", "import torch,os; print(os.path.dirname(torch.__file__)); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"]).output().expect("Python torch discovery");
        assert!(output.status.success(), "PyTorch discovery failed");
        let text = String::from_utf8(output.stdout).unwrap();
        let mut lines = text.lines();
        (
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
        )
    };
    assert!(abi == "0" || abi == "1", "invalid LibTorch C++ ABI");
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nccl_include =
        std::env::var("JOPE_NCCL_INCLUDE").unwrap_or_else(|_| format!("{cuda}/include"));
    cc::Build::new()
        .cpp(true)
        .compiler(std::env::var("CXX").unwrap_or_else(|_| "clang++".into()))
        .warnings(false)
        .file("native/runtime.cpp")
        .include(format!("{torch}/include"))
        .include(format!("{torch}/include/torch/csrc/api/include"))
        .include(format!("{cuda}/include"))
        .include(nccl_include)
        .define("_GLIBCXX_USE_CXX11_ABI", Some(abi.as_str()))
        .flag("-std=c++20")
        .flag("-g0")
        .compile("parallel_native");
    println!("cargo:rustc-link-search=native={torch}/lib");
    // SmartCluster clears the worker environment. CUDA's indirect dependencies
    // must resolve through the qualified SDK paths without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    let library_paths =
        std::env::var("JOPE_CUDA_LIBRARY_PATH").unwrap_or_else(|_| format!("{cuda}/lib64"));
    for path in std::env::split_paths(&library_paths) {
        println!("cargo:rustc-link-search=native={}", path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
    for (variable, fallback) in [
        ("JOPE_CUDART_SONAME", "cudart"),
        ("JOPE_NCCL_SONAME", "nccl"),
    ] {
        if let Ok(soname) = std::env::var(variable) {
            assert!(
                !soname.contains('/') && soname.starts_with("lib") && soname.contains(".so"),
                "invalid CUDA SONAME"
            );
            println!("cargo:rustc-link-lib=dylib:+verbatim={soname}");
        } else {
            println!("cargo:rustc-link-lib={fallback}");
        }
    }
    for lib in ["torch_cuda", "c10_cuda"] {
        println!("cargo:rustc-link-lib={lib}");
    }
    // CUDA operators register through shared-library initializers. Without an
    // explicit retained dependency, --as-needed can discard libtorch_cuda even
    // though CUDA driver calls link successfully through c10_cuda.
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{torch}/lib");
}
