fn main() {
    println!("cargo:rerun-if-changed=native/runtime.cpp");
    if std::env::var_os("CARGO_FEATURE_NEURAL").is_none() {
        return;
    }
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let output=std::process::Command::new(python).args(["-c","import torch,os; print(os.path.dirname(torch.__file__)); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"]).output().expect("Python torch discovery");
    assert!(
        output.status.success(),
        "PyTorch is required for the neural build"
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let mut lines = text.lines();
    let torch = lines.next().unwrap();
    let abi = lines.next().unwrap();
    cc::Build::new()
        .cpp(true)
        .compiler(std::env::var("CXX").unwrap_or_else(|_| "clang++".into()))
        .warnings(false)
        .file("native/runtime.cpp")
        .include(format!("{torch}/include"))
        .include(format!("{torch}/include/torch/csrc/api/include"))
        .include("/usr/local/cuda/include")
        .define("_GLIBCXX_USE_CXX11_ABI", Some(abi))
        .flag("-std=c++17")
        .flag("-g0")
        .compile("parallel_native");
    println!("cargo:rustc-link-search=native={torch}/lib");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    for lib in ["torch_cuda", "c10_cuda", "nccl", "cudart"] {
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
