use feat_math::cuda::Emitted;
use std::{fs, path::Path, process::Command};

pub fn enabled() -> bool {
    std::env::var_os("JOPE_CUDA_PARITY").is_some()
}
pub fn command(root: &Path, program: &str, args: &[&str]) -> String {
    let result = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .unwrap_or_else(|e| panic!("required tool {program}: {e}"));
    assert!(
        result.status.success(),
        "{program} {args:?}:\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    for line in String::from_utf8_lossy(&result.stderr)
        .lines()
        .filter(|s| s.starts_with("CUDA_LINEAR_DEVICE_PASS "))
    {
        eprintln!("{line}");
    }
    String::from_utf8(result.stdout).unwrap()
}
pub fn build(root: &Path, bundle: &Emitted) {
    fs::create_dir_all(root).unwrap();
    for (path, source) in &bundle.files {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, source).unwrap();
    }
    command(
        &root.join("cuda"),
        "rustc",
        &[
            "--edition=2021",
            "-O",
            "-C",
            "panic=unwind",
            "--crate-type=staticlib",
            "model_native.rs",
            "-o",
            "libmodel_native.a",
        ],
    );
}
pub fn consumer(root: &Path, source: &str) -> String {
    fs::write(root.join("consumer.cpp"), source).unwrap();
    let mode = std::env::var("JOPE_CUDA_PARITY").unwrap_or_else(|_| "host".into());
    match mode.as_str() {
        "host" => {
            command(
                root,
                "g++",
                &[
                    "-x",
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-DJOPE_CUDA_HOST_ONLY",
                    "-ffp-contract=off",
                    "-fno-fast-math",
                    "model.cu",
                    "consumer.cpp",
                    "-x",
                    "none",
                    "libmodel_native.a",
                    "-ldl",
                    "-lpthread",
                    "-lm",
                    "-o",
                    "consumer",
                ],
            );
        }
        "compile" | "device" => {
            let compiler = std::env::var("JOPE_NVCC").unwrap_or_else(|_| "nvcc".into());
            command(
                root,
                &compiler,
                &[
                    "-std=c++17",
                    "-O2",
                    "--fmad=false",
                    "--ftz=false",
                    "--prec-div=true",
                    "--prec-sqrt=true",
                    "-Xcompiler=-ffp-contract=off",
                    "-Xcompiler=-fno-fast-math",
                    "-Xcompiler=-Wall,-Wextra,-Werror",
                    "-DJOPE_CUDA_COMPILED",
                    "model.cu",
                    "consumer.cpp",
                    "libmodel_native.a",
                    "-ldl",
                    "-lpthread",
                    "-lm",
                    "-o",
                    "consumer",
                ],
            );
        }
        _ => panic!("invalid JOPE_CUDA_PARITY={mode:?}; expected host, compile, or device"),
    }
    let out = command(root, "./consumer", &[]);
    eprintln!("CUDA C++ {mode}: caller executed");
    out
}
fn number(value: f64) -> String {
    format!("d(UINT64_C({}))", value.to_bits())
}
fn vector(v: &[f64]) -> String {
    format!(
        "{{{}}}",
        v.iter().map(|v| number(*v)).collect::<Vec<_>>().join(",")
    )
}
fn rows(x: &[Vec<f64>]) -> String {
    format!(
        "{{{}}}",
        x.iter().map(|r| vector(r)).collect::<Vec<_>>().join(",")
    )
}

pub fn predictions(
    root: &Path,
    bundle: &Emitted,
    x: &[Vec<f64>],
    y: &[f64],
    test: &[Vec<f64>],
) -> Vec<u64> {
    build(root, bundle);
    let mode = std::env::var("JOPE_CUDA_PARITY").unwrap_or_else(|_| "host".into());
    let manifest: serde_json::Value =
        serde_json::from_str(&bundle.files["cuda/manifest.json"]).unwrap();
    let prefix = manifest["native_prefix"].as_str().unwrap();
    let source = format!(
        r#"
#include "model.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
extern "C" {{
int {prefix}fit(const double*,std::size_t,const double*,std::size_t,std::size_t,std::size_t,std::uint64_t*);
int {prefix}free(std::uint64_t);
int {prefix}linear_width(std::uint64_t,std::size_t*);
int {prefix}linear(std::uint64_t,const double*,std::size_t,std::size_t,std::size_t,double*,std::size_t,double*,std::size_t);
}}
static double d(std::uint64_t b){{double v;std::memcpy(&v,&b,8);return v;}}
static std::uint64_t bits(double v){{std::uint64_t b;std::memcpy(&b,&v,8);return b;}}
int main(){{
    jope_cuda::Rows train={train},test={test};std::vector<double> y={targets};
    jope_cuda::Model m;m.fit(train,y);auto a=m.predict(test);
    assert(a.size()==test.size());auto repeat=m.predict(test);assert(repeat.size()==a.size());
    for(std::size_t j=0;j<a.size();++j)assert(bits(a[j])==bits(repeat[j]));
    const bool supported=m.supports_cuda();
    if(supported && !test.empty()){{
        std::vector<double> tx,px;for(const auto& row:train)tx.insert(tx.end(),row.begin(),row.end());
        for(const auto& row:test)px.insert(px.end(),row.begin(),row.end());
        const auto width=train.front().size();std::uint64_t h=0;std::size_t p=0;
        assert({prefix}fit(tx.data(),tx.size(),y.data(),y.size(),train.size(),width,&h)==0);
        assert({prefix}linear_width(h,&p)==0);
        std::vector<double> features(test.size()*p),w(p+1);
        assert({prefix}linear(h,px.data(),px.size(),test.size(),width,features.data(),features.size(),w.data(),w.size())==0);
        for(std::size_t row=0;row<test.size();++row){{
            double z=w[0];for(std::size_t j=0;j<p;++j)z+=w[j+1]*features[row*p+j];
            z=z < -60.0 ? -60.0 : (z > 60.0 ? 60.0 : z);assert(bits(z)==bits(a[row]));
        }}
        assert({prefix}free(h)==0);
    }}
    if({device} && supported){{
        assert(jope_cuda::Model::cuda_available());
        auto b=m.predict_cuda(test);assert(a.size()==b.size());
        for(std::size_t j=0;j<a.size();++j)assert(bits(a[j])==bits(b[j]));
        assert(m.predict_cuda({{}}).empty());
        std::cerr<<"CUDA_LINEAR_DEVICE_PASS rows="<<a.size()<<'\n';
    }}else if(!supported || {host}){{
        bool refused=false;try{{m.predict_cuda(test);}}catch(const jope_cuda::Error& e){{refused=e.status==(supported?6:5);}}assert(refused);
    }}
    for(double v:a)std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<bits(v)<<'\n';
}}
"#,
        train = rows(x),
        test = rows(test),
        targets = vector(y),
        device = mode == "device",
        host = mode == "host"
    );
    consumer(&root.join("cuda"), &source)
        .lines()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect()
}

fn string(value: &str) -> String {
    format!(
        "std::string(\"{}\",{})",
        value
            .as_bytes()
            .iter()
            .map(|b| format!("\\{b:03o}"))
            .collect::<String>(),
        value.len()
    )
}
pub fn bound_predictions(root: &Path, bundle: &Emitted, rows: &[Vec<String>]) -> Vec<u64> {
    build(root, bundle);
    let literals = rows
        .iter()
        .map(|r| {
            format!(
                "{{{}}}",
                r.iter().map(|s| string(s)).collect::<Vec<_>>().join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let source = format!(
        r#"
#include "model.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
int main(){{
    auto m=jope_cuda::Model::load();jope_cuda::StringRows rows={{{literals}}};
    auto a=m.predict_strings(rows);assert(a.size()==rows.size());
    auto b=m.predict_strings(rows);assert(b.size()==a.size());
    assert(a.empty() || std::memcmp(a.data(),b.data(),a.size()*8)==0);
    assert(m.predict_strings({{}}).empty());assert(!m.supports_cuda());
    bool bad=false;try{{m.predict_csv("wrong header\n",0);}}catch(const jope_cuda::Error& e){{bad=e.status==1;}}assert(bad);
    bad=false;try{{m.predict_csv(std::string("\377",1),0);}}catch(const jope_cuda::Error& e){{bad=e.status==1;}}assert(bad);
    bad=false;try{{m.predict_strings({{{{}}}});}}catch(const jope_cuda::Error& e){{bad=e.status==1;}}assert(bad);
    bool refused=false;try{{m.fit({{{{1}}}},{{1}});}}catch(const jope_cuda::Error& e){{refused=e.status==5;}}assert(refused);
    for(double v:a){{std::uint64_t n;std::memcpy(&n,&v,8);std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<n<<'\n';}}
}}
"#
    );
    consumer(&root.join("cuda"), &source)
        .lines()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect()
}

/// Same registry fixtures and comparison denominator as the TypeScript target.
pub fn run(
    root: &Path,
    bundle: &Emitted,
    input: &serde_json::Value,
    expected: &[f64],
    fitted: bool,
) {
    if !enabled() {
        return;
    }
    let got = if fitted {
        let rows: Vec<Vec<String>> = serde_json::from_value(input["rows"].clone()).unwrap();
        bound_predictions(root, bundle, &rows)
    } else {
        let x: Vec<Vec<f64>> = serde_json::from_value(input["train"].clone()).unwrap();
        let y: Vec<f64> = serde_json::from_value(input["targets"].clone()).unwrap();
        let test: Vec<Vec<f64>> = serde_json::from_value(input["test"].clone()).unwrap();
        predictions(root, bundle, &x, &y, &test)
    };
    assert_eq!(
        got,
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "CUDA C++ bits at {}",
        root.display()
    );
}
