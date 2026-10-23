//! Executable language-target qualification. Device qualification is a distinct mode.
#[path = "support/cuda.rs"]
#[allow(dead_code)]
mod cuda;
use feat_math::{
    artifact_manifest::{Target, TargetState},
    cuda as emitter,
    transpile::{self, EmitOptions},
    GenomeIr,
};
use std::{fs, path::PathBuf};

fn root(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("cuda-{name}"));
    fs::create_dir_all(&root).unwrap();
    root
}
fn fixture() -> GenomeIr {
    GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap()
}

#[test]
fn complete_bundle_is_deterministic_binds_every_input_and_keeps_original_rust() {
    let g = fixture();
    let opts = EmitOptions::default();
    let a = transpile::emit_cuda(&g, &opts).unwrap();
    let b = transpile::emit_cuda(&g, &opts).unwrap();
    assert_eq!(a.files, b.files);
    assert_eq!(a.files.len(), 6);
    let pair = transpile::emit(&g, &opts).unwrap();
    assert!(a.files["cuda/model_core.rs"].starts_with(&pair.rust));
    assert!(a.files["cuda/model_core.rs"].contains("#![forbid(unsafe_code)]"));
    let m: serde_json::Value = serde_json::from_str(&a.files["cuda/manifest.json"]).unwrap();
    assert_eq!(m["device"], "linear-head-f64-v1");
    assert_eq!(m["execution"], "unknown");
    for f in m["files"].as_array().unwrap() {
        let s = &a.files[f["path"].as_str().unwrap()];
        assert_eq!(
            f["sha256"],
            feat_math::artifact_manifest::sha256(s.as_bytes())
        );
        assert_eq!(f["bytes"], s.len());
    }
    assert!(transpile::supports_cuda(&g).is_ok());
    assert!(transpile::emit_cuda(
        &g,
        &EmitOptions {
            math_kernel: transpile::MathKernel::Portable,
            ..opts
        }
    )
    .is_err());
}

#[test]
fn v3_closure_aliases_and_build_flags_are_authenticated() {
    let g = fixture();
    assert_eq!(Target::parse("cuda-cpp").unwrap(), Target::Cuda);
    assert_eq!(Target::parse("cu").unwrap(), Target::Cuda);
    assert!(Target::parse_list("cuda,cu").is_err());
    let bundle =
        transpile::emit_targets_v3(&g, &EmitOptions::default(), &[Target::Cuda, Target::Rust])
            .unwrap();
    assert!(bundle.manifest.all_supported());
    let record = bundle
        .manifest
        .targets
        .iter()
        .find(|r| r.target == Target::Cuda)
        .unwrap();
    let TargetState::Supported { files, build, .. } = &record.state else {
        panic!("missing CUDA target")
    };
    assert_eq!(files.len(), 6);
    assert_eq!(build.abi, "cuda-native-v1");
    for flag in emitter::DEVICE_FLAGS {
        assert!(build.flags.contains(&flag.to_string()));
    }
    for file in files {
        let mut manifest = bundle.manifest.clone();
        if let TargetState::Supported { files, .. } = &mut manifest
            .targets
            .iter_mut()
            .find(|r| r.target == Target::Cuda)
            .unwrap()
            .state
        {
            files.retain(|f| f.path != file.path);
        }
        assert!(manifest.validate().is_err(), "missing {}", file.path);
    }
    let mut manifest = bundle.manifest.clone();
    if let TargetState::Supported { build, .. } = &mut manifest
        .targets
        .iter_mut()
        .find(|r| r.target == Target::Cuda)
        .unwrap()
        .state
    {
        build.flags.retain(|s| s != "--fmad=false");
    }
    assert!(manifest.validate().is_err());
    use feat_math::artifact_manifest::{Domain, ExpectedArtifactV3};
    let parent = root("v3-tamper");
    let folder = tempfile::tempdir_in(parent).unwrap();
    for (path, source) in &bundle.sources {
        let path = folder.path().join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, source).unwrap();
    }
    let bytes = bundle.manifest.envelope_bytes(Domain::Native).unwrap();
    let expected = ExpectedArtifactV3::from_emission(&bytes, &bundle.manifest);
    fs::write(folder.path().join("manifest.json"), bytes).unwrap();
    feat_math::artifact_manifest::verify_directory(folder.path(), "manifest.json", &expected)
        .unwrap();
    for (path, source) in &bundle.sources {
        fs::write(folder.path().join(path), "tampered compilation input").unwrap();
        assert!(
            feat_math::artifact_manifest::verify_directory(
                folder.path(),
                "manifest.json",
                &expected
            )
            .is_err(),
            "{path}"
        );
        fs::write(folder.path().join(path), source).unwrap();
    }
}

#[test]
fn native_lifecycle_shapes_refit_concurrency_and_explicit_device_errors() {
    let root = root("contracts");
    cuda::build(
        &root,
        &emitter::emit(&fixture(), &EmitOptions::default()).unwrap(),
    );
    let output = cuda::consumer(
        &root.join("cuda"),
        include_str!("support/cuda_contract.cpp"),
    );
    assert_eq!(output.trim(), "CUDA C++ lifecycle passed");
}

#[test]
fn nonlinear_symbolic_target_and_linear_recipes_match_reference_bits() {
    let x: Vec<_> = (0..24)
        .map(|i| vec![i as f64 / 7.0 - 1.0, (i * i % 17) as f64 / 9.0])
        .collect();
    let y: Vec<_> = x.iter().map(|r| 1.4 * r[0] + 0.2 * r[1]).collect();
    for (i, text) in [
        "(pipe X (ridge))",
        "(pipe X (ridge_gcv))",
        "(pipe X (elasticnet))",
        "(pipe X (histgbm depth=2 rounds=3))",
        "(sym affine \"c0+c1\")",
        "(pipe X (target kind=zscore) (standardize) (ridge))",
    ]
    .iter()
    .enumerate()
    {
        let g = GenomeIr::parse(text).unwrap();
        let expected = feat_math::fit_predict(&g, &x, &y, &x).unwrap();
        let got = cuda::predictions(
            &root(&format!("recipe-{i}")),
            &emitter::emit(&g, &EmitOptions::default()).unwrap(),
            &x,
            &y,
            &x,
        );
        assert_eq!(
            got,
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{text}"
        );
    }
}

#[test]
fn caller_metadata_is_data_and_fast_profile_is_preserved() {
    let opts = EmitOptions {
        title: Some("CUDA\r\ncompile_error!(\"bad\");\u{2028}end".into()),
        fast_runtime: true,
        guard_zero_width_roll: true,
        partition_point_quantile_bounds: true,
        ..EmitOptions::default()
    };
    let g = fixture();
    let bundle = emitter::emit(&g, &opts).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&bundle.files["cuda/manifest.json"]).unwrap();
    assert_eq!(manifest["options"]["title"], opts.title.as_deref().unwrap());
    assert_eq!(manifest["options"]["fast_runtime"], true);
    let x = vec![vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0]];
    let y = vec![1.0, 2.0, 3.0];
    let expected = feat_math::fit_predict(&g, &x, &y, &x).unwrap();
    assert_eq!(
        cuda::predictions(&root("metadata"), &bundle, &x, &y, &x),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn native_extra_boundaries_preserve_outputs_and_refuse_aliases_overflows_and_stale_handles() {
    let bundle = emitter::emit(&fixture(), &EmitOptions::default()).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&bundle.files["cuda/manifest.json"]).unwrap();
    let prefix = manifest["native_prefix"].as_str().unwrap();
    let root = root("native-boundary");
    cuda::build(&root, &bundle);
    let source = include_str!("support/cuda_native_contract.cpp").replace("__PREFIX__", prefix);
    assert_eq!(
        cuda::consumer(&root.join("cuda"), &source).trim(),
        "CUDA native boundary passed"
    );
}

#[test]
fn independently_emitted_models_link_together_with_isolated_symbols() {
    let a = emitter::emit(&fixture(), &EmitOptions::default()).unwrap();
    let b = emitter::emit(
        &GenomeIr::parse("(pipe X (elasticnet))").unwrap(),
        &EmitOptions::default(),
    )
    .unwrap();
    let ma: serde_json::Value = serde_json::from_str(&a.files["cuda/manifest.json"]).unwrap();
    let mb: serde_json::Value = serde_json::from_str(&b.files["cuda/manifest.json"]).unwrap();
    let na = ma["cpp_namespace"].as_str().unwrap();
    let nb = mb["cpp_namespace"].as_str().unwrap();
    assert_ne!(na, nb);
    let root = root("two-libraries");
    cuda::build(&root.join("a"), &a);
    cuda::build(&root.join("b"), &b);
    for name in ["a", "b"] {
        cuda::command(
            &root.join(name).join("cuda"),
            "g++",
            &[
                "-std=c++17",
                "-x",
                "c++",
                "-DJOPE_CUDA_HOST_ONLY",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-c",
                "model.cu",
                "-o",
                "model.o",
            ],
        );
    }
    fs::write(
        root.join("both.cpp"),
        format!(
            r#"
#define JOPE_CUDA_NO_DEFAULT_NAMESPACE
#include "a/cuda/model.hpp"
#include "b/cuda/model.hpp"
#include <cassert>
int main(){{
 {na}::Model a;{nb}::Model b;
 a.fit({{{{0,1}},{{1,2}},{{2,3}}}},{{0,1,2}});
 b.fit({{{{0,1}},{{1,2}},{{2,3}}}},{{7,5,3}});
 const auto x=a.predict({{{{0.5,1.5}}}}),y=b.predict({{{{0.5,1.5}}}});
 assert(x.size()==1 && y.size()==1 && x[0]!=y[0]);a.close();
 assert(b.predict({{{{0.5,1.5}}}})==y);
}}
"#
        ),
    )
    .unwrap();
    cuda::command(
        &root,
        "g++",
        &[
            "-std=c++17",
            "both.cpp",
            "a/cuda/model.o",
            "b/cuda/model.o",
            "a/cuda/libmodel_native.a",
            "b/cuda/libmodel_native.a",
            "-ldl",
            "-lpthread",
            "-lm",
            "-o",
            "both",
        ],
    );
    cuda::command(&root, "./both", &[]);
}

#[test]
fn emitted_cmake_builds_the_declared_library_and_a_real_consumer() {
    let root = root("cmake");
    let bundle = emitter::emit(&fixture(), &EmitOptions::default()).unwrap();
    for (path, source) in &bundle.files {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, source).unwrap();
    }
    fs::write(root.join("CMakeLists.txt"),"cmake_minimum_required(VERSION 3.24)\nproject(consumer LANGUAGES CXX)\nadd_subdirectory(cuda)\nadd_executable(consumer main.cpp)\ntarget_link_libraries(consumer PRIVATE model)\n").unwrap();
    fs::write(root.join("main.cpp"),"#include \"model.hpp\"\nint main(){jope_cuda::Model m;m.fit({{0},{1},{2}},{0,1,2});return m.predict({{0.5}}).size()!=1;}\n").unwrap();
    let mode = std::env::var("JOPE_CUDA_PARITY").unwrap_or_else(|_| "host".into());
    let mut args = vec![
        "-S".to_string(),
        ".".into(),
        "-B".into(),
        "build".into(),
        "-DCMAKE_BUILD_TYPE=Release".into(),
    ];
    if mode == "host" {
        args.push("-DJOPE_CUDA_HOST_ONLY=ON".into());
    } else {
        args.push(format!(
            "-DCMAKE_CUDA_COMPILER={}",
            std::env::var("JOPE_NVCC").unwrap_or_else(|_| "nvcc".into())
        ));
        args.push("-DCMAKE_CUDA_ARCHITECTURES=75".into());
    }
    cuda::command(
        &root,
        "cmake",
        &args.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    cuda::command(&root, "cmake", &["--build", "build", "--parallel", "2"]);
    cuda::command(&root, "./build/consumer", &[]);
}

#[test]
fn device_stream_and_batch_boundaries_are_strictly_qualified_when_requested() {
    let root = root("device-contract");
    cuda::build(
        &root,
        &emitter::emit(&fixture(), &EmitOptions::default()).unwrap(),
    );
    let device = std::env::var("JOPE_CUDA_PARITY").as_deref() == Ok("device");
    let source = format!(
        "#define JOPE_TEST_DEVICE {}\n{}",
        u8::from(device),
        include_str!("support/cuda_device_contract.cpp")
    );
    assert_eq!(
        cuda::consumer(&root.join("cuda"), &source).trim(),
        if device {
            "CUDA device contract passed"
        } else {
            "CUDA device contract not requested"
        }
    );
}

#[test]
fn fitted_raw_strings_preserve_ordered_tails_and_reject_stale_rows() {
    use feat_math::advanced_v3::{
        fit_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
    };
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".into()],
            order_by: vec!["time".into()],
        },
    };
    let training = RawTable::from_csv("group,time,value\nA,1,1\nA,2,2\nB,1,10\nB,2,20\n").unwrap();
    let test = RawTable::from_csv("group,time,value\nA,4,4\nA,3,3\nA,3,5\nB,3,30\nC,1,\n").unwrap();
    for (index, stage) in ["lag k=2", "roll k=3"].iter().enumerate() {
        let g = GenomeIr::parse(&format!(
            "(pipe X (input numeric=median categorical=ordered prior=0) ({stage}) (ridge))"
        ))
        .unwrap();
        let m = fit_table(
            &g,
            &TaskSpec::regression(Objective::R2).unwrap(),
            &schema,
            &training,
            &[1., 2., 10., 20.],
        )
        .unwrap();
        let expected = m.predict_table(&test).unwrap();
        let bundle = emitter::emit_bound(&m, &EmitOptions::default()).unwrap();
        let root = root(&format!("ordered-{index}"));
        assert_eq!(
            cuda::bound_predictions(&root, &bundle, &test.rows),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let source = r#"#include "model.hpp"
#include <cassert>
int main(){auto m=jope_cuda::Model::load();bool refused=false;
try{m.predict_strings({{"A","2","99"}});}catch(const jope_cuda::Error& e){refused=e.status==1;}assert(refused);
assert(m.predict_strings({{"A","3","3"}}).size()==1);}
"#;
        cuda::consumer(&root.join("cuda"), source);
        let v3 = feat_math::advanced_v3::BoundEmitted::emit_targets_v3(
            &m,
            &g,
            &EmitOptions::default(),
            &[Target::Cuda],
        )
        .unwrap();
        assert!(v3.manifest.all_supported());
        assert_eq!(v3.sources.len(), 6);
    }
}
