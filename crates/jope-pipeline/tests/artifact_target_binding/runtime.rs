use super::*;

#[test]
fn artifact_runtime_fixture_executes_every_supported_product() {
    use std::process::Command;
    fn execute(command: &mut Command) {
        println!("executed: {command:?}");
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let dir = scratch();
    let root = dir.path();
    let mut targets = Target::RECIPES.to_vec();
    targets.extend([Target::Cpp17RecipeV2, Target::Cpp17Fitted, Target::C17]);
    if matches!(
        std::env::var("JOPE_CUDA_PARITY").as_deref(),
        Ok("host" | "compile" | "device")
    ) {
        targets.push(Target::Cuda);
    }
    if typescript_support::enabled() {
        targets.push(Target::TypeScript);
    }
    if go_support::enabled() {
        targets.push(Target::Go);
    }
    let program = "(pipe X (ridge alpha=1.0))";
    let out = emit::emit_v3(program, &targets, root, Some(&rows())).unwrap();
    assert!(out.manifest.all_supported());
    emit::read_identity_v3(root, &out.expected).unwrap();
    let training = root.join("train.csv");
    let probes = root.join("probes.csv");
    fs::write(
        &training,
        (0..10)
            .map(|i| format!("{i},{}\n", 2 * i + 1))
            .collect::<String>(),
    )
    .unwrap();
    fs::write(&probes, "0.5\n2.5\n").unwrap();
    let train = rows();
    let x: Vec<_> = train.iter().map(|r| vec![r[0]]).collect();
    let y: Vec<_> = train.iter().map(|r| r[1]).collect();
    let expected = feat_math::fit(&feat_math::GenomeIr::parse(program).unwrap(), &x, &y)
        .unwrap()
        .predict(&vec![vec![0.5], vec![2.5]])
        .unwrap();
    let bits = |v: &[f64]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for target in targets {
        let predictions = root.join(format!("{}.predictions", target.as_str()));
        match target {
            Target::Cuda => {
                let cdir = root.join("cuda");
                execute(Command::new("rustc").current_dir(&cdir).args([
                    "--edition=2021",
                    "-O",
                    "-C",
                    "panic=unwind",
                    "--crate-type=staticlib",
                    "model_native.rs",
                    "-o",
                    "libmodel_native.a",
                ]));
                fs::write(
                    cdir.join("consumer.cpp"),
                    r#"#include "model.hpp"
#include <fstream>
#include <iomanip>
int main(int argc,char** argv){
 if(argc!=2)return 1;
 jope_cuda::Rows x;std::vector<double> y;
 for(int i=0;i<10;++i){x.push_back({static_cast<double>(i)});y.push_back(2*i+1);}
 jope_cuda::Model m;m.fit(x,y);auto out=m.predict({{0.5},{2.5}});
 std::ofstream f(argv[1]);f<<std::setprecision(17);
 for(double v:out)f<<v<<'\n';return f?0:2;
}"#,
                )
                .unwrap();
                execute(Command::new("g++").current_dir(&cdir).args([
                    "-std=c++17",
                    "-x",
                    "c++",
                    "-O2",
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
                ]));
                execute(Command::new(cdir.join("consumer")).arg(&predictions));
            }
            Target::C17 => {
                let cdir = root.join("c17");
                execute(Command::new("rustc").current_dir(&cdir).args([
                    "--edition=2021",
                    "-O",
                    "-C",
                    "panic=unwind",
                    "--crate-type=staticlib",
                    "model_ffi.rs",
                    "-o",
                    "libmodel.a",
                ]));
                fs::write(
                    cdir.join("consumer.c"),
                    r#"#include "model.h"
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    double x[10], y[10], t[]={0.5,2.5}, out[2]; uint64_t h=0;
    for (int i=0;i<10;i++) {x[i]=i; y[i]=2*i+1;}
    if (jope_model_fit(x,10,y,10,10,1,&h)) return 2;
    if (jope_model_predict(h,t,2,2,1,out,2)) return 3;
    FILE *f=fopen(argv[1],"w"); if (!f) return 4;
    fprintf(f,"%.17g\n%.17g\n",out[0],out[1]);
    if (fclose(f)) return 5;
    return jope_model_free(h);
}
"#,
                )
                .unwrap();
                execute(Command::new("gcc").current_dir(&cdir).args([
                    "-std=c17",
                    "-pedantic-errors",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                    "consumer.c",
                    "libmodel.a",
                    "-ldl",
                    "-lpthread",
                    "-lm",
                    "-o",
                    "consumer",
                ]));
                execute(Command::new(cdir.join("consumer")).arg(&predictions));
            }
            Target::TypeScript => {
                // Execute the exact two sources admitted by read_identity_v3.
                // The shared qualifier checks actual CJS/ESM predictions with
                // the declared Wasm tolerance, rather than the native bit gate.
                typescript_support::run(
                    &root.join("verified-typescript"),
                    &fs::read_to_string(root.join("model.ts")).unwrap(),
                    &fs::read_to_string(root.join("model.kernel.rs")).unwrap(),
                    &serde_json::json!({"train":x,"targets":y,"test":[[0.5],[2.5]]}),
                    &expected,
                    false,
                );
                continue;
            }
            Target::Go => {
                // Execute the exact two sources admitted by read_identity_v3.
                // The shared qualifier checks actual CJS/ESM predictions with
                // the declared Wasm tolerance, rather than the native bit gate.
                go_support::run(
                    &root.join("verified-go"),
                    &fs::read_to_string(root.join("model.go")).unwrap(),
                    &fs::read_to_string(root.join("model.go.kernel.rs")).unwrap(),
                    &serde_json::json!({"train":x,"targets":y,"test":[[0.5],[2.5]]}),
                    &expected,
                    false,
                );
                continue;
            }
            Target::Python => execute(
                Command::new("python3")
                    .arg(root.join("model.py"))
                    .arg(&training)
                    .arg(&probes)
                    .arg(&predictions),
            ),
            Target::Rust => {
                let binary = root.join("model-rust");
                execute(
                    Command::new("rustc")
                        .args(["--edition=2021", "-O"])
                        .arg(root.join("model.rs"))
                        .arg("-o")
                        .arg(&binary),
                );
                execute(
                    Command::new(&binary)
                        .arg(&training)
                        .arg(&probes)
                        .arg(&predictions),
                );
            }
            Target::JavaScript => {
                let driver = include_str!("../../../../deploy/ci/emitters/driver.js");
                let source = fs::read_to_string(root.join(target.path())).unwrap();
                let executable = root.join("program.js");
                fs::write(&executable, format!("{source}\n{driver}")).unwrap();
                execute(
                    Command::new("node")
                        .arg(&executable)
                        .arg(&training)
                        .arg(&probes)
                        .arg(&predictions),
                );
            }
            Target::Cpp17 | Target::Cpp20 | Target::Cpp17RecipeV2 => {
                let driver = include_str!("../../../../deploy/ci/emitters/driver.cpp");
                let source = fs::read_to_string(root.join(target.path())).unwrap();
                let executable = root.join(format!("program-{}.cpp", target.as_str()));
                fs::write(&executable, format!("{source}\n{driver}")).unwrap();
                let binary = root.join(format!("model-{}", target.as_str()));
                let std = if target == Target::Cpp20 {
                    "-std=c++20"
                } else {
                    "-std=c++17"
                };
                execute(
                    Command::new(feat_math::cxx::compiler())
                        .args([std, "-O3", "-ffp-contract=off", "-fno-fast-math"])
                        .arg(&executable)
                        .arg("-o")
                        .arg(&binary),
                );
                execute(
                    Command::new(&binary)
                        .arg(&training)
                        .arg(&probes)
                        .arg(&predictions),
                );
            }
            Target::Cpp17Fitted => {
                let binary = root.join("model-fitted");
                execute(
                    Command::new(feat_math::cxx::compiler())
                        .args([
                            "-std=c++17",
                            "-O3",
                            "-ffp-contract=off",
                            "-fno-fast-math",
                            "-DJOPE_STANDALONE",
                        ])
                        .arg(root.join("cpp17-fitted/standalone.cpp"))
                        .arg("-o")
                        .arg(&binary),
                );
                let header = root.join("probes-header.csv");
                fs::write(&header, "a\n0.5\n2.5\n").unwrap();
                execute(Command::new(&binary).arg(&header).arg(&predictions));
            }
        }
        let text = fs::read_to_string(&predictions).unwrap();
        let actual: Vec<f64> = text
            .lines()
            .filter(|s| *s != "prediction")
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(bits(&actual), bits(&expected), "{target:?}");
        println!(
            "{}: predictions_sha256={}",
            target.as_str(),
            sha256(text.as_bytes())
        );
    }
}
