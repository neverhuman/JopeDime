use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionsIdentity {
    pub title: Option<String>,
    pub feature_names: Vec<String>,
    pub context: Option<String>,
    pub fast_runtime: bool,
    pub math_policy: String,
    pub guard_zero_width_roll: bool,
    pub partition_point_quantile_bounds: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hpc_v1: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hpc_v2: bool,
}
impl From<&EmitOptions> for OptionsIdentity {
    fn from(o: &EmitOptions) -> Self {
        Self {
            title: o.title.clone(),
            feature_names: o.feature_names.clone(),
            context: o.context.clone(),
            fast_runtime: o.fast_runtime,
            math_policy: o.math_kernel.policy_id().into(),
            guard_zero_width_roll: o.guard_zero_width_roll,
            partition_point_quantile_bounds: o.partition_point_quantile_bounds,
            hpc_v1: o.hpc_v1,
            hpc_v2: o.hpc_v2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildIdentity {
    pub toolchain: String,
    pub standard: String,
    pub flags: Vec<String>,
    pub backend: String,
    pub abi: String,
    pub execution: BuildExecution,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildExecution {
    Unknown,
}
impl BuildIdentity {
    pub fn declared(target: Target) -> Self {
        let (toolchain, standard, flags, abi) = match target {
            Target::Cuda => (
                "rustc+nvcc".into(),
                "rust2021+cuda-c++17".into(),
                vec![
                    "--edition=2021".into(),
                    "-O".into(),
                    "-Cpanic=unwind".into(),
                    "--crate-type=staticlib".into(),
                    "-std=c++17".into(),
                ]
                .into_iter()
                .chain(crate::cuda::DEVICE_FLAGS.iter().map(|s| s.to_string()))
                .collect(),
                "cuda-native-v1",
            ),
            Target::C17 => (
                "rustc".into(),
                "rust2021+c17".into(),
                vec![
                    "--edition=2021".into(),
                    "-O".into(),
                    "-C".into(),
                    "panic=unwind".into(),
                    "--crate-type=staticlib".into(),
                ],
                "c17-v1",
            ),
            Target::Python => ("python3".into(), "python3".into(), vec![], "script"),
            Target::Rust => (
                "rustc".into(),
                "rust2021".into(),
                vec!["-O".into()],
                "script",
            ),
            Target::JavaScript => ("node".into(), "es2020-commonjs".into(), vec![], "commonjs"),
            Target::TypeScript => (
                "rustc+tsc".into(),
                "typescript-es2020+wasm32".into(),
                vec![
                    "--edition=2021".into(),
                    "--target=wasm32-unknown-unknown".into(),
                    "--crate-type=cdylib".into(),
                    "--crate-name=model_kernel".into(),
                    "-Copt-level=3".into(),
                    "-Cpanic=abort".into(),
                    "--strict".into(),
                    "--target=ES2020".into(),
                    "--module=commonjs".into(),
                    "--lib=ES2020,DOM".into(),
                    "--declaration".into(),
                ],
                "typescript-wasm-v1",
            ),
            Target::Go => (
                "rustc+go".into(),
                "go1.25+wasm32".into(),
                vec![
                    "--edition=2021".into(),
                    "--target=wasm32-unknown-unknown".into(),
                    "--crate-type=cdylib".into(),
                    "--crate-name=model_kernel".into(),
                    "-Copt-level=3".into(),
                    "-Cpanic=abort".into(),
                    "CGO_ENABLED=0".into(),
                    "-mod=readonly".into(),
                    "wazero/v1.12.0".into(),
                ],
                "go-wasm-v1",
            ),
            _ => {
                let standard = if target == Target::Cpp20 {
                    "c++20"
                } else {
                    "c++17"
                };
                let mut flags = vec![format!("-std={standard}"), "-O3".into()];
                flags.extend(crate::cxx::march_flag(None));
                if target == Target::Cpp17Fitted {
                    flags.push("-DJOPE_STANDALONE".into());
                }
                flags.extend(["-ffp-contract=off".into(), "-fno-fast-math".into()]);
                (crate::cxx::compiler(), standard.into(), flags, "c")
            }
        };
        Self {
            toolchain,
            standard,
            flags,
            backend: if matches!(target, Target::Go | Target::TypeScript) {
                "wasm32-scalar"
            } else if target == Target::Cuda {
                "cpu-native+cuda-linear-head"
            } else {
                "cpu-scalar"
            }
            .into(),
            abi: abi.into(),
            execution: BuildExecution::Unknown,
        }
    }
}
