use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifestV3 {
    pub schema: String,
    pub dsl: String,
    pub genome_sha256: String,
    pub dialect: String,
    pub registry_sha256: String,
    pub options: OptionsIdentity,
    pub requested: Vec<Target>,
    pub targets: Vec<TargetRecord>,
}
impl ArtifactManifestV3 {
    pub fn new(
        program: &str,
        dialect: &str,
        options: &EmitOptions,
        requested: &[Target],
        targets: Vec<TargetRecord>,
    ) -> Result<Self> {
        let m = Self {
            schema: SCHEMA.into(),
            dsl: program.into(),
            genome_sha256: sha256(program.as_bytes()),
            dialect: dialect.into(),
            registry_sha256: crate::kernel::registry_v2::source_sha256(),
            options: options.into(),
            requested: requested.to_vec(),
            targets,
        };
        m.validate()?;
        Ok(m)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == SCHEMA, "unsupported artifact schema");
        ensure!(
            ["legacy_recipe", "advanced_v3"].contains(&self.dialect.as_str()),
            "unknown program dialect"
        );
        ensure!(
            self.genome_sha256 == sha256(self.dsl.as_bytes()),
            "selected program identity mismatch"
        );
        ensure!(
            crate::GenomeIr::parse(&self.dsl)?.canonical() == self.dsl,
            "noncanonical selected program"
        );
        check_digest(&self.registry_sha256)?;
        ensure!(
            self.options.math_policy == "platform-libm-v1",
            "unsupported math policy"
        );
        validate_requests(&self.requested)?;
        ensure!(
            self.targets.len() == self.requested.len(),
            "requested targets are incomplete"
        );
        let mut seen = BTreeSet::new();
        let mut paths: BTreeMap<&str, (Target, &FileBinding)> = BTreeMap::new();
        for record in &self.targets {
            ensure!(
                self.requested.contains(&record.target) && seen.insert(record.target),
                "unexpected or duplicate target record"
            );
            match &record.state {
                TargetState::Unsupported {
                    reason,
                    registry_sha256,
                } => {
                    ensure!(
                        !reason.trim().is_empty() && registry_sha256 == &self.registry_sha256,
                        "unsupported target lacks registry evidence"
                    );
                }
                TargetState::Supported {
                    product,
                    emitter,
                    files,
                    build,
                    fitted,
                } => {
                    ensure!(
                        product == record.target.product(&self.dialect)
                            && !emitter.trim().is_empty(),
                        "target product/emitter mismatch"
                    );
                    let expected: Vec<_> = if record.target == Target::Cpp17Fitted {
                        vec![
                            record.target.path(),
                            "cpp17-fitted/model.codec.json",
                            "cpp17-fitted/standalone.cpp",
                        ]
                    } else if record.target == Target::Cuda {
                        vec![
                            "cuda/CMakeLists.txt",
                            "cuda/manifest.json",
                            "cuda/model.cu",
                            "cuda/model.hpp",
                            "cuda/model_core.rs",
                            "cuda/model_native.rs",
                        ]
                    } else if record.target == Target::C17 {
                        vec![
                            "c17/manifest.json",
                            "c17/model.h",
                            "c17/model_core.rs",
                            "c17/model_ffi.rs",
                        ]
                    } else if record.target == Target::TypeScript {
                        vec!["model.ts", "model.kernel.rs"]
                    } else if record.target == Target::Go {
                        vec!["model.go", "model.go.kernel.rs", "go.mod", "go.sum"]
                    } else {
                        vec![record.target.path()]
                    };
                    ensure!(
                        files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>() == expected,
                        "incomplete or ambiguous artifact closure"
                    );
                    let declared = BuildIdentity::declared(record.target);
                    // Compiler selection/flags belong to the pinned recipe, not the consumer's environment.
                    ensure!(
                        build.standard == declared.standard
                            && build.backend == declared.backend
                            && build.abi == declared.abi
                            && !build.toolchain.is_empty(),
                        "target build identity mismatch"
                    );
                    if matches!(record.target, Target::C17 | Target::Cuda) {
                        ensure!(build == &declared, "native build recipe mismatch");
                    } else if matches!(record.target, Target::Go | Target::TypeScript) {
                        ensure!(
                            build.toolchain == declared.toolchain && build.flags == declared.flags,
                            "Wasm target build recipe mismatch"
                        );
                    }
                    if matches!(
                        record.target,
                        Target::Cpp17 | Target::Cpp20 | Target::Cpp17RecipeV2 | Target::Cpp17Fitted
                    ) {
                        ensure!(
                            build.flags.contains(&format!("-std={}", build.standard))
                                && build.flags.contains(&"-ffp-contract=off".into())
                                && build.flags.contains(&"-fno-fast-math".into()),
                            "unsafe or mismatched build flags"
                        );
                    }
                    ensure!(
                        fitted.is_some()
                            == (record.target == Target::Cpp17Fitted
                                || self.dialect == "advanced_v3"),
                        "fitted state identity missing or unexpected"
                    );
                    if let Some(f) = fitted {
                        for d in [
                            &f.state_sha256,
                            &f.input_sha256,
                            &f.task_sha256,
                            &f.schema_sha256,
                            &f.vocabulary_sha256,
                        ] {
                            check_digest(d)?;
                        }
                    }
                    for file in files {
                        safe_relative(&file.path)?;
                        check_digest(&file.sha256)?;
                        ensure!(
                            file.bytes > 0 && file.bytes <= 9_007_199_254_740_991,
                            "invalid artifact length"
                        );
                        if let Some((other, previous)) =
                            paths.insert(&file.path, (record.target, file))
                        {
                            ensure!(
                                matches!(
                                    (other, record.target),
                                    (Target::Cpp17, Target::Cpp20) | (Target::Cpp20, Target::Cpp17)
                                ) && previous == file,
                                "ambiguous shared artifact path"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub fn all_supported(&self) -> bool {
        self.validate().is_ok()
            && self
                .targets
                .iter()
                .all(|t| matches!(t.state, TargetState::Supported { .. }))
    }
    pub fn envelope_bytes(&self, domain: ArtifactDomain) -> Result<Vec<u8>> {
        self.validate()?;
        let payload = canonical(self)?;
        canonical(&ArtifactEnvelope {
            envelope_schema: ENVELOPE.into(),
            domain,
            schema_id: SCHEMA.into(),
            own_id: OwnId {
                formula: FORMULA.into(),
                schema_id: SCHEMA.into(),
                payload_byte_len: payload.len() as u64,
                sha256: domain_hash(&payload),
            },
            payload: self.clone(),
        })
    }
}
