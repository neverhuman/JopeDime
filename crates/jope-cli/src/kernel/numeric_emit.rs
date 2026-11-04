use super::*;

pub(crate) const FITTED_STANDALONE: &str = jope_pipeline::emit::fitted::STANDALONE;

pub(crate) fn read_numeric_training_csv(path: &Path) -> Result<(feat_math::Rows, Vec<f64>)> {
    let (x, y, _) = read_numeric_training_csv_with_schema(path)?;
    Ok((x, y))
}

pub(crate) fn read_numeric_training_csv_with_schema(
    path: &Path,
) -> Result<(feat_math::Rows, Vec<f64>, Vec<String>)> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("training CSV needs a UTF-8 filename")?;
    let bytes = Directory::open(parent)
        .and_then(|directory| directory.read(filename, MAX_FILE_BYTES))
        .with_context(|| format!("read training CSV {}", path.display()))?;
    let csv = std::str::from_utf8(&bytes).context("training CSV is not UTF-8")?;
    let table = RawTable::from_csv(csv).context("parse training CSV")?;
    anyhow::ensure!(
        table.headers.len() >= 2 && table.headers.last().is_some_and(|name| name == "target"),
        "training CSV needs feature columns followed by target"
    );
    anyhow::ensure!(!table.rows.is_empty(), "training CSV has no rows");
    let width = table.headers.len() - 1;
    let mut x = Vec::with_capacity(table.rows.len());
    let mut y = Vec::with_capacity(table.rows.len());
    for (index, row) in table.rows.iter().enumerate() {
        let numbers = row
            .iter()
            .map(|cell| cell.parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("non-numeric cell on training row {}", index + 1))?;
        anyhow::ensure!(
            numbers.iter().all(|value| value.is_finite()),
            "non-finite value on training row {}",
            index + 1
        );
        x.push(numbers[..width].to_vec());
        y.push(numbers[width]);
    }
    Ok((x, y, table.headers[..width].to_vec()))
}

/// Emit-only v3 adapter, selected with the existing `--lang v3:<targets>` string.
/// Existing language aliases, report schema and default emission stay compatible.
pub fn emit_genome_v3(
    genome: &str,
    lang: &str,
    train: Option<&Path>,
    output: &Path,
) -> Result<jope_pipeline::emit::EmittedV3> {
    use feat_math::artifact_manifest::Target;
    let requested = Target::parse_list(lang)?;
    anyhow::ensure!(
        train.is_none() || requested.contains(&Target::Cpp17Fitted),
        "--train requires cpp17-fitted"
    );
    let rows = if let Some(path) = train {
        let (x, y) = read_numeric_training_csv(path)?;
        Some(
            x.into_iter()
                .zip(y)
                .map(|(mut row, target)| {
                    row.push(target);
                    row
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    jope_pipeline::emit::emit_v3(genome, &requested, output, rows.as_deref())
}

#[cfg(test)]
mod artifact_v3_tests {
    use super::*;
    #[test]
    fn emit_adapter_preserves_default_route_and_binds_opt_in_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("legacy");
        emit_genome(DEFAULT_GENOME, "rust", Some(&legacy)).unwrap();
        assert!(!legacy.join("manifest.json").exists());
        let routed = dir.path().join("routed");
        let report = emit_genome(DEFAULT_GENOME, "v3:rs,js,cpp20", Some(&routed)).unwrap();
        assert_eq!(report.schema, "jope.emit.v3");
        assert!(report.ok);
        assert!(report.artifacts.iter().any(|a| a.lang == "manifest-v3"));
        for language in ["ts", "typescript", "v3:typescript"] {
            let output = dir.path().join(language.replace(':', "-"));
            let report = emit_genome(DEFAULT_GENOME, language, Some(&output)).unwrap();
            assert!(report.ok);
            assert!(output.join("model.ts").is_file());
            assert!(output.join("model.kernel.rs").is_file());
            assert!(report.artifacts.iter().any(|a| a.lang == "typescript"));
        }
        for language in ["golang", "go", "v3:go"] {
            let output = dir.path().join(language.replace(':', "-"));
            let report = emit_genome(DEFAULT_GENOME, language, Some(&output)).unwrap();
            assert!(report.ok);
            assert!(output.join("model.go").is_file());
            assert!(output.join("model.go.kernel.rs").is_file());
            assert!(report.artifacts.iter().any(|a| a.lang == "go"));
        }
        if matches!(
            std::env::var("JOPE_CUDA_PARITY").as_deref(),
            Ok("host" | "compile" | "device")
        ) {
            for language in ["cuda", "cu", "cuda-cpp", "v3:cuda"] {
                let output = dir.path().join(language.replace(':', "-"));
                let report = emit_genome(DEFAULT_GENOME, language, Some(&output)).unwrap();
                assert!(report.ok);
                for file in [
                    "CMakeLists.txt",
                    "model.cu",
                    "model.hpp",
                    "model_core.rs",
                    "model_native.rs",
                    "manifest.json",
                ] {
                    assert!(output.join("cuda").join(file).is_file());
                }
                assert!(report.artifacts.iter().any(|a| a.lang == "cuda"));
            }
        }
        let refused = dir.path().join("unsupported");
        let err = emit_genome("(pipe X (histgbm))", "v3:js", Some(&refused)).unwrap_err();
        assert!(err.to_string().contains("unsupported requested targets"));
        assert!(refused.join("manifest.json").is_file());
        let new = dir.path().join("v3");
        let out = emit_genome_v3(DEFAULT_GENOME, "rs,js,cpp20", None, &new).unwrap();
        assert!(out.manifest.all_supported());
        jope_pipeline::emit::read_identity_v3(&new, &out.expected).unwrap();
        assert!(emit_genome_v3(DEFAULT_GENOME, "rs,rust", None, &dir.path().join("bad")).is_err());
    }
    #[test]
    fn fitted_emit_adapter_binds_validated_training_rows() {
        let dir = tempfile::tempdir().unwrap();
        let train = dir.path().join("train.csv");
        fs::write(&train, "a,target\n0,1\n1,3\n2,5\n3,7\n").unwrap();
        let new = dir.path().join("v3");
        let out = emit_genome_v3(DEFAULT_GENOME, "cpp17-fitted", Some(&train), &new).unwrap();
        assert!(out.manifest.all_supported());
        jope_pipeline::emit::read_identity_v3(&new, &out.expected).unwrap();
        fs::write(new.join("cpp17-fitted/model.codec.json"), "{}").unwrap();
        assert!(jope_pipeline::emit::read_identity_v3(&new, &out.expected).is_err());
    }
}

/// Explicit fitted numeric profile route; does not change legacy emit defaults.
pub fn emit_numeric_v2(
    genome: &str,
    train: &Path,
    profile: feat_math::numeric_v2::NumericProfileV2,
    options: feat_math::numeric_v2_emit::OptionsV2,
    output: &Path,
) -> Result<feat_math::numeric_v2_emit::FittedArtifactV2> {
    let (x, y, schema) = read_numeric_training_csv_with_schema(train)?;
    anyhow::ensure!(
        profile.schema == schema,
        "numeric profile/CSV schema order mismatch"
    );
    jope_pipeline::emit::emit_numeric_v2(genome, profile, &x, &y, options, output)
}

#[cfg(test)]
mod numeric_v2_tests {
    use super::*;
    #[test]
    fn numeric_v2_kernel_export_and_option_selection() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/task-evidence/I10/cli-tests");
        fs::create_dir_all(&root).unwrap();
        let dir = tempfile::tempdir_in(root).unwrap();
        let csv = dir.path().join("train.csv");
        fs::write(&csv, "a,target\n0,-200\n1,0\n2,200\n").unwrap();
        let options = feat_math::numeric_v2_emit::OptionsV2::new(
            feat_math::numeric_v2_emit::TargetV2::Python,
        );
        let artifact = emit_numeric_v2(
            "(pipe X (ridge))",
            &csv,
            feat_math::numeric_v2::NumericProfileV2::new(vec!["a".into()]),
            options.clone(),
            dir.path(),
        )
        .unwrap();
        assert!(emit_numeric_v2(
            "(pipe X (ridge))",
            &csv,
            feat_math::numeric_v2::NumericProfileV2::new(vec!["wrong".into()]),
            options.clone(),
            dir.path()
        )
        .is_err());
        let model =
            jope_pipeline::emit::reload_numeric_v2(dir.path(), &options, &artifact.model_sha256)
                .unwrap();
        assert!(model.predict(&["a".into()], &vec![vec![2.]]).unwrap()[0] > 60.);
        let recipe = "(pipe X (quantile_bins bins=4) (ridge))";
        let old = emit_genome(recipe, "python", None).unwrap();
        let explicit = emit_genome_with_options(
            recipe,
            "python",
            None,
            None,
            &EmitOptions {
                title: Some("jope emit".into()),
                ..EmitOptions::default()
            },
        )
        .unwrap();
        assert_eq!(old.artifacts[0].sha256, explicit.artifacts[0].sha256);
        let new = emit_genome_with_options(
            recipe,
            "python",
            None,
            None,
            &EmitOptions {
                title: Some("jope emit".into()),
                partition_point_quantile_bounds: true,
                ..EmitOptions::default()
            },
        )
        .unwrap();
        assert_ne!(old.artifacts[0].sha256, new.artifacts[0].sha256);
        let cpp20 = emit_genome(recipe, "cpp20", None).unwrap();
        let parsed = GenomeIr::parse(recipe).unwrap();
        let historical = jope_pipeline::emit::cpp_source(&parsed).unwrap().unwrap();
        assert_eq!(cpp20.artifacts[0].sha256, digest(&historical));
    }
}
