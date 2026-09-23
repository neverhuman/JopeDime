//! Private executable worker contract; no dataset discovery or unbounded launcher.
use super::*;
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Component,
};

pub const WORKER_VERSION: &str = "jope-context-worker/1";
pub const COMPONENT_BUNDLE_MANIFEST_SHA256: &str =
    "5e7884ec1c0e0b955a2381d5b69c54beddad9ec66a1980c94ad279d974486d05";
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerMode {
    Produce,
    QualifyCase,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerInvocation {
    pub mode: WorkerMode,
    pub input: AssetFile,
    pub assets: AssetFile,
    pub source_sha40: String,
    pub producer_sha256: String,
    pub output: PathBuf,
}
impl WorkerInvocation {
    /// Stable argv used by the shared pool; no shell interpolation.
    pub fn arguments(&self) -> Vec<String> {
        vec![
            "--mode".into(),
            match self.mode {
                WorkerMode::Produce => "produce",
                WorkerMode::QualifyCase => "qualify-case",
            }
            .into(),
            "--input".into(),
            self.input.path.to_string_lossy().into(),
            "--input-sha256".into(),
            self.input.sha256.clone(),
            "--assets".into(),
            self.assets.path.to_string_lossy().into(),
            "--assets-sha256".into(),
            self.assets.sha256.clone(),
            "--source-sha40".into(),
            self.source_sha40.clone(),
            "--producer-sha256".into(),
            self.producer_sha256.clone(),
            "--output".into(),
            self.output.to_string_lossy().into(),
        ]
    }
    pub fn parse(arguments: &[String]) -> Result<Self> {
        ensure!(
            arguments.len() == 16,
            "worker requires exactly8 named arguments"
        );
        let mut args = BTreeMap::new();
        for pair in arguments.chunks_exact(2) {
            ensure!(
                args.insert(pair[0].as_str(), pair[1].as_str()).is_none(),
                "duplicate worker argument"
            );
        }
        let field = |key| {
            args.get(key)
                .copied()
                .with_context(|| format!("missing worker argument {key}"))
        };
        let mode = match field("--mode")? {
            "produce" => WorkerMode::Produce,
            "qualify-case" => WorkerMode::QualifyCase,
            _ => anyhow::bail!("unsupported context worker mode"),
        };
        let result = Self {
            mode,
            input: AssetFile {
                path: field("--input")?.into(),
                sha256: field("--input-sha256")?.into(),
            },
            assets: AssetFile {
                path: field("--assets")?.into(),
                sha256: field("--assets-sha256")?.into(),
            },
            source_sha40: field("--source-sha40")?.into(),
            producer_sha256: field("--producer-sha256")?.into(),
            output: field("--output")?.into(),
        };
        validate_hex(&result.source_sha40, 40)?;
        validate_hex(&result.producer_sha256, 64)?;
        Ok(result)
    }
}

/// File-only closure inventory. Parent verifies every model/notice before
/// scheduling; the governed component loader verifies them again in the child.
/// This does not load tensors or invoke an encoder.
pub fn asset_inventory(files: &EncoderFiles) -> Result<Vec<AssetFile>> {
    let root = &files.component_bundle;
    ensure!(
        root.is_absolute() && root.canonicalize()? == *root,
        "component bundle root must be physical"
    );
    let manifest_path = [
        root.join("model-bundle.v1.json"),
        root.join("artifacts/model-bundle.v1.json"),
    ]
    .into_iter()
    .find(|p| p.exists())
    .context("component bundle manifest absent")?;
    let manifest = AssetFile {
        path: manifest_path,
        sha256: COMPONENT_BUNDLE_MANIFEST_SHA256.into(),
    };
    let bytes = manifest.read(1 << 20)?;
    let model: starforge::bundle::ModelBundleV1 = serde_json::from_slice(&bytes)?;
    model.validate_metadata()?;
    let mut inventory = vec![
        files.hyperion_weights.clone(),
        files.hyperion_manifest.clone(),
        files.frozen_transforms.clone(),
        manifest,
    ];
    for file in &model.files {
        inventory.push(AssetFile {
            path: relative(root, &file.path)?,
            sha256: file.sha256.clone(),
        });
    }
    for (path, digest) in &model.notices {
        inventory.push(AssetFile {
            path: relative(root, path)?,
            sha256: digest.clone(),
        });
    }
    let mut seen = BTreeMap::new();
    for file in inventory {
        verify_file(&file, 512 << 20)?;
        if let Some(old) = seen.insert(file.path.clone(), file.sha256.clone()) {
            ensure!(old == file.sha256, "conflicting model asset identity");
        }
    }
    Ok(seen
        .into_iter()
        .map(|(path, sha256)| AssetFile { path, sha256 })
        .collect())
}
fn relative(root: &std::path::Path, path: &str) -> Result<PathBuf> {
    let p = std::path::Path::new(path);
    ensure!(
        !path.is_empty() && p.components().all(|v| matches!(v, Component::Normal(_))),
        "non-relative model asset"
    );
    Ok(root.join(p))
}
/// Streaming full-file validation; never prints secret or model contents.
pub fn verify_file(file: &AssetFile, maximum: u64) -> Result<()> {
    use sha2::{Digest, Sha256};
    ensure!(
        file.path.is_absolute() && file.path.canonicalize()? == file.path,
        "artifact must be physical and absolute"
    );
    let meta = fs::symlink_metadata(&file.path)?;
    ensure!(
        meta.is_file() && meta.len() <= maximum,
        "artifact is not bounded regular file"
    );
    let mut handle = fs::File::open(&file.path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut bytes = 0u64;
    loop {
        let n = handle.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        ensure!(bytes <= maximum, "artifact grew beyond bound");
        hash.update(&buffer[..n]);
    }
    ensure!(
        format!("{:x}", hash.finalize()) == file.sha256 && bytes == meta.len(),
        "artifact identity changed"
    );
    Ok(())
}
pub fn execute(invocation: &WorkerInvocation) -> Result<()> {
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        ensure!(
            std::env::var(key).as_deref() == Ok("1"),
            "worker requires pool numerical thread limit: {key}"
        );
    }
    ensure!(
        std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("-1"),
        "CPU worker requires explicit GPU exclusion"
    );
    let binary = std::env::current_exe()?.canonicalize()?;
    verify_file(
        &AssetFile {
            path: binary,
            sha256: invocation.producer_sha256.clone(),
        },
        512 << 20,
    )?;
    let input = invocation.input.read(128 << 20)?;
    let files: EncoderFiles = serde_json::from_slice(&invocation.assets.read(1 << 20)?)?;
    // Full closure is pinned by the parent and revalidated before any model load.
    asset_inventory(&files)?;
    let load_started = std::time::Instant::now();
    let producer = CpuProducer::load(&files)?;
    eprintln!(
        "context stage=model_load elapsed_seconds={:.6}",
        load_started.elapsed().as_secs_f64()
    );
    let bytes = match invocation.mode {
        WorkerMode::Produce => serde_json::to_vec(&producer.produce(
            &input,
            &invocation.source_sha40,
            &invocation.producer_sha256,
        )?)?,
        WorkerMode::QualifyCase => {
            let result = producer.produce_reference_case(
                &input,
                &invocation.input.sha256,
                &invocation.source_sha40,
                &invocation.producer_sha256,
            )?;
            if let Err(error) = admission::validate_reference_case(&input, &result) {
                // Preserve actual failing values for diagnosis without creating
                // the requested successful output or changing the exit status.
                write_new(
                    &invocation.output.with_extension("unqualified.json"),
                    &serde_json::to_vec(&result)?,
                )?;
                return Err(error);
            }
            serde_json::to_vec(&result)?
        }
    };
    write_new(&invocation.output, &bytes)?;
    println!("{} {}", WORKER_VERSION, sha(&bytes));
    Ok(())
}
/// Immutable private output; parent must create the physical job directory.
pub fn write_new(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("output has no parent")?;
    ensure!(
        path.is_absolute() && parent.canonicalize()? == parent && path.file_name().is_some(),
        "worker output must be absolute under physical parent"
    );
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut output = options.open(path)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invocation_roundtrip_and_extra_paths_refused() {
        let i = WorkerInvocation {
            mode: WorkerMode::Produce,
            input: AssetFile {
                path: "/private/input.json".into(),
                sha256: "a".repeat(64),
            },
            assets: AssetFile {
                path: "/private/assets.json".into(),
                sha256: "b".repeat(64),
            },
            source_sha40: "c".repeat(40),
            producer_sha256: "d".repeat(64),
            output: "/private/job/output.json".into(),
        };
        let mut args = i.arguments();
        assert_eq!(WorkerInvocation::parse(&args).unwrap().arguments(), args);
        args.extend(["--dataset-root".into(), "/private/corpus".into()]);
        assert!(WorkerInvocation::parse(&args).is_err());
    }
    #[test]
    fn bundle_relative_paths_refuse_escape() {
        assert!(relative(std::path::Path::new("/models"), "../data.csv").is_err());
        assert!(relative(std::path::Path::new("/models"), "/data.csv").is_err());
    }
}
