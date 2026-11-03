//! Build immutable source and installed-package receipts for the experimental CPU proposal worker.
//!
//! The caller supplies the isolated Python runtime identity and distribution inventory collected
//! by its launcher. This module owns validation, byte hashing, deterministic JSON, and durable
//! receipt creation; it never loads a policy model or fits data.

use anyhow::{ensure, Context, Result};
use serde::ser::{
    Error as _, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant,
    SerializeTuple, SerializeTupleStruct, SerializeTupleVariant, Serializer,
};
use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Artifact {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct DistributionFile {
    pub item: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Distribution {
    pub name: String,
    pub version: String,
    pub files: Vec<DistributionFile>,
}

/// Runtime metadata captured by the isolated Python bootstrap.
#[derive(Clone, Debug)]
pub struct RuntimeContext {
    pub executable: PathBuf,
    pub isolated: bool,
    pub identity: Value,
    pub distributions: Vec<Distribution>,
}

/// Serialize compact JSON with lexically ordered object keys and reject non-finite floats.
pub fn encoded<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    value.serialize(FiniteValue)?;
    fn sorted(value: Value) -> Value {
        match value {
            Value::Array(values) => Value::Array(values.into_iter().map(sorted).collect()),
            Value::Object(values) => {
                let values = values.into_iter().collect::<BTreeMap<_, _>>();
                let mut output = Map::new();
                for (key, value) in values {
                    output.insert(key, sorted(value));
                }
                Value::Object(output)
            }
            other => other,
        }
    }
    Ok(serde_json::to_vec(&sorted(serde_json::to_value(value)?))?)
}

struct FiniteValue;

impl Serializer for FiniteValue {
    type Ok = ();
    type Error = serde_json::Error;
    type SerializeSeq = FiniteCompound;
    type SerializeTuple = FiniteCompound;
    type SerializeTupleStruct = FiniteCompound;
    type SerializeTupleVariant = FiniteCompound;
    type SerializeMap = FiniteMap;
    type SerializeStruct = FiniteCompound;
    type SerializeStructVariant = FiniteCompound;

    fn serialize_bool(self, _: bool) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_i8(self, _: i8) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_i16(self, _: i16) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_i32(self, _: i32) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_i64(self, _: i64) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_i128(self, _: i128) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_u8(self, _: u8) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_u16(self, _: u16) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_u32(self, _: u32) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_u64(self, _: u64) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_u128(self, _: u128) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_f32(self, value: f32) -> std::result::Result<(), Self::Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(serde_json::Error::custom("non-finite floating-point value"))
        }
    }
    fn serialize_f64(self, value: f64) -> std::result::Result<(), Self::Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(serde_json::Error::custom("non-finite floating-point value"))
        }
    }
    fn serialize_char(self, _: char) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_str(self, _: &str) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_bytes(self, _: &[u8]) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_none(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_some<T: ?Sized + Serialize>(
        self,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn serialize_unit(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn serialize_seq(
        self,
        _: Option<usize>,
    ) -> std::result::Result<Self::SerializeSeq, Self::Error> {
        Ok(FiniteCompound)
    }
    fn serialize_tuple(self, _: usize) -> std::result::Result<Self::SerializeTuple, Self::Error> {
        Ok(FiniteCompound)
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> std::result::Result<Self::SerializeTupleStruct, Self::Error> {
        Ok(FiniteCompound)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> std::result::Result<Self::SerializeTupleVariant, Self::Error> {
        Ok(FiniteCompound)
    }
    fn serialize_map(
        self,
        _: Option<usize>,
    ) -> std::result::Result<Self::SerializeMap, Self::Error> {
        Ok(FiniteMap)
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> std::result::Result<Self::SerializeStruct, Self::Error> {
        Ok(FiniteCompound)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> std::result::Result<Self::SerializeStructVariant, Self::Error> {
        Ok(FiniteCompound)
    }
    fn collect_str<T: ?Sized + std::fmt::Display>(
        self,
        _: &T,
    ) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

struct FiniteCompound;

impl SerializeSeq for FiniteCompound {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_element<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn end(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeTuple for FiniteCompound {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_element<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn end(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeTupleStruct for FiniteCompound {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn end(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeTupleVariant for FiniteCompound {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn end(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeStruct for FiniteCompound {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _: &'static str,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn end(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeStructVariant for FiniteCompound {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _: &'static str,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn end(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

struct FiniteMap;

impl SerializeMap for FiniteMap {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_key<T: ?Sized + Serialize>(
        &mut self,
        key: &T,
    ) -> std::result::Result<(), Self::Error> {
        key.serialize(FiniteValue)
    }
    fn serialize_value<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> std::result::Result<(), Self::Error> {
        value.serialize(FiniteValue)
    }
    fn end(self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

/// SHA-256 the exact bytes of a file without loading the entire file into memory.
pub fn sha(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut block = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut block)?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Resolve a regular file and bind its canonical path to the hash of its bytes.
pub fn artifact(path: &Path) -> Result<Artifact> {
    let path = fs::canonicalize(path)?;
    ensure!(path.is_file(), "runtime dependency must be a regular file");
    Ok(Artifact {
        sha256: sha(&path)?,
        path: path.to_string_lossy().into_owned(),
    })
}

/// Create a new private JSON receipt, flush its bytes to disk, then return its artifact reference.
pub fn write(path: &Path, value: &Value) -> Result<Artifact> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(&encoded(value)?)?;
    file.flush()?;
    file.sync_all()?;
    artifact(path)
}

fn absolute(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn resolve_non_strict(path: &Path) -> Result<PathBuf> {
    let absolute = absolute(path)?;
    if absolute.exists() {
        return Ok(fs::canonicalize(absolute)?);
    }
    let mut ancestor = absolute.as_path();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        let name = ancestor
            .file_name()
            .context("runtime output has no existing parent")?;
        suffix.push(name.to_os_string());
        ancestor = ancestor
            .parent()
            .context("runtime output has no existing parent")?;
    }
    let mut resolved = fs::canonicalize(ancestor)?;
    for component in suffix.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn under_tmp(path: &Path) -> bool {
    path == Path::new("/tmp") || path.starts_with("/tmp")
}

fn source_files(root: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            source_files(&path, output)?;
        } else if path.extension().is_some_and(|extension| extension == "py") {
            output.push(path);
        }
    }
    Ok(())
}

fn distribution_key(name: &str) -> String {
    name.to_lowercase().replace('_', "-")
}

/// Prepare deterministic source, distribution, runtime, and manifest receipts in a new directory.
///
/// `context.identity` is the exact `training_campaign.runtime_identity()` value gathered by the
/// physical interpreter; `context.distributions` is its `importlib.metadata` inventory.
pub fn prepare(
    source_root: &Path,
    site_packages: &Path,
    output: &Path,
    context: &RuntimeContext,
) -> Result<Artifact> {
    let source_root = fs::canonicalize(source_root)?;
    let site_packages = fs::canonicalize(site_packages)?;
    ensure!(
        site_packages.is_dir(),
        "explicit proposal source and site-packages roots are required"
    );
    let output = absolute(output)?;
    let resolved_output = resolve_non_strict(&output)?;
    ensure!(
        !under_tmp(&resolved_output) && !output.exists(),
        "runtime preparation requires a new durable output directory"
    );
    let executable_absolute = absolute(&context.executable)?;
    let executable_resolved = fs::canonicalize(&context.executable)?;
    ensure!(
        executable_absolute == executable_resolved && context.isolated,
        "invoke the physical interpreter with -I -B"
    );
    ensure!(
        source_root
            .join("jope/policy/experimental_proposal.py")
            .is_file(),
        "explicit proposal source and site-packages roots are required"
    );

    let mut sources = Vec::new();
    source_files(&source_root.join("jope"), &mut sources)?;
    sources.sort();
    let sources = sources
        .iter()
        .map(|path| artifact(path))
        .collect::<Result<Vec<_>>>()?;

    let mut packages = Map::new();
    let mut installed = BTreeMap::<String, Artifact>::new();
    for distribution in &context.distributions {
        let name = distribution_key(&distribution.name);
        ensure!(
            !packages.contains_key(&name),
            "ambiguous installed distribution"
        );
        let mut files = Map::new();
        let mut items = distribution.files.iter().collect::<Vec<_>>();
        items.sort_by(|left, right| left.item.cmp(&right.item));
        for item in items {
            let path = Path::new(&item.item);
            if path.extension().is_some_and(|extension| extension == "pyc")
                || path
                    .components()
                    .any(|part| part.as_os_str() == "__pycache__")
            {
                continue;
            }
            let reference = artifact(&item.path)?;
            files.insert(item.item.clone(), serde_json::to_value(&reference)?);
            match installed.get(&reference.path) {
                Some(prior) => ensure!(prior == &reference, "shared package file differs"),
                None => {
                    installed.insert(reference.path.clone(), reference);
                }
            }
        }
        packages.insert(
            name,
            json!({"version": distribution.version, "files": files}),
        );
    }
    let identity_packages = context
        .identity
        .get("packages")
        .and_then(Value::as_object)
        .context("runtime identity package inventory is absent")?
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    ensure!(
        packages.keys().cloned().collect::<BTreeSet<_>>() == identity_packages,
        "runtime_identity must include every installed distribution"
    );

    let runtime = json!({
        "program": artifact(&context.executable)?,
        "source_root": source_root,
        "site_packages": site_packages,
        "source_files": sources,
        "package_files": installed.into_values().collect::<Vec<_>>(),
        "runtime_sha256": format!("{:x}", Sha256::digest(encoded(&context.identity)?)),
        "environment": {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    });

    fs::create_dir_all(&output)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&output, fs::Permissions::from_mode(0o700))?;
    }
    let identity_ref = write(&output.join("runtime-identity.json"), &context.identity)?;
    let manifest_ref = write(
        &output.join("runtime-manifest.json"),
        &json!({
            "schema": "jope.experimental-python-runtime/v1",
            "runtime": runtime,
            "identity": identity_ref,
            "packages": packages,
        }),
    )?;
    let mut result = runtime
        .as_object()
        .cloned()
        .context("runtime receipt must be a JSON object")?;
    result.insert("manifest".to_owned(), serde_json::to_value(manifest_ref)?);
    write(&output.join("runtime.json"), &Value::Object(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target/runtime-contract-fixtures")
                .join(format!(
                    "{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
            fs::create_dir_all(&root).unwrap();
            Self(root)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_text(path: &Path, value: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, value).unwrap();
    }

    fn context(
        executable: PathBuf,
        isolated: bool,
        identity: Value,
        distributions: Vec<Distribution>,
    ) -> RuntimeContext {
        RuntimeContext {
            executable,
            isolated,
            identity,
            distributions,
        }
    }

    #[test]
    fn runtime_requires_isolation_and_new_durable_output() {
        let fixture = Fixture::new();
        let source = fixture.0.join("source");
        let packages = fixture.0.join("packages");
        write_text(
            &source.join("jope/policy/experimental_proposal.py"),
            "# fixture",
        );
        fs::create_dir_all(&packages).unwrap();
        let executable = std::env::current_exe().unwrap();
        let runtime = context(executable, false, json!({"packages": {}}), vec![]);
        let output = fixture.0.join("output");
        let error = prepare(&source, &packages, &output, &runtime).unwrap_err();
        assert!(error.to_string().contains("physical interpreter"));

        let runtime = context(
            std::env::current_exe().unwrap(),
            true,
            json!({"packages": {}}),
            vec![],
        );
        let error = prepare(&source, &packages, &packages, &runtime).unwrap_err();
        assert!(error.to_string().contains("new durable"));
        let error = prepare(
            &source,
            &packages,
            Path::new("/tmp/jope-uncreated-fixture"),
            &runtime,
        )
        .unwrap_err();
        assert!(error.to_string().contains("new durable"));
    }

    #[test]
    fn runtime_captures_exact_source_and_distribution_bytes() {
        let fixture = Fixture::new();
        let source = fixture.0.join("source");
        let packages = fixture.0.join("packages");
        let module = source.join("jope/policy/experimental_proposal.py");
        let package = packages.join("synthetic.py");
        write_text(&module, "# fixture proposal source");
        write_text(&package, "# fixture distribution");
        let executable = std::env::current_exe().unwrap();
        let distribution_files = json!({"synthetic.py": sha(&package).unwrap()});
        let distribution_digest = format!(
            "{:x}",
            Sha256::digest(encoded(&distribution_files).unwrap())
        );
        let identity = json!({
            "python": artifact(&executable).unwrap(),
            "sources": {"policy/experimental_proposal.py": sha(&module).unwrap()},
            "packages": {"synthetic": {
                "version": "1",
                "files": 1,
                "files_sha256": distribution_digest,
            }},
        });
        let distribution = Distribution {
            name: "synthetic".into(),
            version: "1".into(),
            files: vec![DistributionFile {
                item: "synthetic.py".into(),
                path: package.clone(),
            }],
        };
        let runtime = context(
            executable.clone(),
            true,
            identity.clone(),
            vec![distribution],
        );
        let output = fixture.0.join("output");
        let result = prepare(&source, &packages, &output, &runtime).unwrap();
        let value: Value = serde_json::from_slice(&fs::read(&result.path).unwrap()).unwrap();
        let manifest: Value =
            serde_json::from_slice(&fs::read(value["manifest"]["path"].as_str().unwrap()).unwrap())
                .unwrap();
        assert_eq!(
            value["program"],
            serde_json::to_value(artifact(&executable).unwrap()).unwrap()
        );
        assert_eq!(value["source_files"], json!([artifact(&module).unwrap()]));
        assert_eq!(value["package_files"], json!([artifact(&package).unwrap()]));
        assert_eq!(
            value["runtime_sha256"],
            format!("{:x}", Sha256::digest(encoded(&identity).unwrap()))
        );
        assert_eq!(
            manifest["packages"]["synthetic"]["files"]["synthetic.py"],
            serde_json::to_value(artifact(&package).unwrap()).unwrap()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&result.path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        assert!(value["environment"].get("PYTHONPATH").is_none());
        let before = value["package_files"][0]["sha256"]
            .as_str()
            .unwrap()
            .to_owned();
        write_text(&package, "# changed distribution");
        assert_ne!(sha(&package).unwrap(), before);
    }

    #[test]
    fn runtime_receipts_are_immutable_and_nonfinite_values_refused() {
        let fixture = Fixture::new();
        let path = fixture.0.join("receipt.json");
        let reference = write(&path, &json!({"source": "fixture"})).unwrap();
        assert_eq!(sha(&path).unwrap(), reference.sha256);
        assert!(write(&path, &json!({"source": "changed"})).is_err());
        assert!(encoded(&f64::NAN).is_err());
    }
}
