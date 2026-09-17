//! `jope.product_evaluator_build.v1` receipts for a built evaluator binary.
//!
//! The ELF inspection is a minimal, read-only ELF64 little-endian parser: it
//! reads the dynamic section (`DT_NEEDED`) and the GNU version-requirement
//! section (`SHT_GNU_verneed`) to report which shared libraries a binary
//! loads and the highest `GLIBC_x.y` symbol version it requires. That bound
//! decides whether a binary can run on an older-glibc appliance.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

pub const SCHEMA: &str = "jope.product_evaluator_build.v1";

const SHT_STRTAB: u32 = 3;
const SHT_DYNAMIC: u32 = 6;
const SHT_GNU_VERNEED: u32 = 0x6fff_fffe;
const DT_NULL: u64 = 0;
const DT_NEEDED: u64 = 1;

/// Dynamic-linking facts read from an ELF64 image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfDynamic {
    /// `DT_NEEDED` entries in file order; `None` when the image has no
    /// `SHT_DYNAMIC` section (static, or section headers stripped), so a
    /// missing section is never mistaken for "links nothing".
    pub needed: Option<Vec<String>>,
    /// Every `(library, version)` pair in the version-requirement section.
    pub version_requirements: Vec<(String, String)>,
}

impl ElfDynamic {
    /// The highest `GLIBC_<numeric>` version required, e.g. `GLIBC_2.34`.
    /// Non-numeric tags such as `GLIBC_PRIVATE` are ignored.
    pub fn max_glibc(&self) -> Option<String> {
        self.version_requirements
            .iter()
            .filter_map(|(_, v)| glibc_version(v).map(|n| (n, v)))
            .max_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, v)| v.clone())
    }

    /// `DT_NEEDED` entries that name a CUDA or LibTorch library; `None`
    /// when the dynamic section is absent (nothing could be checked).
    pub fn gpu_needed(&self) -> Option<Vec<String>> {
        let needed = self.needed.as_ref()?;
        Some(
            needed
                .iter()
                .filter(|n| is_gpu_library(n))
                .cloned()
                .collect(),
        )
    }
}

/// Parses `GLIBC_2.39` into `[2, 39]`; `None` for anything else.
pub fn glibc_version(tag: &str) -> Option<Vec<u32>> {
    let rest = tag.strip_prefix("GLIBC_")?;
    rest.split('.').map(|p| p.parse::<u32>().ok()).collect()
}

/// Name prefixes (lowercase) of CUDA, NVIDIA runtime and LibTorch libraries.
/// The CUDA names are listed individually rather than as a bare `libcu`
/// prefix, which would also match unrelated `libcurl`/`libcups`.
const GPU_LIBRARY_PREFIXES: &[&str] = &[
    "libcuda",
    "libcudart",
    "libcudnn",
    "libcublas",
    "libcurand",
    "libcusparse",
    "libcufft",
    "libcusolver",
    "libcupti",
    "libcufile",
    "libcutensor",
    "libnpp",
    "libnvjpeg",
    "libnccl",
    "libnvrtc",
    "libnvjitlink",
    "libnvtoolsext",
    "libnvidia",
    "libtorch",
    "libc10",
];

pub fn is_gpu_library(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    GPU_LIBRARY_PREFIXES.iter().any(|p| n.starts_with(p))
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).context("ELF offset overflow")
}

fn slice(data: &[u8], offset: u64, len: u64) -> Result<&[u8]> {
    let start = usize::try_from(offset)?;
    let end = start
        .checked_add(usize::try_from(len)?)
        .context("ELF range overflow")?;
    data.get(start..end)
        .with_context(|| format!("ELF range {start}..{end} outside file"))
}
fn u16_at(d: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        d.get(o..add(o, 2)?).context("ELF truncated")?.try_into()?,
    ))
}
fn u32_at(d: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        d.get(o..add(o, 4)?).context("ELF truncated")?.try_into()?,
    ))
}
fn u64_at(d: &[u8], o: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        d.get(o..add(o, 8)?).context("ELF truncated")?.try_into()?,
    ))
}
fn c_str(table: &[u8], offset: u32) -> Result<String> {
    let tail = table
        .get(offset as usize..)
        .context("string offset outside table")?;
    let end = tail
        .iter()
        .position(|b| *b == 0)
        .context("unterminated ELF string")?;
    Ok(String::from_utf8(tail[..end].to_vec())?)
}

struct Section {
    kind: u32,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
}

/// Reads `DT_NEEDED` and GNU version requirements from an ELF64 LE image.
pub fn parse_elf_dynamic(data: &[u8]) -> Result<ElfDynamic> {
    ensure!(
        data.len() >= 64 && &data[..4] == b"\x7fELF",
        "not an ELF file"
    );
    ensure!(data[4] == 2, "only ELF64 is supported");
    ensure!(data[5] == 1, "only little-endian ELF is supported");
    let shoff = u64_at(data, 0x28)?;
    let shentsize = u16_at(data, 0x3a)? as u64;
    let shnum = u16_at(data, 0x3c)? as u64;
    ensure!(
        shentsize >= 64,
        "unexpected section header size {shentsize}"
    );
    let mut sections = Vec::new();
    for i in 0..shnum {
        let at = i
            .checked_mul(shentsize)
            .and_then(|o| o.checked_add(shoff))
            .context("section header offset overflow")?;
        let h = slice(data, at, 64)?;
        sections.push(Section {
            kind: u32_at(h, 4)?,
            offset: u64_at(h, 24)?,
            size: u64_at(h, 32)?,
            link: u32_at(h, 40)?,
            info: u32_at(h, 44)?,
        });
    }
    let strtab = |link: u32| -> Result<&[u8]> {
        let s = sections
            .get(link as usize)
            .context("section link out of range")?;
        ensure!(s.kind == SHT_STRTAB, "section link is not a string table");
        slice(data, s.offset, s.size)
    };
    let mut needed: Option<Vec<String>> = None;
    let mut version_requirements = Vec::new();
    for s in &sections {
        match s.kind {
            SHT_DYNAMIC => {
                let strings = strtab(s.link)?;
                let body = slice(data, s.offset, s.size)?;
                let needed = needed.get_or_insert_with(Vec::new);
                for entry in body.chunks_exact(16) {
                    let tag = u64_at(entry, 0)?;
                    if tag == DT_NULL {
                        break;
                    }
                    if tag == DT_NEEDED {
                        needed.push(c_str(strings, u32::try_from(u64_at(entry, 8)?)?)?);
                    }
                }
            }
            SHT_GNU_VERNEED => {
                let strings = strtab(s.link)?;
                let body = slice(data, s.offset, s.size)?;
                let mut at = 0usize;
                for _ in 0..s.info {
                    let count = u16_at(body, add(at, 2)?)?;
                    let file = c_str(strings, u32_at(body, add(at, 4)?)?)?;
                    let mut aux = add(at, u32_at(body, add(at, 8)?)? as usize)?;
                    for _ in 0..count {
                        let name = c_str(strings, u32_at(body, add(aux, 8)?)?)?;
                        version_requirements.push((file.clone(), name));
                        let next = u32_at(body, add(aux, 12)?)? as usize;
                        if next == 0 {
                            break;
                        }
                        aux = add(aux, next)?;
                    }
                    let next = u32_at(body, add(at, 12)?)? as usize;
                    if next == 0 {
                        break;
                    }
                    at = add(at, next)?;
                }
            }
            _ => {}
        }
    }
    Ok(ElfDynamic {
        needed,
        version_requirements,
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn run(program: &str, args: &[&str], dir: Option<&Path>) -> Option<String> {
    let mut command = Command::new(program);
    command.args(args).env("GIT_OPTIONAL_LOCKS", "0");
    if let Some(dir) = dir {
        command.current_dir(dir);
    }
    let output = command.output().ok()?;
    output.status.success().then(|| {
        String::from_utf8_lossy(&output.stdout)
            .trim_end()
            .to_owned()
    })
}

/// Read-only git identity of `dir`: HEAD commit, the tree object of `dir` at
/// HEAD, and whether the working tree under `dir` differs from HEAD.
fn git_identity(dir: &Path) -> Value {
    let Some(commit) = run("git", &["rev-parse", "HEAD"], Some(dir)) else {
        return Value::Null;
    };
    let prefix = run("git", &["rev-parse", "--show-prefix"], Some(dir)).unwrap_or_default();
    let spec = format!("HEAD:{}", prefix.trim_end_matches('/'));
    let tree = run("git", &["rev-parse", &spec], Some(dir));
    let status = run(
        "git",
        &["status", "--porcelain", "--untracked-files=all", "--", "."],
        Some(dir),
    );
    json!({
        "commit": commit,
        "path": prefix.trim_end_matches('/'),
        "tree": tree,
        "clean": status.map(|s| s.is_empty()),
    })
}

/// Seconds since the epoch as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_timestamp(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

pub struct ReceiptInput<'a> {
    pub binary: &'a Path,
    pub source_dir: &'a Path,
    pub lineage: Option<&'a str>,
    /// Captured `rustc -vV` of the toolchain that built the binary. When
    /// absent, the local `rustc -vV` is recorded.
    pub rustc_vv: Option<String>,
    /// The builder's `RUSTUP_TOOLCHAIN` (empty or `None` when unset). It
    /// overrides the source's `rust-toolchain.toml`. When `rustc_vv` is
    /// absent, the local environment's value is recorded instead.
    pub rustup_toolchain: Option<&'a str>,
    pub command: Option<&'a str>,
    pub builder_image: Option<&'a str>,
    /// Cargo profile the binary was built with, e.g. `release`.
    pub profile: &'a str,
    /// Cargo features enabled for the build (empty for the default set).
    pub features: &'a [String],
}

/// The `channel` of a `rust-toolchain.toml` `[toolchain]` table.
pub fn toolchain_file_channel(text: &str) -> Option<String> {
    let mut in_toolchain = false;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') {
            in_toolchain = line == "[toolchain]";
            continue;
        }
        if !in_toolchain {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            if key.trim() == "channel" {
                let value = value.trim();
                let unquoted = value
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))?;
                return Some(unquoted.to_owned());
            }
        }
    }
    None
}

/// Requested (`rust-toolchain.toml`) versus actual (`rustc -vV`) toolchain.
/// A builder `RUSTUP_TOOLCHAIN` silently wins over the file, so both sides
/// and the override are recorded and `matches_file` makes a mismatch visible.
pub fn toolchain_record(
    source_dir: &Path,
    rustc_vv: &str,
    rustup_toolchain: Option<&str>,
) -> Value {
    let file = source_dir.join("rust-toolchain.toml");
    let channel = std::fs::read_to_string(&file)
        .ok()
        .and_then(|t| toolchain_file_channel(&t));
    let release = rustc_vv
        .lines()
        .find_map(|l| l.strip_prefix("release: "))
        .map(str::trim);
    let override_ = rustup_toolchain.filter(|s| !s.is_empty());
    json!({
        "file": file.is_file().then_some("rust-toolchain.toml"),
        "file_channel": channel,
        "rustup_toolchain_override": override_,
        "rustc_release": release,
        "matches_file": match (&channel, release) {
            (Some(c), Some(r)) => Value::Bool(c == r),
            _ => Value::Null,
        },
    })
}

pub fn build_receipt(input: &ReceiptInput) -> Result<Value> {
    let binary =
        std::fs::read(input.binary).with_context(|| format!("read {}", input.binary.display()))?;
    let elf = parse_elf_dynamic(&binary)
        .with_context(|| format!("parse ELF {}", input.binary.display()))?;
    let hash_file = |name: &str| -> Result<String> {
        let path = input.source_dir.join(name);
        Ok(sha256_hex(
            &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        ))
    };
    let local_override;
    let (rustc, rustup_toolchain) = match &input.rustc_vv {
        Some(text) => (text.clone(), input.rustup_toolchain),
        None => {
            local_override = std::env::var("RUSTUP_TOOLCHAIN").ok();
            (
                run("rustc", &["-vV"], Some(input.source_dir))
                    .context("rustc -vV failed; pass the captured output instead")?,
                local_override.as_deref(),
            )
        }
    };
    if !input.source_dir.is_dir() {
        bail!(
            "source dir {} is not a directory",
            input.source_dir.display()
        );
    }
    let gpu = elf.gpu_needed();
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    Ok(json!({
        "schema": SCHEMA,
        "lineage": input.lineage,
        "source_dir": input.source_dir.display().to_string(),
        "git": git_identity(input.source_dir),
        "cargo_toml_sha256": hash_file("Cargo.toml")?,
        "cargo_lock_sha256": hash_file("Cargo.lock")?,
        "command": input.command,
        "builder_image": input.builder_image,
        "rustc": rustc.lines().collect::<Vec<_>>(),
        "toolchain": toolchain_record(input.source_dir, &rustc, rustup_toolchain),
        "features": input.features,
        "profile": input.profile,
        "binary_path": input.binary.display().to_string(),
        "binary_sha256": sha256_hex(&binary),
        "bytes": binary.len(),
        "glibc_max_needed": elf.max_glibc(),
        "needed": elf.needed,
        "links_cuda_or_torch": gpu.as_ref().map(|g| !g.is_empty()),
        "gpu_needed": gpu,
        "host": run("uname", &["-n"], None),
        "built_at": utc_timestamp(secs),
    }))
}

/// Enforces deployment bounds on a receipt: the binary's GLIBC
/// requirement must not exceed `max_glibc` (e.g. `2.36`), and with
/// `forbid_gpu` no NEEDED entry may name a CUDA/NVIDIA/LibTorch library.
/// A receipt without a dynamic section (`links_cuda_or_torch: null`) fails.
pub fn check_bounds(receipt: &Value, max_glibc: Option<&str>, forbid_gpu: bool) -> Result<()> {
    if let Some(bound) = max_glibc {
        let limit = glibc_version(&format!("GLIBC_{bound}"))
            .with_context(|| format!("invalid --max-glibc {bound}"))?;
        let required = receipt["glibc_max_needed"].as_str();
        let version = required
            .and_then(glibc_version)
            .with_context(|| format!("binary has no numeric GLIBC requirement ({required:?})"))?;
        ensure!(
            version <= limit,
            "binary requires {} which exceeds GLIBC_{bound}",
            required.unwrap_or_default()
        );
    }
    if forbid_gpu {
        ensure!(
            receipt["links_cuda_or_torch"] == Value::Bool(false),
            "binary links CUDA/LibTorch or has no dynamic section: {}",
            receipt["gpu_needed"]
        );
    }
    Ok(())
}

/// Runs [`check_bounds`] and records the checked limits and the verdict in
/// the receipt as `bounds`, so a rejected build's receipt is
/// distinguishable from a passing one. Returns the check's result.
pub fn apply_bounds(receipt: &mut Value, max_glibc: Option<&str>, forbid_gpu: bool) -> Result<()> {
    let result = check_bounds(receipt, max_glibc, forbid_gpu);
    receipt["bounds"] = json!({
        "max_glibc": max_glibc,
        "forbid_gpu_libraries": forbid_gpu,
        "passed": result.is_ok(),
        "error": result.as_ref().err().map(|e| format!("{e:#}")),
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_are_numeric_and_fail_closed() {
        let ok = json!({"glibc_max_needed": "GLIBC_2.34", "links_cuda_or_torch": false, "gpu_needed": []});
        check_bounds(&ok, Some("2.36"), true).unwrap();
        check_bounds(&ok, Some("2.34"), true).unwrap();
        assert!(check_bounds(&ok, Some("2.33"), true).is_err());
        assert!(check_bounds(&ok, Some("banana"), false).is_err());
        let new = json!({"glibc_max_needed": "GLIBC_2.39"});
        assert!(check_bounds(&new, Some("2.36"), false).is_err());
        let none = json!({"glibc_max_needed": null});
        assert!(check_bounds(&none, Some("2.36"), false).is_err());
        let gpu = json!({"glibc_max_needed": "GLIBC_2.17", "links_cuda_or_torch": true, "gpu_needed": ["libcuda.so.1"]});
        assert!(check_bounds(&gpu, Some("2.36"), true).is_err());
        check_bounds(&gpu, Some("2.36"), false).unwrap();
        assert!(check_bounds(&json!({}), None, true).is_err());
    }

    /// Builds a minimal ELF64 LE image with a string table, a dynamic
    /// section and a version-requirement section (no program headers).
    fn synthetic_elf() -> Vec<u8> {
        let mut strings = vec![0u8];
        let mut add = |s: &str| {
            let at = strings.len() as u32;
            strings.extend_from_slice(s.as_bytes());
            strings.push(0);
            at
        };
        let libc = add("libc.so.6");
        let libm = add("libm.so.6");
        let torch = add("libtorch_cuda.so");
        let v217 = add("GLIBC_2.17");
        let v234 = add("GLIBC_2.34");
        let v29 = add("GLIBC_2.9");
        let private = add("GLIBC_PRIVATE");
        let v227 = add("GLIBC_2.27");

        let mut dynamic = Vec::new();
        for (tag, val) in [
            (DT_NEEDED, libm),
            (DT_NEEDED, libc),
            (DT_NEEDED, torch),
            (DT_NULL, 0),
            (DT_NEEDED, 999_999),
        ] {
            dynamic.extend_from_slice(&tag.to_le_bytes());
            dynamic.extend_from_slice(&u64::from(val).to_le_bytes());
        }

        let verneed_entry = |cnt: u16, file: u32, next: u32| {
            let mut v = Vec::new();
            v.extend_from_slice(&1u16.to_le_bytes());
            v.extend_from_slice(&cnt.to_le_bytes());
            v.extend_from_slice(&file.to_le_bytes());
            v.extend_from_slice(&16u32.to_le_bytes());
            v.extend_from_slice(&next.to_le_bytes());
            v
        };
        let aux = |name: u32, next: u32| {
            let mut v = Vec::new();
            v.extend_from_slice(&0u32.to_le_bytes());
            v.extend_from_slice(&0u16.to_le_bytes());
            v.extend_from_slice(&0u16.to_le_bytes());
            v.extend_from_slice(&name.to_le_bytes());
            v.extend_from_slice(&next.to_le_bytes());
            v
        };
        let mut verneed = Vec::new();
        // libm: one version; libc: four versions (incl. PRIVATE, 2.9 < 2.17).
        verneed.extend(verneed_entry(1, libm, 32));
        verneed.extend(aux(v227, 0));
        verneed.extend(verneed_entry(4, libc, 0));
        verneed.extend(aux(v217, 16));
        verneed.extend(aux(v234, 16));
        verneed.extend(aux(private, 16));
        verneed.extend(aux(v29, 0));

        let mut image = vec![0u8; 64];
        image[..4].copy_from_slice(b"\x7fELF");
        image[4] = 2;
        image[5] = 1;
        image[6] = 1;
        let place = |image: &mut Vec<u8>, body: &[u8]| {
            let at = image.len() as u64;
            image.extend_from_slice(body);
            at
        };
        let str_off = place(&mut image, &strings);
        let dyn_off = place(&mut image, &dynamic);
        let ver_off = place(&mut image, &verneed);
        let shoff = image.len() as u64;
        let header = |kind: u32, off: u64, size: u64, link: u32, info: u32| {
            let mut h = vec![0u8; 64];
            h[4..8].copy_from_slice(&kind.to_le_bytes());
            h[24..32].copy_from_slice(&off.to_le_bytes());
            h[32..40].copy_from_slice(&size.to_le_bytes());
            h[40..44].copy_from_slice(&link.to_le_bytes());
            h[44..48].copy_from_slice(&info.to_le_bytes());
            h
        };
        image.extend(header(0, 0, 0, 0, 0));
        image.extend(header(SHT_STRTAB, str_off, strings.len() as u64, 0, 0));
        image.extend(header(SHT_DYNAMIC, dyn_off, dynamic.len() as u64, 1, 0));
        image.extend(header(SHT_GNU_VERNEED, ver_off, verneed.len() as u64, 1, 2));
        image[0x28..0x30].copy_from_slice(&shoff.to_le_bytes());
        image[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes());
        image[0x3c..0x3e].copy_from_slice(&4u16.to_le_bytes());
        image
    }

    #[test]
    fn synthetic_elf_needed_and_glibc_bound() {
        let elf = parse_elf_dynamic(&synthetic_elf()).unwrap();
        assert_eq!(
            elf.needed.as_ref().unwrap(),
            &["libm.so.6", "libc.so.6", "libtorch_cuda.so"],
            "stops at DT_NULL"
        );
        assert_eq!(elf.version_requirements.len(), 5);
        assert_eq!(
            elf.version_requirements[0],
            ("libm.so.6".into(), "GLIBC_2.27".into())
        );
        // Numeric, not lexical: 2.34 > 2.27 > 2.17 > 2.9; PRIVATE ignored.
        assert_eq!(elf.max_glibc().as_deref(), Some("GLIBC_2.34"));
        assert_eq!(elf.gpu_needed().unwrap(), ["libtorch_cuda.so"]);
    }

    #[test]
    fn rejects_non_elf_and_truncated_input() {
        assert!(parse_elf_dynamic(b"#!/bin/sh\n").is_err());
        let mut image = synthetic_elf();
        image.truncate(image.len() - 10);
        assert!(parse_elf_dynamic(&image).is_err());
        let mut be = synthetic_elf();
        be[5] = 2;
        assert!(parse_elf_dynamic(&be).is_err());
    }

    #[test]
    fn system_true_links_glibc() {
        let Ok(bytes) = std::fs::read("/bin/true") else {
            return;
        };
        if bytes.get(4) != Some(&2) {
            return;
        }
        let elf = parse_elf_dynamic(&bytes).unwrap();
        assert!(
            elf.needed
                .as_ref()
                .unwrap()
                .iter()
                .any(|n| n == "libc.so.6"),
            "{elf:?}"
        );
        let max = elf.max_glibc().expect("glibc requirement");
        assert!(glibc_version(&max).unwrap() >= vec![2, 2], "{max}");
        assert!(elf.gpu_needed().unwrap().is_empty());
    }

    #[test]
    fn version_ordering_and_gpu_names() {
        assert!(glibc_version("GLIBC_2.39") > glibc_version("GLIBC_2.4"));
        assert!(glibc_version("GLIBC_2.2.5") < glibc_version("GLIBC_2.3"));
        assert_eq!(glibc_version("GLIBC_PRIVATE"), None);
        assert_eq!(glibc_version("GCC_3.0"), None);
        assert!(is_gpu_library("libcuda.so.1"));
        assert!(is_gpu_library("libtorch_cpu.so"));
        assert!(!is_gpu_library("libc.so.6"));
        for gpu in [
            "libcublasLt.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
            "libcudart.so.12",
            "libnccl.so.2",
            "libnvrtc.so.12",
            "libcurand.so.10",
            "libcusparse.so.12",
            "libcufft.so.11",
            "libcusolver.so.11",
            "libnvJitLink.so.12",
            "libnvToolsExt.so.1",
            "libc10_cuda.so",
        ] {
            assert!(is_gpu_library(gpu), "{gpu}");
        }
        for cpu in [
            "libcurl.so.4",
            "libcups.so.2",
            "libcrypt.so.1",
            "libgcc_s.so.1",
            "libm.so.6",
            "ld-linux-x86-64.so.2",
        ] {
            assert!(!is_gpu_library(cpu), "{cpu}");
        }
    }

    #[test]
    fn hostile_header_offsets_error_instead_of_overflowing() {
        let mut image = synthetic_elf();
        image[0x28..0x30].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse_elf_dynamic(&image).is_err());
        let mut image = synthetic_elf();
        image[0x28..0x30].copy_from_slice(&(u64::MAX - 70).to_le_bytes());
        assert!(parse_elf_dynamic(&image).is_err());
    }

    #[test]
    fn missing_dynamic_section_is_unknown_not_clean() {
        let mut image = synthetic_elf();
        let shoff = u64::from_le_bytes(image[0x28..0x30].try_into().unwrap()) as usize;
        // Section 2 is SHT_DYNAMIC; demote it to SHT_NULL.
        image[shoff + 2 * 64 + 4..shoff + 2 * 64 + 8].copy_from_slice(&0u32.to_le_bytes());
        let elf = parse_elf_dynamic(&image).unwrap();
        assert_eq!(elf.needed, None);
        assert_eq!(elf.gpu_needed(), None);
        // Section headers stripped entirely.
        let mut stripped = synthetic_elf();
        stripped[0x3c..0x3e].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(parse_elf_dynamic(&stripped).unwrap().needed, None);
        // A receipt carrying that state cannot pass --forbid-gpu-libraries.
        let receipt = json!({"glibc_max_needed": "GLIBC_2.34", "links_cuda_or_torch": null, "gpu_needed": null});
        assert!(check_bounds(&receipt, None, true).is_err());
        check_bounds(&receipt, Some("2.36"), false).unwrap();
    }

    #[test]
    fn bounds_verdict_is_recorded_in_receipt() {
        let mut pass = json!({"glibc_max_needed": "GLIBC_2.34", "links_cuda_or_torch": false, "gpu_needed": []});
        apply_bounds(&mut pass, Some("2.36"), true).unwrap();
        assert_eq!(
            pass["bounds"],
            json!({"max_glibc": "2.36", "forbid_gpu_libraries": true, "passed": true, "error": null})
        );
        let mut fail = json!({"glibc_max_needed": "GLIBC_2.39", "links_cuda_or_torch": false, "gpu_needed": []});
        let err = apply_bounds(&mut fail, Some("2.36"), true).unwrap_err();
        assert_eq!(fail["bounds"]["passed"], false);
        assert_eq!(fail["bounds"]["max_glibc"], "2.36");
        assert_eq!(fail["bounds"]["error"], format!("{err:#}"));
        assert!(err.to_string().contains("GLIBC_2.39"));
        let mut unbounded = json!({"glibc_max_needed": "GLIBC_2.39"});
        apply_bounds(&mut unbounded, None, false).unwrap();
        assert_eq!(unbounded["bounds"]["max_glibc"], Value::Null);
        assert_eq!(unbounded["bounds"]["passed"], true);
    }

    #[test]
    fn toolchain_file_channel_parses_toolchain_table() {
        let text = "[toolchain]\nchannel = \"1.97.1\"\nprofile = \"minimal\"\n";
        assert_eq!(toolchain_file_channel(text).as_deref(), Some("1.97.1"));
        let commented = "# pinned\n[other]\nchannel = \"x\"\n[toolchain]\n channel='stable' # c\n";
        assert_eq!(toolchain_file_channel(commented).as_deref(), Some("stable"));
        assert_eq!(toolchain_file_channel("[other]\nchannel = \"x\"\n"), None);
        assert_eq!(toolchain_file_channel(""), None);
    }

    #[test]
    fn toolchain_record_exposes_rustup_override() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("rust-toolchain.toml"),
            "[toolchain]\nchannel = \"1.97.1\"\n",
        )
        .unwrap();
        let vv = "rustc 1.95.0 (abc 2026-01-01)\nbinary: rustc\nrelease: 1.95.0\n";
        assert_eq!(
            toolchain_record(dir.path(), vv, Some("1.95.0")),
            json!({
                "file": "rust-toolchain.toml",
                "file_channel": "1.97.1",
                "rustup_toolchain_override": "1.95.0",
                "rustc_release": "1.95.0",
                "matches_file": false,
            })
        );
        let honoured = toolchain_record(dir.path(), "release: 1.97.1\n", Some(""));
        assert_eq!(honoured["matches_file"], true);
        assert_eq!(honoured["rustup_toolchain_override"], Value::Null);
        let empty = tempfile::tempdir().unwrap();
        let none = toolchain_record(empty.path(), vv, None);
        assert_eq!(none["file"], Value::Null);
        assert_eq!(none["matches_file"], Value::Null);
    }

    #[test]
    fn timestamp_is_utc_civil() {
        assert_eq!(utc_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_timestamp(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_timestamp(1_789_599_389), "2026-09-16T22:56:29Z");
    }
}
