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
    /// `DT_NEEDED` entries in file order.
    pub needed: Vec<String>,
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

    /// `DT_NEEDED` entries that name a CUDA or LibTorch library.
    pub fn gpu_needed(&self) -> Vec<String> {
        self.needed
            .iter()
            .filter(|n| is_gpu_library(n))
            .cloned()
            .collect()
    }
}

/// Parses `GLIBC_2.39` into `[2, 39]`; `None` for anything else.
pub fn glibc_version(tag: &str) -> Option<Vec<u32>> {
    let rest = tag.strip_prefix("GLIBC_")?;
    rest.split('.').map(|p| p.parse::<u32>().ok()).collect()
}

pub fn is_gpu_library(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.starts_with("libcuda") || n.starts_with("libtorch") || n.starts_with("libc10")
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
        d.get(o..o + 2).context("ELF truncated")?.try_into()?,
    ))
}
fn u32_at(d: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        d.get(o..o + 4).context("ELF truncated")?.try_into()?,
    ))
}
fn u64_at(d: &[u8], o: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        d.get(o..o + 8).context("ELF truncated")?.try_into()?,
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
        let h = slice(data, shoff + i * shentsize, 64)?;
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
    let mut needed = Vec::new();
    let mut version_requirements = Vec::new();
    for s in &sections {
        match s.kind {
            SHT_DYNAMIC => {
                let strings = strtab(s.link)?;
                let body = slice(data, s.offset, s.size)?;
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
                    let count = u16_at(body, at + 2)?;
                    let file = c_str(strings, u32_at(body, at + 4)?)?;
                    let mut aux = at + u32_at(body, at + 8)? as usize;
                    for _ in 0..count {
                        let name = c_str(strings, u32_at(body, aux + 8)?)?;
                        version_requirements.push((file.clone(), name));
                        let next = u32_at(body, aux + 12)? as usize;
                        if next == 0 {
                            break;
                        }
                        aux += next;
                    }
                    let next = u32_at(body, at + 12)? as usize;
                    if next == 0 {
                        break;
                    }
                    at += next;
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
    pub command: Option<&'a str>,
    pub builder_image: Option<&'a str>,
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
    let rustc = match &input.rustc_vv {
        Some(text) => text.clone(),
        None => run("rustc", &["-vV"], Some(input.source_dir))
            .context("rustc -vV failed; pass the captured output instead")?,
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
        "binary_path": input.binary.display().to_string(),
        "binary_sha256": sha256_hex(&binary),
        "bytes": binary.len(),
        "glibc_max_needed": elf.max_glibc(),
        "needed": elf.needed,
        "links_cuda_or_torch": !gpu.is_empty(),
        "gpu_needed": gpu,
        "host": run("uname", &["-n"], None),
        "built_at": utc_timestamp(secs),
    }))
}

/// Enforces deployment bounds on a written receipt: the binary's GLIBC
/// requirement must not exceed `max_glibc` (e.g. `2.36`), and with
/// `forbid_gpu` no NEEDED entry may name libcuda/libtorch/libc10.
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
            "binary links CUDA/LibTorch: {}",
            receipt["gpu_needed"]
        );
    }
    Ok(())
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
        let mut place = |image: &mut Vec<u8>, body: &[u8]| {
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
            elf.needed,
            ["libm.so.6", "libc.so.6", "libtorch_cuda.so"],
            "stops at DT_NULL"
        );
        assert_eq!(elf.version_requirements.len(), 5);
        assert_eq!(
            elf.version_requirements[0],
            ("libm.so.6".into(), "GLIBC_2.27".into())
        );
        // Numeric, not lexical: 2.34 > 2.27 > 2.17 > 2.9; PRIVATE ignored.
        assert_eq!(elf.max_glibc().as_deref(), Some("GLIBC_2.34"));
        assert_eq!(elf.gpu_needed(), ["libtorch_cuda.so"]);
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
        assert!(elf.needed.iter().any(|n| n == "libc.so.6"), "{elf:?}");
        let max = elf.max_glibc().expect("glibc requirement");
        assert!(glibc_version(&max).unwrap() >= vec![2, 2], "{max}");
        assert!(elf.gpu_needed().is_empty());
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
    }

    #[test]
    fn timestamp_is_utc_civil() {
        assert_eq!(utc_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_timestamp(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_timestamp(1_789_599_389), "2026-09-16T22:56:29Z");
    }
}
