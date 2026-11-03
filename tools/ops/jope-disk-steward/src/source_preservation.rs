//! Source-history preflight for compiler-cache cleanup; never edits source checkouts.
//! Native Rust replacement for `source_preservation.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::json::Json;
use crate::json_obj;
use crate::sha256::{sha256_file, sha256_str};
use crate::sys::{atomic_write_str, disk_usage, hostname, utc_iso, utc_stamp_micros, FlockGuard};

const STRIPPED_GIT_VARS: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicState {
    pub head: String,
    pub status: String,
    pub refs: String,
    pub stashes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AuditState {
    pub basic: BasicState,
    pub source: String,
    pub common: String,
    pub shallow: bool,
    pub local_only_commits: Vec<String>,
    pub published_tag_commits: Vec<String>,
    pub unique_commits: Vec<String>,
    pub unmatched_tags: BTreeMap<String, String>,
    pub remote_tags: BTreeMap<String, String>,
    pub required_objects: Vec<String>,
    pub utc: String,
}

#[derive(Debug, Clone)]
pub struct EnsureRecord {
    pub source: String,
    pub audit: String,
    pub basic: BasicState,
    pub preservation: Json,
}

fn git_cmd(source: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::new("git");
    for k in STRIPPED_GIT_VARS {
        cmd.env_remove(k);
    }
    cmd.env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .env("GIT_NO_LAZY_FETCH", "1")
        .arg("-c")
        .arg("core.fsmonitor=false")
        .arg("-c")
        .arg("pack.threads=1")
        .arg("-C")
        .arg(source)
        .args(args);
    cmd
}

pub fn git(source: &Path, args: &[&str], stdin_data: Option<&str>) -> Result<String, String> {
    let label = args.first().copied().unwrap_or("git");
    let mut cmd = git_cmd(source, args);
    if stdin_data.is_some() {
        cmd.stdin(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("incomplete git {label} probe: {e}"))?;
    if let Some(data) = stdin_data {
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(data.as_bytes())
                .map_err(|e| format!("incomplete git {label} probe stdin: {e}"))?;
        }
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("incomplete git {label} probe: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let msg: String = stderr.chars().take(300).collect();
        return Err(format!("git {label} failed: {msg}"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn basic(source: &Path) -> Result<BasicState, String> {
    let head = git(source, &["rev-parse", "--verify", "HEAD^{commit}"], None)?
        .trim()
        .to_string();
    let status = git(
        source,
        &["status", "--porcelain=v1", "--untracked-files=normal"],
        None,
    )?;
    let refs = git(
        source,
        &["for-each-ref", "--format=%(objectname) %(refname)"],
        None,
    )?;
    let stashes = git(source, &["stash", "list", "--format=%H"], None)?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Ok(BasicState {
        head,
        status,
        refs,
        stashes,
    })
}

pub fn audit(source: &Path) -> Result<AuditState, String> {
    let resolved = fs::canonicalize(source)
        .map_err(|e| format!("cannot canonicalize {}: {e}", source.display()))?;
    let b = basic(&resolved)?;
    let common = git(
        &resolved,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        None,
    )?
    .trim()
    .to_string();
    let shallow = git(&resolved, &["rev-parse", "--is-shallow-repository"], None)?
        .trim()
        != "false";

    let partial_out = git_cmd(&resolved, &["config", "--get", "extensions.partialClone"])
        .output()
        .map_err(|e| format!("git config probe failed: {e}"))?;
    if partial_out.status.code() != Some(1) {
        return Err("partial-clone or unreadable source configuration".to_string());
    }

    let mut local: BTreeSet<String> = git(
        &resolved,
        &["log", "--all", "--not", "--remotes", "--format=%H"],
        None,
    )?
    .lines()
    .map(|s| s.trim().to_string())
    .filter(|s| !s.is_empty())
    .collect();
    for sha in git(
        &resolved,
        &["log", "HEAD", "--not", "--remotes", "--format=%H"],
        None,
    )?
    .lines()
    {
        let s = sha.trim();
        if !s.is_empty() {
            local.insert(s.to_string());
        }
    }

    let mut tags = BTreeMap::new();
    for line in git(
        &resolved,
        &["for-each-ref", "refs/tags", "--format=%(objectname) %(refname)"],
        None,
    )?
    .lines()
    {
        let mut parts = line.split_whitespace();
        if let (Some(sha), Some(r)) = (parts.next(), parts.next()) {
            tags.insert(r.to_string(), sha.to_string());
        }
    }

    let mut remote_tags = BTreeMap::new();
    let has_origin = git_cmd(&resolved, &["remote", "get-url", "origin"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if has_origin {
        if let Ok(ls_out) = git(&resolved, &["ls-remote", "--tags", "origin"], None) {
            for line in ls_out.lines() {
                let mut parts = line.split_whitespace();
                if let (Some(sha), Some(r)) = (parts.next(), parts.next()) {
                    if !r.ends_with("^{}") {
                        remote_tags.insert(r.to_string(), sha.to_string());
                    }
                }
            }
        }
    }

    let published: Vec<String> = tags
        .iter()
        .filter(|(r, sha)| remote_tags.get(*r) == Some(*sha))
        .map(|(r, _)| r.clone())
        .collect();

    let mut published_commits = BTreeSet::new();
    if !published.is_empty() {
        let mut args: Vec<&str> = vec!["rev-list"];
        for p in &published {
            args.push(p.as_str());
        }
        args.push("--not");
        args.push("--remotes");
        for line in git(&resolved, &args, None)?.lines() {
            let s = line.trim();
            if !s.is_empty() {
                published_commits.insert(s.to_string());
            }
        }
    }

    let unmatched_tags: BTreeMap<String, String> = tags
        .iter()
        .filter(|(r, sha)| remote_tags.get(*r) != Some(*sha))
        .map(|(r, sha)| (r.clone(), sha.clone()))
        .collect();

    let local_only_commits: Vec<String> = local.iter().cloned().collect();
    let published_tag_commits: Vec<String> =
        local.intersection(&published_commits).cloned().collect();
    let unique_commits: Vec<String> = local.difference(&published_commits).cloned().collect();

    let mut req: BTreeSet<String> = unique_commits.iter().cloned().collect();
    for s in &b.stashes {
        req.insert(s.clone());
    }
    for s in unmatched_tags.values() {
        req.insert(s.clone());
    }
    let required_objects: Vec<String> = req.into_iter().collect();

    Ok(AuditState {
        basic: b,
        source: resolved.to_string_lossy().to_string(),
        common,
        shallow,
        local_only_commits,
        published_tag_commits,
        unique_commits,
        unmatched_tags,
        remote_tags,
        required_objects,
        utc: utc_iso(),
    })
}

pub fn default_custody() -> Result<PathBuf, String> {
    if let Some(custom) = std::env::var_os("JOPE_DISK_STEWARD_CUSTODY") {
        return Ok(PathBuf::from(custom));
    }
    let host = hostname();
    match host.as_str() {
        "xbabe2" => Ok(PathBuf::from("/mnt/fast-scratch/disk-steward/source-preservation")),
        "xbabe3" => Ok(PathBuf::from(
            "/mnt/xbabe3-external/disk-steward-source-preservation",
        )),
        "xbabe1" => {
            let amphora = Path::new("/mnt/amphora-backup");
            if amphora.is_dir() {
                Ok(amphora.join("disk-steward-source-preservation"))
            } else {
                Ok(PathBuf::from(
                    "/mnt/jope-build-native/disk-steward-source-preservation",
                ))
            }
        }
        _ => Err("no qualified off-root source custody on this host".to_string()),
    }
}

pub fn object_check(repo: &Path, objects: &[String]) -> Result<(), String> {
    if objects.is_empty() {
        return Ok(());
    }
    let input = format!("{}\n", objects.join("\n"));
    let result = git(repo, &["cat-file", "--batch-check"], Some(&input))?;
    let lines: Vec<&str> = result.lines().collect();
    if lines.len() != objects.len() || lines.iter().any(|l| l.contains(" missing")) {
        return Err("independent source restore lacks required objects".to_string());
    }
    Ok(())
}

pub fn verify_chain(folder: &Path, receipt: &Json) -> Result<(), String> {
    let restored = folder.join("verified.git");
    if restored.join("objects/info/alternates").exists() {
        return Err("restore has source alternates".to_string());
    }
    let bundles = receipt
        .get("bundles")
        .and_then(|v| v.as_arr())
        .ok_or_else(|| "missing bundles in receipt".to_string())?;
    for record in bundles {
        let name = record
            .get_str("name")
            .ok_or_else(|| "bundle missing name".to_string())?;
        let expected_sha = record
            .get_str("sha256")
            .ok_or_else(|| "bundle missing sha256".to_string())?;
        let actual_sha = sha256_file(&folder.join(name))
            .map_err(|e| format!("failed hashing bundle {name}: {e}"))?;
        if actual_sha != expected_sha {
            return Err("source bundle checksum changed".to_string());
        }
    }
    let required: Vec<String> = receipt
        .get("required_objects")
        .and_then(|v| v.as_arr())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    object_check(&restored, &required)?;
    git(&restored, &["fsck", "--full"], None)?;
    Ok(())
}

pub fn preserve(r: &AuditState, custody: Option<&Path>) -> Result<Json, String> {
    if r.shallow {
        return Err("unique shallow source history requires separate preservation".to_string());
    }
    let base = match custody {
        Some(c) => c.to_path_buf(),
        None => default_custody()?,
    };
    if custody.is_none() {
        let mut ancestor = base.as_path();
        while !ancestor.exists() {
            ancestor = ancestor
                .parent()
                .ok_or_else(|| "invalid custody path".to_string())?;
        }
        let anc_dev = fs::metadata(ancestor)
            .map_err(|e| format!("stat custody ancestor failed: {e}"))?
            .dev();
        let root_dev = fs::metadata("/")
            .map_err(|e| format!("stat / failed: {e}"))?
            .dev();
        if anc_dev == root_dev {
            return Err("source custody provider missing or on root filesystem".to_string());
        }
    }
    fs::create_dir_all(&base).map_err(|e| format!("mkdir custody failed: {e}"))?;
    let key = &sha256_str(&r.common)[..24];
    let folder = base.join(key);
    fs::create_dir_all(&folder).map_err(|e| format!("mkdir custody folder failed: {e}"))?;
    let _lock = FlockGuard::acquire_nonblocking(&folder.join("preservation.lock"))
        .map_err(|_| "existing source-preservation worker is busy".to_string())?;
    preserve_locked(r, &base, &folder)
}

fn preserve_locked(r: &AuditState, base: &Path, folder: &Path) -> Result<Json, String> {
    let current = folder.join("receipt.json");
    let old = if current.exists() {
        Some(Json::load_file(&current).map_err(|e| format!("read receipt.json: {e}"))?)
    } else {
        None
    };

    if let Some(ref old_rec) = old {
        verify_chain(folder, old_rec)?;
        if object_check(&folder.join("verified.git"), &r.required_objects).is_ok() {
            return Ok(json_obj! {
                "status" => "preserved",
                "receipt" => current.to_string_lossy().to_string(),
                "required_objects" => r.required_objects.len(),
                "reused" => true,
            });
        }
    }

    let source_path = Path::new(&r.source);
    let mut size_kb: u64 = 0;
    let mut size_pack_kb: u64 = 0;
    for line in git(source_path, &["count-objects", "-v"], None)?.lines() {
        if let Some((field, val)) = line.split_once(':') {
            if let Ok(n) = val.trim().parse::<u64>() {
                match field.trim() {
                    "size" => size_kb = n,
                    "size-pack" => size_pack_kb = n,
                    _ => {}
                }
            }
        }
    }
    let estimated = 1024u64.saturating_mul(size_kb.saturating_add(size_pack_kb));
    let du_stats = disk_usage(base).map_err(|e| format!("statvfs custody: {e}"))?;
    let reserve = (4u64 * 1024 * 1024 * 1024).max(3u64.saturating_mul(estimated));
    let du_out = Command::new("du")
        .args(["-s", "-B1", "--", &base.to_string_lossy()])
        .output()
        .map_err(|e| format!("du custody failed: {e}"))?;
    if !du_out.status.success() || !du_out.stderr.is_empty() {
        return Err("source custody measurement incomplete".to_string());
    }
    let existing: u64 = String::from_utf8_lossy(&du_out.stdout)
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "source custody measurement parse error".to_string())?;
    let cap_bytes: u64 = 64 * 1024 * 1024 * 1024;
    let total_pool = du_stats.used_bytes.saturating_add(du_stats.available_bytes);
    if du_stats.available_bytes < reserve
        || existing.saturating_add(reserve) > cap_bytes
        || ((du_stats.used_bytes.saturating_add(reserve)) as f64) >= 0.85 * (total_pool as f64)
    {
        return Err("off-root source custody budget or filesystem floor insufficient".to_string());
    }

    let stamp = utc_stamp_micros();
    let assembly = folder.join(format!("assembly-{stamp}.git"));
    fs::create_dir_all(&assembly).map_err(|e| format!("mkdir assembly: {e}"))?;
    git(&assembly, &["init", "--bare"], None)?;

    let objects_dir = PathBuf::from(
        git(
            source_path,
            &["rev-parse", "--path-format=absolute", "--git-path", "objects"],
            None,
        )?
        .trim(),
    );
    let mut alt_lines = vec![objects_dir.to_string_lossy().to_string()];
    let src_alt = objects_dir.join("info/alternates");
    if src_alt.is_file() {
        if let Ok(content) = fs::read_to_string(&src_alt) {
            for line in content.lines() {
                let t = line.trim();
                if !t.is_empty() {
                    alt_lines.push(t.to_string());
                }
            }
        }
    }
    fs::write(
        assembly.join("objects/info/alternates"),
        format!("{}\n", alt_lines.join("\n")),
    )
    .map_err(|e| format!("write assembly alternates: {e}"))?;

    let mut refs_map = BTreeMap::new();
    for line in r.basic.refs.lines() {
        let mut parts = line.split_whitespace();
        if let (Some(sha), Some(refname)) = (parts.next(), parts.next()) {
            refs_map.insert(refname.to_string(), sha.to_string());
        }
    }
    let prefix = format!("refs/disk-steward-preservation-{}", stamp.replace('.', "-"));
    refs_map.insert(format!("{prefix}/head"), r.basic.head.clone());
    for sha in &r.basic.stashes {
        refs_map.insert(format!("{prefix}/stash/{sha}"), sha.clone());
    }
    for line in git(source_path, &["worktree", "list", "--porcelain"], None)?.lines() {
        if let Some(sha) = line.strip_prefix("HEAD ") {
            let sha = sha.trim();
            if !sha.is_empty() && !sha.chars().all(|c| c == '0') {
                refs_map.insert(format!("{prefix}/worktree/{sha}"), sha.to_string());
            }
        }
    }
    for sha in &r.unique_commits {
        refs_map.insert(format!("{prefix}/required/{sha}"), sha.clone());
    }

    let mut update_stdin = String::new();
    for (refname, sha) in &refs_map {
        update_stdin.push_str(&format!("create {refname} {sha}\n"));
    }
    git(&assembly, &["update-ref", "--stdin"], Some(&update_stdin))?;

    let mut bundles: Vec<Json> = old
        .as_ref()
        .and_then(|o| o.get("bundles"))
        .and_then(|b| b.as_arr())
        .cloned()
        .unwrap_or_default();

    let mut exclusions = String::new();
    if old.is_some() {
        let prev_shas: BTreeSet<String> = git(
            &folder.join("verified.git"),
            &["for-each-ref", "--format=%(objectname)"],
            None,
        )?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
        if !prev_shas.is_empty() {
            let check_in = format!(
                "{}\n",
                prev_shas.iter().cloned().collect::<Vec<_>>().join("\n")
            );
            let checked = git(&assembly, &["cat-file", "--batch-check"], Some(&check_in))?;
            for line in checked.lines() {
                if !line.contains(" missing") {
                    if let Some(sha) = line.split_whitespace().next() {
                        exclusions.push_str(&format!("^{sha}\n"));
                    }
                }
            }
        }
    }

    let bundle_name = format!("{stamp}.bundle");
    let bundle_path = folder.join(&bundle_name);
    let bundle_str = bundle_path.to_string_lossy().to_string();
    if !exclusions.is_empty() {
        git(
            &assembly,
            &["bundle", "create", &bundle_str, "--all", "--stdin"],
            Some(&exclusions),
        )?;
    } else {
        git(&assembly, &["bundle", "create", &bundle_str, "--all"], None)?;
    }

    let restored = folder.join("verified.git");
    if old.is_none() && !restored.exists() {
        fs::create_dir_all(&restored).map_err(|e| format!("mkdir verified.git: {e}"))?;
        git(&restored, &["init", "--bare"], None)?;
    }
    if restored.join("objects/info/alternates").exists() {
        return Err("independent restore must not share source objects".to_string());
    }
    git(&restored, &["bundle", "verify", &bundle_str], None)?;
    git(
        &restored,
        &["fetch", "--no-write-fetch-head", &bundle_str, "+refs/*:refs/*"],
        None,
    )?;

    let mut required_set: BTreeSet<String> = r.required_objects.iter().cloned().collect();
    if let Some(ref old_rec) = old {
        if let Some(arr) = old_rec.get("required_objects").and_then(|v| v.as_arr()) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    required_set.insert(s.to_string());
                }
            }
        }
    }
    let required: Vec<String> = required_set.into_iter().collect();
    object_check(&restored, &required)?;
    git(&restored, &["fsck", "--full"], None)?;

    let bundle_sha =
        sha256_file(&bundle_path).map_err(|e| format!("sha256 bundle failed: {e}"))?;
    let bundle_bytes = fs::metadata(&bundle_path)
        .map_err(|e| format!("stat bundle failed: {e}"))?
        .len();
    bundles.push(json_obj! {
        "name" => bundle_name,
        "sha256" => bundle_sha,
        "bytes" => bundle_bytes,
    });

    let receipt = json_obj! {
        "utc" => utc_iso(),
        "status" => "verified",
        "common" => r.common.clone(),
        "source" => r.source.clone(),
        "required_objects" => required.clone(),
        "required_sha256" => sha256_str(&required.join("\n")),
        "bundles" => Json::Arr(bundles),
        "independent_restore" => restored.to_string_lossy().to_string(),
        "restore_uses_alternates" => false,
        "stash_entries_in_snapshot" => r.basic.stashes.len(),
        "snapshot_refs" => r.basic.refs.clone(),
        "stash_order" => r.basic.stashes.clone(),
    };
    let receipt_text = format!("{}\n", receipt.to_pretty());
    atomic_write_str(&current, &receipt_text)
        .map_err(|e| format!("write receipt.json: {e}"))?;
    atomic_write_str(&folder.join(format!("receipt-{stamp}.json")), &receipt_text)
        .map_err(|e| format!("write stamped receipt: {e}"))?;

    // Clean up temporary bare assembly directory after verified fetch into verified.git
    let _ = fs::remove_dir_all(&assembly);

    Ok(json_obj! {
        "status" => "preserved",
        "receipt" => current.to_string_lossy().to_string(),
        "required_objects" => r.required_objects.len(),
        "reused" => false,
    })
}

fn audit_record_json(r: &AuditState, preservation: Option<&Json>) -> Json {
    let mut remote_tracking = BTreeMap::new();
    for line in r.basic.refs.lines() {
        let mut parts = line.split_whitespace();
        if let (Some(sha), Some(refname)) = (parts.next(), parts.next()) {
            if refname.starts_with("refs/remotes/") {
                remote_tracking.insert(refname.to_string(), Json::Str(sha.to_string()));
            }
        }
    }
    let unmatched_obj: BTreeMap<String, Json> = r
        .unmatched_tags
        .iter()
        .map(|(k, v)| (k.clone(), Json::Str(v.clone())))
        .collect();
    let remote_tags_obj: BTreeMap<String, Json> = r
        .remote_tags
        .iter()
        .map(|(k, v)| (k.clone(), Json::Str(v.clone())))
        .collect();

    let mut obj = json_obj! {
        "utc" => r.utc.clone(),
        "source" => r.source.clone(),
        "common" => r.common.clone(),
        "head" => r.basic.head.clone(),
        "status" => r.basic.status.clone(),
        "shallow" => r.shallow,
        "stashes" => r.basic.stashes.clone(),
        "published_tag_commits" => r.published_tag_commits.clone(),
        "unmatched_tags" => Json::Obj(unmatched_obj),
        "remote_tags" => Json::Obj(remote_tags_obj),
        "schema_version" => 2,
        "refs_count" => r.basic.refs.lines().count(),
        "refs_sha256" => sha256_str(&Json::Str(r.basic.refs.clone()).to_compact()),
        "local_only_commits_count" => r.local_only_commits.len(),
        "local_only_commits_sha256" => sha256_str(&Json::from(r.local_only_commits.clone()).to_compact()),
        "unique_commits_count" => r.unique_commits.len(),
        "unique_commits_sha256" => sha256_str(&Json::from(r.unique_commits.clone()).to_compact()),
        "required_objects_count" => r.required_objects.len(),
        "required_objects_sha256" => sha256_str(&Json::from(r.required_objects.clone()).to_compact()),
        "remote_tracking_refs" => Json::Obj(remote_tracking),
    };
    if let Some(p) = preservation {
        obj.insert("preservation", p.clone());
    }
    obj
}

pub fn ensure(
    source: &Path,
    out: &Path,
    custody: Option<&Path>,
) -> Result<EnsureRecord, String> {
    let resolved = fs::canonicalize(source)
        .map_err(|e| format!("cannot canonicalize {}: {e}", source.display()))?;
    let r = audit(&resolved)?;
    let name = format!("{}.source.json", sha256_str(&r.source));
    let path = out.join(name);
    atomic_write_str(&path, &format!("{}\n", audit_record_json(&r, None).to_pretty()))
        .map_err(|e| format!("write source audit: {e}"))?;
    if !r.basic.status.is_empty() {
        return Err("source checkout has uncommitted or untracked changes; retained".to_string());
    }
    let preserved = if !r.required_objects.is_empty() {
        preserve(&r, custody)?
    } else {
        json_obj! {
            "status" => "published_no_unique_objects",
            "required_objects" => 0,
        }
    };
    atomic_write_str(
        &path,
        &format!("{}\n", audit_record_json(&r, Some(&preserved)).to_pretty()),
    )
    .map_err(|e| format!("write source audit with preservation: {e}"))?;

    Ok(EnsureRecord {
        source: r.source,
        audit: path.to_string_lossy().to_string(),
        basic: r.basic,
        preservation: preserved,
    })
}

pub fn revalidate(record: &EnsureRecord) -> Result<(), String> {
    let current = basic(Path::new(&record.source))?;
    if current != record.basic {
        return Err("source status/refs/stashes changed after preservation".to_string());
    }
    Ok(())
}
