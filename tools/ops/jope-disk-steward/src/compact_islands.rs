//! Gene-first island archives: one streamed archive/GET per island, strict winner guard.
//! Native Rust replacement for `compact_islands.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cache_namespaces::is_symlink;
use crate::fleet_enforcer::{consumers, expand_glob, is_protected_str};
use crate::json::Json;
use crate::json_obj;
use crate::sha256::sha256_file;
use crate::sys::{
    atomic_write_str, base_dir, current_uid, emit_jsonl_locked, home_dir, hostname, now_nanos,
    utc_iso, FlockGuard,
};

const BUCKET: &str = "veox-jopedime-datasets-use2";
const DERIVED_SUFFIXES: &[&str] = &[
    ".step",
    ".stl",
    "_solid_mesh.npz",
    "_D1.json",
    "_surface.json",
    "_step_check.json",
    "_cad_eval.json",
    "_grind_law.json",
];

fn evo_root() -> PathBuf {
    home_dir()
        .join("jarambit_runs/evo_continuous")
        .join(hostname())
}

fn identity_vec(p: &Path) -> Result<Vec<Json>, String> {
    let s = fs::symlink_metadata(p).map_err(|e| format!("lstat {}: {e}", p.display()))?;
    Ok(vec![
        Json::from(s.dev()),
        Json::from(s.ino()),
        Json::from(s.size()),
        Json::from(s.mtime_nsec()),
        Json::from(s.ctime_nsec()),
        Json::from(s.uid()),
        Json::from(s.nlink()),
    ])
}

fn derivative_label(p: &Path) -> Option<String> {
    let name = p.file_name()?.to_str()?;
    for suf in DERIVED_SUFFIXES {
        if let Some(stem) = name.strip_suffix(suf) {
            return Some(stem.to_string());
        }
    }
    None
}

fn primary_record(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower.ends_with("_genome.csv")
        || matches!(
            lower.as_str(),
            "island.json" | "seed_scores.tsv" | "leaderboard.json" | "leaderboard.md"
        )
        || [
            "best_so_far",
            "master_scorecard",
            "runtime",
            "_rs_",
            "rs_v2",
            "round_",
        ]
        .iter()
        .any(|tok| lower.contains(tok))
}

fn seed_source(raw: &str, root: &Path) -> Option<(PathBuf, Option<Json>)> {
    let p = PathBuf::from(raw);
    if p.is_file() && !is_symlink(&p) {
        return Some((p, None));
    }
    let parent_evo = root.parent()?;
    if p.parent() == Some(&parent_evo.join("seeds_bootstrap")) {
        let old = parent_evo.join("seeds_bootstrap.pre-canonical-20261022");
        let sums = old.join("SHA256SUMS");
        let name = p.file_name()?;
        let candidate = old.join(name);
        if sums.is_file() && candidate.is_file() && !is_symlink(&candidate) {
            let text = fs::read_to_string(&sums).ok()?;
            let mut expected = BTreeMap::new();
            for line in text.lines() {
                let mut parts = line.split_whitespace();
                if let (Some(digest), Some(fname)) = (parts.next(), parts.next()) {
                    expected.insert(fname.trim_start_matches('*').to_string(), digest.to_string());
                }
            }
            let name_str = name.to_str()?;
            if let Some(exp_sha) = expected.get(name_str) {
                if sha256_file(&candidate).ok().as_ref() == Some(exp_sha) {
                    let sums_sha = sha256_file(&sums).ok()?;
                    return Some((
                        candidate,
                        Some(json_obj! {
                            "requested_path" => raw,
                            "source_checksum_manifest" => sums.to_string_lossy().to_string(),
                            "source_checksum_manifest_sha256" => sums_sha,
                            "expected_sha256" => exp_sha.clone(),
                        }),
                    ));
                }
            }
        }
    }
    None
}

fn visit_best_so_far(v: &Json, labels: &mut BTreeSet<String>, hashes: &mut BTreeSet<String>) {
    match v {
        Json::Obj(m) => {
            if let Some(Json::Str(lbl)) = m.get("label") {
                labels.insert(lbl.clone());
            }
            if let Some(Json::Str(gsha)) = m.get("genome_sha256") {
                hashes.insert(gsha.clone());
            }
            for child in m.values() {
                visit_best_so_far(child, labels, hashes);
            }
        }
        Json::Arr(a) => {
            for child in a {
                visit_best_so_far(child, labels, hashes);
            }
        }
        _ => {}
    }
}

fn winners(root: &Path) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    let pool = root.join("leaders.pool");
    let before = identity_vec(&pool)?;
    let text = fs::read_to_string(&pool).map_err(|e| format!("read leaders.pool: {e}"))?;
    if text.trim().is_empty() {
        return Err("empty/transient leader pool".to_string());
    }
    let mut paths = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 2 || !fields[1].ends_with("_genome.csv") {
            return Err("malformed pool".to_string());
        }
        let p = PathBuf::from(fields[1]);
        if !p.is_file() || is_symlink(&p) {
            return Err("missing leader genome".to_string());
        }
        paths.push(p);
    }
    if identity_vec(&pool)? != before {
        return Err("leader pool changed while reading".to_string());
    }
    for p in expand_glob(&root.join("leaders"), "*") {
        if p.file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.ends_with("_genome.csv"))
        {
            paths.push(p);
        }
    }
    let mut labels = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    for p in paths {
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if let Some(lbl) = name.strip_suffix("_genome.csv") {
                labels.insert(lbl.to_string());
            }
        }
        let h = sha256_file(&p).map_err(|e| format!("hash leader {}: {e}", p.display()))?;
        hashes.insert(h);
    }

    for island_dir in expand_glob(&root.join("islands"), "i*") {
        if let Ok(rd) = fs::read_dir(&island_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if fname.starts_with("best_so_far") && fname.ends_with(".json") {
                    if let Ok(j) = Json::load_file(&p) {
                        visit_best_so_far(&j, &mut labels, &mut hashes);
                    }
                }
            }
        }
        let island_json = island_dir.join("island.json");
        if let Ok(cfg) = Json::load_file(&island_json) {
            for raw in cfg.get_str("seeds").unwrap_or("").split(',') {
                if !raw.is_empty() {
                    if let Some((seed_p, _)) = seed_source(raw, root) {
                        if let Ok(h) = sha256_file(&seed_p) {
                            hashes.insert(h);
                        }
                        if let Some(lbl) = seed_p
                            .file_name()
                            .and_then(|s| s.to_str())
                            .and_then(|s| s.strip_suffix("_genome.csv"))
                        {
                            labels.insert(lbl.to_string());
                        }
                    }
                }
            }
        }
    }
    Ok((labels, hashes))
}

fn is_protected_candidate(
    p: &Path,
    root: &Path,
    labels: &BTreeSet<String>,
    hashes: &BTreeSet<String>,
) -> bool {
    if let Ok(rel) = p.strip_prefix(root) {
        if is_protected_str(&rel.to_string_lossy()) {
            return true;
        }
    }
    let Some(label) = derivative_label(p) else {
        return true;
    };
    if labels.contains(&label) {
        return true;
    }
    let Some(parent) = p.parent() else {
        return true;
    };
    let gene = parent.join(format!("{label}_genome.csv"));
    if !gene.is_file() {
        return true;
    }
    match sha256_file(&gene) {
        Ok(h) => hashes.contains(&h),
        Err(_) => true,
    }
}

fn collect_island_files(island: &Path) -> Result<Vec<Json>, String> {
    let root_dev = fs::metadata(island)
        .map_err(|e| format!("stat island: {e}"))?
        .dev();
    let mut rows = Vec::new();
    let mut stack = vec![island.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = fs::read_dir(&dir).map_err(|e| format!("readdir {}: {e}", dir.display()))?;
        for entry in rd {
            let entry = entry.map_err(|e| format!("direntry: {e}"))?;
            let p = entry.path();
            let s = fs::symlink_metadata(&p).map_err(|e| format!("lstat {}: {e}", p.display()))?;
            if s.file_type().is_symlink() {
                return Err(format!("symlink in island: {}", p.display()));
            }
            if s.file_type().is_dir() {
                stack.push(p);
                continue;
            }
            let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if fname == "RETENTION_RECEIPT.json" {
                continue;
            }
            if !s.file_type().is_file() || s.dev() != root_dev {
                return Err(format!("nonregular/off-device file: {}", p.display()));
            }
            let rel = p
                .strip_prefix(island)
                .map_err(|_| "strip_prefix failed".to_string())?
                .to_string_lossy()
                .to_string();
            rows.push(json_obj! {
                "path" => p.to_string_lossy().to_string(),
                "member" => rel,
                "identity" => Json::Arr(identity_vec(&p)?),
            });
        }
    }
    rows.sort_by(|a, b| {
        a.get_str("member")
            .unwrap_or("")
            .cmp(b.get_str("member").unwrap_or(""))
    });
    Ok(rows)
}

fn archive_island(island: &Path, rows: &[Json], config: &Json, out: &Path, root: &Path) -> Result<Json, String> {
    let seeds_str = config.get_str("seeds").unwrap_or("");
    let missing_seeds: Vec<String> = seeds_str
        .split(',')
        .filter(|r| !r.is_empty() && seed_source(r, root).is_none())
        .map(|s| s.to_string())
        .collect();

    let staging = out.join("archive_staging");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging).map_err(|e| format!("mkdir staging: {e}"))?;

    let mut manifest = Vec::new();
    for row in rows {
        let p_str = row.get_str("path").unwrap_or("");
        let member = row.get_str("member").unwrap_or("");
        let p = PathBuf::from(p_str);
        if Some(&Json::Arr(identity_vec(&p)?)) != row.get("identity") {
            return Err("file changed before archive".to_string());
        }
        let digest = sha256_file(&p).map_err(|e| format!("hash {}: {e}", p.display()))?;
        let dst = staging.join(member);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir staging parent: {e}"))?;
        }
        fs::copy(&p, &dst).map_err(|e| format!("copy to staging: {e}"))?;
        if Some(&Json::Arr(identity_vec(&p)?)) != row.get("identity") {
            return Err("file changed during archive".to_string());
        }
        let mut m_row = row.clone();
        m_row.insert("sha256", digest);
        manifest.push(m_row);
    }

    let mut seed_records = Vec::new();
    let gene_seeds_dir = staging.join("GENE_SEEDS");
    fs::create_dir_all(&gene_seeds_dir).map_err(|e| format!("mkdir GENE_SEEDS: {e}"))?;
    for raw in seeds_str.split(',') {
        if raw.is_empty() || missing_seeds.iter().any(|m| m == raw) {
            continue;
        }
        let Some((p, source_proof)) = seed_source(raw, root) else {
            return Err("seed input changed".to_string());
        };
        let old_id = identity_vec(&p)?;
        let digest = sha256_file(&p).map_err(|e| format!("hash seed: {e}"))?;
        let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("seed.csv");
        let member = format!("GENE_SEEDS/{digest}_{fname}");
        fs::copy(&p, staging.join(&member)).map_err(|e| format!("copy seed: {e}"))?;
        if identity_vec(&p)? != old_id {
            return Err("seed input changed during copy".to_string());
        }
        seed_records.push(json_obj! {
            "path" => p.to_string_lossy().to_string(),
            "requested_path" => raw,
            "sha256" => digest,
            "member" => member,
            "historical_source_proof" => source_proof.unwrap_or(Json::Null),
        });
    }

    let retention_manifest = json_obj! {
        "config" => config.clone(),
        "files" => Json::Arr(manifest.clone()),
        "seed_inputs" => Json::Arr(seed_records.clone()),
        "missing_seed_inputs" => missing_seeds.clone(),
    };
    fs::write(
        staging.join("RETENTION_MANIFEST.json"),
        format!("{}\n", retention_manifest.to_pretty()),
    )
    .map_err(|e| format!("write RETENTION_MANIFEST.json: {e}"))?;

    let tar_path = out.join("archive.tar.gz");
    let tar_status = Command::new("tar")
        .args([
            "-czf",
            &tar_path.to_string_lossy(),
            "-C",
            &staging.to_string_lossy(),
            ".",
        ])
        .status()
        .map_err(|e| format!("tar failed: {e}"))?;
    let _ = fs::remove_dir_all(&staging);
    if !tar_status.success() {
        return Err("tar creation failed".to_string());
    }

    let body_sha = sha256_file(&tar_path).map_err(|e| format!("sha256 tar: {e}"))?;
    let size = fs::metadata(&tar_path)
        .map_err(|e| format!("stat tar: {e}"))?
        .len();
    let island_name = island.file_name().and_then(|s| s.to_str()).unwrap_or("i0");
    let key = format!(
        "jarambit/evo_continuous/{}/islands/{}/retention/{:032x}.tar.gz",
        hostname(),
        island_name,
        now_nanos()
    );
    let aws_bin = home_dir().join(".local/bin/aws");
    let s3_uri = format!("s3://{BUCKET}/{key}");

    let up_out = Command::new(&aws_bin)
        .args(["s3", "cp", &tar_path.to_string_lossy(), &s3_uri, "--only-show-errors"])
        .env("AWS_PROFILE", "veox-jepson")
        .env("AWS_PAGER", "")
        .output()
        .map_err(|e| format!("aws s3 cp: {e}"))?;
    let _ = fs::write(out.join("upload.stdout"), &up_out.stdout);
    let _ = fs::write(out.join("upload.stderr"), &up_out.stderr);
    let _ = fs::remove_file(&tar_path);
    if !up_out.status.success() || !up_out.stderr.is_empty() {
        return Err("aws s3 cp upload failed".to_string());
    }

    let head_out = Command::new(&aws_bin)
        .args([
            "s3api",
            "head-object",
            "--bucket",
            BUCKET,
            "--key",
            &key,
            "--output",
            "json",
        ])
        .env("AWS_PROFILE", "veox-jepson")
        .env("AWS_PAGER", "")
        .output()
        .map_err(|e| format!("aws head-object: {e}"))?;
    if !head_out.status.success() {
        return Err("aws head-object failed".to_string());
    }
    let head_json =
        Json::parse(&String::from_utf8_lossy(&head_out.stdout)).map_err(|e| format!("parse head: {e}"))?;
    let version = head_json
        .get_str("VersionId")
        .ok_or_else(|| "missing VersionId".to_string())?
        .to_string();
    if version == "null" || head_json.get_u64("ContentLength") != Some(size) {
        return Err("S3 head-object version or size mismatch".to_string());
    }

    let verify_file = out.join("verify_download.tar.gz");
    let get_out = Command::new(&aws_bin)
        .args([
            "s3api",
            "get-object",
            "--bucket",
            BUCKET,
            "--key",
            &key,
            "--version-id",
            &version,
            &verify_file.to_string_lossy(),
            "--output",
            "json",
        ])
        .env("AWS_PROFILE", "veox-jepson")
        .env("AWS_PAGER", "")
        .output()
        .map_err(|e| format!("aws get-object: {e}"))?;
    let _ = fs::write(out.join("get.json"), &get_out.stdout);
    let _ = fs::write(out.join("get.stderr"), &get_out.stderr);
    let verify_sha = sha256_file(&verify_file).unwrap_or_default();
    let verify_len = fs::metadata(&verify_file).map(|m| m.len()).unwrap_or(0);
    let _ = fs::remove_file(&verify_file);
    if !get_out.status.success() || verify_len != size || verify_sha != body_sha {
        return Err("S3 get-object verification mismatch".to_string());
    }

    let proof = json_obj! {
        "bucket" => BUCKET,
        "key" => key,
        "version_id" => version,
        "bytes" => size,
        "sha256" => body_sha,
        "files" => Json::Arr(manifest),
        "seed_inputs" => Json::Arr(seed_records),
        "missing_seed_inputs" => missing_seeds,
        "full_GET_verified" => true,
    };
    atomic_write_str(&out.join("archive.json"), &format!("{}\n", proof.to_pretty()))
        .map_err(|e| format!("write archive.json: {e}"))?;
    Ok(proof)
}

fn compact_one(island: &Path, root: &Path, base: &Path) -> Result<Json, String> {
    let island_name = island
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let config =
        Json::load_file(&island.join("island.json")).map_err(|e| format!("read island.json: {e}"))?;
    let rows = collect_island_files(island)?;
    let receipt_path = island.join("RETENTION_RECEIPT.json");
    let prior = if receipt_path.exists() {
        Some(Json::load_file(&receipt_path).map_err(|e| format!("read receipt: {e}"))?)
    } else {
        None
    };

    let (proof, out) = if let Some(ref pr) = prior {
        let proof_path = PathBuf::from(pr.get_str("archive_proof").unwrap_or(""));
        let actual_sha = sha256_file(&proof_path).map_err(|e| format!("hash proof: {e}"))?;
        if Some(actual_sha.as_str()) != pr.get_str("archive_proof_sha256") {
            return Err("archive_proof_sha256 mismatch".to_string());
        }
        let loaded_proof = Json::load_file(&proof_path).map_err(|e| format!("load proof: {e}"))?;
        if loaded_proof.get_bool("full_GET_verified") != Some(true) {
            return Err("prior proof not full_GET_verified".to_string());
        }
        let proof_files = loaded_proof
            .get("files")
            .and_then(|v| v.as_arr())
            .cloned()
            .unwrap_or_default();
        for pf in &proof_files {
            let p = PathBuf::from(pf.get_str("path").unwrap_or(""));
            if primary_record(&p) && !p.is_file() {
                return Err("primary local gene/score record missing; archive retained, no pruning".to_string());
            }
        }
        if pr.get_str("status") == Some("complete") && pr.get_u64("nonwinner_candidates") == Some(0)
        {
            return Ok(json_obj! {
                "island" => island_name,
                "status" => "complete",
                "deleted_files" => 0u64,
                "bytes_freed" => 0u64,
                "nonwinner_candidates" => 0u64,
            });
        }
        (
            loaded_proof,
            proof_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| base.join("state/compaction")),
        )
    } else {
        let out_dir = base
            .join("state/compaction")
            .join(format!("{}-{}-{:032x}", hostname(), island_name, now_nanos()));
        fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir out: {e}"))?;
        let prf = archive_island(island, &rows, &config, &out_dir, root)?;
        if collect_island_files(island)? != rows {
            return Err("island changed during archive".to_string());
        }
        (prf, out_dir)
    };

    let (labels, hashes) = winners(root)?;
    let proof_files = proof
        .get("files")
        .and_then(|v| v.as_arr())
        .cloned()
        .unwrap_or_default();
    let mut candidates = Vec::new();
    for r in &proof_files {
        let p = PathBuf::from(r.get_str("path").unwrap_or(""));
        if p.exists()
            && derivative_label(&p).is_some()
            && !is_protected_candidate(&p, root, &labels, &hashes)
        {
            candidates.push(r.clone());
        }
    }

    let missing_seeds_empty = proof
        .get("missing_seed_inputs")
        .and_then(|v| v.as_arr())
        .map(|a| a.is_empty())
        .unwrap_or(true);
    let mut eligible = Vec::new();
    let uid = current_uid();
    let now_s = crate::sys::now_epoch_secs();
    if missing_seeds_empty {
        for r in &candidates {
            let p = PathBuf::from(r.get_str("path").unwrap_or(""));
            if let Ok(s) = fs::symlink_metadata(&p) {
                let latest = s.mtime().max(s.ctime());
                if s.uid() == uid && s.nlink() == 1 && now_s - latest >= 21600 {
                    eligible.push(r.clone());
                }
            }
        }
    }

    let (probe, mut consumer_hold) = if !eligible.is_empty() {
        let prb = consumers(&[island.to_path_buf()], None);
        let hold = prb.get_bool("complete") != Some(true)
            || prb
                .get("consumers")
                .and_then(|c| c.get(&island.to_string_lossy()))
                .and_then(|v| v.as_arr())
                .map(|a| !a.is_empty())
                .unwrap_or(true);
        (prb, hold)
    } else {
        (
            json_obj! {
                "status" => "not_required_no_deletion_eligible_derivatives",
                "complete" => false,
                "errors" => Json::empty_arr(),
                "consumers" => Json::empty_obj(),
            },
            false,
        )
    };

    let archive_json_path = out.join("archive.json");
    let archive_proof_sha =
        sha256_file(&archive_json_path).map_err(|e| format!("hash archive.json: {e}"))?;
    let s3_uri = format!("s3://{BUCKET}/{}", proof.get_str("key").unwrap_or(""));
    let mut receipt = json_obj! {
        "utc" => utc_iso(),
        "host" => hostname(),
        "island" => island_name.clone(),
        "status" => "archived_verified",
        "archive_proof" => archive_json_path.to_string_lossy().to_string(),
        "archive_proof_sha256" => archive_proof_sha,
        "s3_uri" => s3_uri.clone(),
        "version_id" => proof.get_str("version_id").unwrap_or(""),
        "archive_sha256" => proof.get_str("sha256").unwrap_or(""),
        "archive_bytes" => proof.get_u64("bytes").unwrap_or(0),
        "gene_commit" => config.get_str("git_sha").unwrap_or(""),
        "binary_sha256" => config.get_str("binary_sha256").unwrap_or(""),
        "seed" => config.get("seed").cloned().unwrap_or(Json::Null),
        "nonwinner_candidates" => candidates.len(),
        "winner_protection" => "leaders.pool CSV bytes+labels, leaders/, all island best_so_far, seed parents; ambiguous genotype KEEP",
        "local_genes_scores" => "always retained",
        "initial_consumer_probe" => probe,
        "consumer_hold" => consumer_hold,
    };
    atomic_write_str(&receipt_path, &format!("{}\n", receipt.to_pretty()))
        .map_err(|e| format!("write stage receipt: {e}"))?;

    let mut freed: u64 = 0;
    let mut count: usize = 0;
    for row in &eligible {
        if consumer_hold || !missing_seeds_empty {
            break;
        }
        let p = PathBuf::from(row.get_str("path").unwrap_or(""));
        let Ok(s) = fs::symlink_metadata(&p) else {
            continue;
        };
        if s.uid() != uid || s.nlink() != 1 || crate::sys::now_epoch_secs() - s.mtime().max(s.ctime()) < 21600 {
            continue;
        }
        let fresh = consumers(&[island.to_path_buf()], None);
        if fresh.get_bool("complete") != Some(true)
            || fresh
                .get("consumers")
                .and_then(|c| c.get(&island.to_string_lossy()))
                .and_then(|v| v.as_arr())
                .map(|a| !a.is_empty())
                .unwrap_or(true)
        {
            consumer_hold = true;
            break;
        }
        let (cur_labels, cur_hashes) = match winners(root) {
            Ok(w) => w,
            Err(e) => {
                receipt.insert("prune_warning", e);
                break;
            }
        };
        if is_protected_candidate(&p, root, &cur_labels, &cur_hashes) {
            continue;
        }
        let cur_id = identity_vec(&p).unwrap_or_default();
        let cur_sha = sha256_file(&p).unwrap_or_default();
        if Some(&Json::Arr(cur_id)) != row.get("identity")
            || Some(cur_sha.as_str()) != row.get_str("sha256")
        {
            receipt.insert("prune_warning", "file identity or bytes changed; retained");
            break;
        }
        let alloc = s.blocks().saturating_mul(512);
        let _ = emit_jsonl_locked(
            &out.join("deletions.jsonl"),
            &json_obj! {
                "utc" => utc_iso(),
                "path" => p.to_string_lossy().to_string(),
                "event" => "unlink_intent",
                "bytes" => alloc,
                "sha256" => cur_sha.clone(),
            }
            .to_compact(),
        );
        if fs::remove_file(&p).is_ok() {
            freed = freed.saturating_add(alloc);
            count += 1;
            let _ = emit_jsonl_locked(
                &out.join("deletions.jsonl"),
                &json_obj! {
                    "utc" => utc_iso(),
                    "path" => p.to_string_lossy().to_string(),
                    "event" => "deleted",
                    "bytes" => alloc,
                    "sha256" => cur_sha,
                }
                .to_compact(),
            );
        }
    }

    let final_status = if !missing_seeds_empty {
        "archived_seed_inputs_missing"
    } else if consumer_hold {
        "archived_consumer_hold"
    } else if receipt.get("prune_warning").is_some() {
        "archived_guard_hold"
    } else {
        "complete"
    };
    receipt.insert("status", final_status);
    receipt.insert(
        "missing_seed_inputs",
        proof
            .get("missing_seed_inputs")
            .cloned()
            .unwrap_or_else(Json::empty_arr),
    );
    receipt.insert("deleted_files", count);
    receipt.insert("bytes_freed", freed);
    let _ = atomic_write_str(&receipt_path, &format!("{}\n", receipt.to_pretty()));
    let _ = atomic_write_str(
        &out.join(format!("result-{}.json", now_nanos())),
        &format!("{}\n", receipt.to_pretty()),
    );
    if count > 0 {
        let _ = emit_jsonl_locked(
            &base.join("DELETED.log"),
            &json_obj! {
                "utc" => utc_iso(),
                "host" => hostname(),
                "path" => island.to_string_lossy().to_string(),
                "event" => "island_compaction_batch",
                "files" => count,
                "bytes" => freed,
                "receipt" => receipt_path.to_string_lossy().to_string(),
                "s3_uri" => s3_uri,
                "version_id" => proof.get_str("version_id").unwrap_or(""),
                "sha256" => proof.get_str("sha256").unwrap_or(""),
            }
            .to_compact(),
        );
    }
    Ok(json_obj! {
        "island" => island_name,
        "status" => final_status,
        "deleted_files" => count,
        "bytes_freed" => freed,
        "nonwinner_candidates" => candidates.len(),
    })
}

pub fn run_compact_islands() -> i32 {
    let root = evo_root();
    if !root.is_dir() {
        return 0;
    }
    let base = base_dir();
    if base.join("state/compaction-paused.json").is_file() {
        println!(
            "{}",
            json_obj! {
                "status" => "maintenance_compaction_paused",
                "note" => "islands and research jobs unchanged; priority backfill on xbabe3",
            }
            .to_compact()
        );
        return 0;
    }
    let Ok(_lock) = FlockGuard::acquire_nonblocking(&base.join("compaction.lock")) else {
        return 0;
    };
    let mut results = Vec::new();
    for island in expand_glob(&root.join("islands"), "i*") {
        if !island.join("EXIT").is_file() {
            continue;
        }
        let r = match compact_one(&island, &root, &base) {
            Ok(res) => res,
            Err(e) => json_obj! {
                "island" => island.file_name().and_then(|s| s.to_str()).unwrap_or("unknown"),
                "status" => "held",
                "error" => e,
            },
        };
        println!("{}", r.to_compact());
        results.push(r);
        let _ = atomic_write_str(
            &base.join("state/compaction-latest.json"),
            &format!(
                "{}\n",
                json_obj! {
                    "utc" => utc_iso(),
                    "host" => hostname(),
                    "results" => Json::Arr(results.clone()),
                }
                .to_pretty()
            ),
        );
    }
    0
}
