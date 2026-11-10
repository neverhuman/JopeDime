//! Apply only a recorded, explicit, one-plan user exception for inherited references.
//! Native Rust replacement for `inherited_reference_policy.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, Metadata};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::json::Json;
use crate::json_obj;
use crate::sha256::{sha256_file, sha256_str};

fn str_val(j: &Json, key: &str) -> String {
    match j.get(key) {
        Some(Json::Str(s)) => s.clone(),
        Some(Json::Int(n)) => n.to_string(),
        Some(Json::UInt(n)) => n.to_string(),
        _ => String::new(),
    }
}

fn inherited_only(root: &str, current: &Json, expected: &Json) -> bool {
    let state = current.get_str("state").unwrap_or("");
    if !matches!(state, "R" | "S") {
        return false;
    }
    for key in ["pid", "start_ticks", "exe", "comm"] {
        if str_val(current, key) != str_val(expected, key) {
            return false;
        }
    }
    let cwd_raw = current.get_str("cwd").unwrap_or("");
    let cwd = cwd_raw.strip_suffix(" (deleted)").unwrap_or(cwd_raw);
    let root_prefix = format!("{}/", root.trim_end_matches('/'));
    if cwd == root || cwd.starts_with(&root_prefix) {
        return false;
    }
    let mut permitted: BTreeSet<&str> = ["cwd", "env:CARGO_TARGET_DIR"].into_iter().collect();
    if expected.get_str("comm") == Some("sccache") {
        permitted.insert("env:LD_LIBRARY_PATH");
    }
    let reasons: BTreeSet<&str> = current
        .get("reasons")
        .and_then(|v| v.as_arr())
        .map(|arr| arr.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    let expected_reasons: BTreeSet<&str> = expected
        .get("reasons")
        .and_then(|v| v.as_arr())
        .map(|arr| arr.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    !reasons.is_empty() && reasons.is_subset(&permitted) && reasons.is_subset(&expected_reasons)
}

pub fn decorate(probe: Json, base: &Path) -> Json {
    let policy_path = base.join("state/inherited-reference-approval.json");
    if !policy_path.is_file() {
        return probe;
    }
    let mut original = probe.clone();
    let mut updated = probe;

    let res: Result<Json, String> = (|| {
        let policy =
            Json::load_file(&policy_path).map_err(|e| format!("read policy JSON: {e}"))?;
        if policy.get_bool("approved") != Some(true)
            || policy.get_str("approval_ref").unwrap_or("").is_empty()
        {
            return Err("policy not approved".to_string());
        }
        let plan_str = policy
            .get_str("plan")
            .ok_or_else(|| "missing plan path".to_string())?;
        let plan_path = PathBuf::from(plan_str);
        let actual_plan_sha =
            sha256_file(&plan_path).map_err(|e| format!("hash plan file: {e}"))?;
        if Some(actual_plan_sha.as_str()) != policy.get_str("plan_sha256") {
            return Err("plan_sha256 mismatch".to_string());
        }
        let plan = Json::load_file(&plan_path).map_err(|e| format!("read plan JSON: {e}"))?;
        let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .map_err(|e| format!("read boot_id: {e}"))?;
        if plan.get_str("boot_id") != Some(boot_id.trim()) {
            return Err("boot_id mismatch".to_string());
        }
        let probe_errors_empty = plan
            .get("probe_errors")
            .and_then(|v| v.as_arr())
            .map(|a| a.is_empty())
            .unwrap_or(false);
        if plan.get_bool("probe_complete") != Some(true) || !probe_errors_empty {
            return Err("plan probe incomplete".to_string());
        }
        let approved_roots: BTreeSet<String> = policy
            .get("approved_roots")
            .and_then(|v| v.as_arr())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let manifest_shas = policy
            .get("manifest_sha256")
            .and_then(|v| v.as_obj())
            .ok_or_else(|| "missing manifest_sha256 map".to_string())?;
        let plan_roots = plan
            .get("roots")
            .and_then(|v| v.as_arr())
            .ok_or_else(|| "missing plan roots".to_string())?;

        let mut contexts = BTreeMap::new();
        for item in plan_roots {
            let Some(root) = item.get_str("path") else {
                continue;
            };
            if !approved_roots.contains(root) {
                continue;
            }
            let has_consumer_entry = updated
                .get("consumers")
                .and_then(|c| c.as_obj())
                .map(|m| m.contains_key(root))
                .unwrap_or(false);
            if !has_consumer_entry {
                continue;
            }
            let plan_parent = plan_path.parent().unwrap_or_else(|| Path::new("."));
            let manifest = plan_parent.join(format!("{}.selected.jsonl", sha256_str(root)));
            let actual_m_sha =
                sha256_file(&manifest).map_err(|e| format!("hash manifest: {e}"))?;
            let exp_m_sha = manifest_shas
                .get(root)
                .and_then(|v| v.as_str())
                .ok_or_else(|| "missing manifest_sha256 entry".to_string())?;
            if actual_m_sha != exp_m_sha {
                return Err("manifest_sha256 mismatch".to_string());
            }

            let mut expected_by_pid = BTreeMap::new();
            if let Some(claims) = item.get("process_claims").and_then(|v| v.as_arr()) {
                for p in claims {
                    if let Some(pid) = p.get("pid").and_then(|v| v.as_i64()) {
                        expected_by_pid.insert(pid, p.clone());
                    }
                }
            }

            let current_claims = updated
                .get("consumers")
                .and_then(|c| c.get(root))
                .and_then(|v| v.as_arr())
                .cloned()
                .unwrap_or_default();
            let mut active = Vec::new();
            let mut ignored = BTreeSet::new();
            for claim in current_claims {
                let pid = claim.get("pid").and_then(|v| v.as_i64()).unwrap_or(-1);
                if let Some(old) = expected_by_pid.get(&pid) {
                    if inherited_only(root, &claim, old) {
                        ignored.insert(pid);
                        continue;
                    }
                }
                active.push(claim);
            }
            if !ignored.is_empty() {
                if let Some(consumers_map) =
                    updated.get_mut_consumers()
                {
                    consumers_map.insert(root.to_string(), Json::Arr(active));
                }
                let ignored_vec: Vec<i64> = ignored.into_iter().collect();
                contexts.insert(
                    root.to_string(),
                    json_obj! {
                        "approval_ref" => policy.get_str("approval_ref").unwrap_or(""),
                        "plan_sha256" => policy.get_str("plan_sha256").unwrap_or(""),
                        "manifest" => manifest.to_string_lossy().to_string(),
                        "manifest_sha256" => exp_m_sha,
                        "ignored_pids" => ignored_vec,
                        "retained_source_commit" => item.get("retained_source_commit").cloned().unwrap_or(Json::Null),
                    },
                );
            }
        }
        updated.insert("approved_reference_exceptions", Json::Obj(contexts));
        Ok(updated)
    })();

    match res {
        Ok(u) => u,
        Err(e) => {
            original.insert("exception_warning", Json::Str(e));
            original
        }
    }
}

impl Json {
    fn get_mut_consumers(&mut self) -> Option<&mut BTreeMap<String, Json>> {
        self.as_obj_mut()?.get_mut("consumers")?.as_obj_mut()
    }
}

pub fn restrict_groups(
    groups: Vec<Vec<(PathBuf, Metadata)>>,
    context: &Json,
) -> Result<Vec<Vec<(PathBuf, Metadata)>>, String> {
    let manifest_str = context
        .get_str("manifest")
        .ok_or_else(|| "missing manifest in context".to_string())?;
    let path = PathBuf::from(manifest_str);
    let actual_sha = sha256_file(&path).map_err(|e| format!("hash manifest: {e}"))?;
    if Some(actual_sha.as_str()) != context.get_str("manifest_sha256") {
        return Err("manifest_sha256 changed".to_string());
    }
    let text = fs::read_to_string(&path).map_err(|e| format!("read manifest: {e}"))?;
    let mut expected = BTreeMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row = Json::parse(line).map_err(|e| format!("parse manifest line: {e}"))?;
        if let Some(p) = row.get_str("path") {
            expected.insert(p.to_string(), row);
        }
    }

    let mut filtered = Vec::new();
    for g in groups {
        let all_match = g.iter().all(|(p, s)| {
            let Some(row) = expected.get(&p.to_string_lossy().to_string()) else {
                return false;
            };
            row.get_u64("device") == Some(s.dev())
                && row.get_u64("inode") == Some(s.ino())
                && row.get_u64("size") == Some(s.size())
                && row.get("mtime_ns").and_then(|v| v.as_i64()) == Some(s.mtime_nsec())
                && row.get("ctime_ns").and_then(|v| v.as_i64()) == Some(s.ctime_nsec())
                && row.get_u64("links") == Some(s.nlink())
        });
        if all_match {
            filtered.push(g);
        }
    }
    Ok(filtered)
}
