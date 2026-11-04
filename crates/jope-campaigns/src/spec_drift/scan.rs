use super::audit::policy_value;
use super::*;

/// Scan one source file for render sites.
pub(super) fn scan_file(repo_root: &Path, rel: &str) -> std::io::Result<Vec<RenderSite>> {
    let full: PathBuf = repo_root.join(rel);
    let raw = std::fs::read_to_string(&full)?;
    if !raw.contains("[resource]") && !raw.contains("render_resource_block()") {
        return Ok(vec![]);
    }
    let is_py = rel.ends_with(".py");
    let body = if is_py {
        raw.as_str()
    } else {
        cut_test_module(&raw)
    };
    let lines = strip_comments(body, is_py);

    let mut out = vec![];
    for (i, ln) in lines.iter().enumerate() {
        let central_policy = ln.contains("render_resource_block()");
        if !ln.contains("[resource]") && !central_policy {
            continue;
        }
        // Unescape literal `\n` before extracting values. Several renderers emit
        // an ENTIRE spec as one single-line `format!` string
        // (`"[job]\nid = ...\n\n[resource]\ngpus_per_task = 0\n..."`), so every
        // key is preceded by the two characters `\` `n`. `find_value` rejects a
        // hit whose preceding byte is alphanumeric -- the guard that stops
        // `packed_ram_mb` matching `ram_mb` -- and `n` is alphanumeric, so every
        // key in those specs was silently skipped and the site reported `params
        // {}`. Turning the escape into a real newline makes the one-line and
        // multi-line styles parse identically. Line numbers are unaffected: they
        // come from the physical index `i`, not from this window.
        let unescape = |s: String| s.replace("\\n", "\n");
        let hi = (i + 60).min(lines.len());
        let window = unescape(lines[i..hi].join("\n"));
        let hi2 = (i + 100).min(lines.len());
        let window2 = unescape(lines[i..hi2].join("\n"));

        let function = enclosing_function(&lines, i, is_py);
        let family = registry_entry(rel, &function).flatten();
        let mut params = BTreeMap::new();
        if central_policy {
            if let Some(family) = family {
                for key in SHARED_KEYS {
                    if let Some(value) = policy_value(family, key) {
                        params.insert((*key).to_owned(), value);
                    }
                }
            }
        } else {
            for key in SHARED_KEYS {
                if let Some(value) = find_value(&window, key) {
                    params.insert((*key).to_owned(), value);
                }
            }
        }
        let dep_form = if (central_policy
            && window2.contains("render_placement_block()")
            && family.is_some_and(|family| !spec_policy::policy(family).smoke.is_empty()))
            || window2.contains("[[dependencies.smoke]]")
        {
            DepForm::Smoke
        } else if window2.contains("smoke = [") {
            DepForm::SmokeInline
        } else if window2.contains("hostname = [") {
            DepForm::HostnameNoOp
        } else {
            DepForm::None
        };
        // A probe reading /etc/hostname is a host pin whichever form carries it.
        let probes_etc_hostname = matches!(dep_form, DepForm::Smoke | DepForm::SmokeInline)
            && window2.contains("/etc/hostname");
        out.push(RenderSite {
            file: rel.to_owned(),
            line: i + 1,
            function,
            family,
            params,
            dep_form,
            emits_resource_node: !central_policy
                && (section_declares(resource_section(&window2), "node")
                    || window.contains("node_line")),
            emits_ram_gb: !central_policy && section_declares(resource_section(&window2), "ram_gb"),
            probes_etc_hostname,
        });
    }
    Ok(out)
}

/// Every source file the scanner considers, repo-relative and sorted.
pub fn scan_targets(repo_root: &Path) -> Vec<String> {
    let mut out = vec![];
    for dir in SCAN_DIRS {
        let Ok(rd) = std::fs::read_dir(repo_root.join(dir)) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            let ok = p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "rs" || e == "py");
            if ok && p.is_file() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    let rel = format!("{dir}/{name}");
                    if !EXEMPT_FILES.contains(&rel.as_str()) {
                        out.push(rel);
                    }
                }
            }
        }
    }
    out.sort();
    out
}

/// Scan the whole repo for spec render sites.
pub fn scan_repo(repo_root: &Path) -> std::io::Result<Vec<RenderSite>> {
    let mut sites = vec![];
    for rel in scan_targets(repo_root) {
        sites.extend(scan_file(repo_root, &rel)?);
    }
    sites.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    Ok(sites)
}
