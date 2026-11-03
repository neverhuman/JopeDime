//! Privileged, read-only batch process/open-file probe.
//! Native Rust replacement for `build_consumers.py`, including the D-state kernel lock guard.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixStream;
use std::path::{Component, Path};
use std::time::Duration;

use crate::cache_namespaces::is_shared_child;
use crate::json::Json;
use crate::json_obj;

pub fn below(p: &str, root: &str) -> bool {
    if p == root {
        return true;
    }
    let prefix = root.trim_end_matches('/');
    p.starts_with(prefix) && p.as_bytes().get(prefix.len()) == Some(&b'/')
}

pub fn normpath(raw: &str) -> String {
    let is_abs = raw.starts_with('/');
    let mut stack: Vec<&str> = Vec::new();
    for part in raw.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            if !stack.is_empty() && stack.last() != Some(&"..") {
                stack.pop();
            } else if !is_abs {
                stack.push("..");
            }
        } else {
            stack.push(part);
        }
    }
    if is_abs {
        format!("/{}", stack.join("/"))
    } else if stack.is_empty() {
        ".".to_string()
    } else {
        stack.join("/")
    }
}

fn unescape_mount_path(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && i + 3 < bytes.len()
            && (b'0'..=b'7').contains(&bytes[i + 1])
            && (b'0'..=b'7').contains(&bytes[i + 2])
            && (b'0'..=b'7').contains(&bytes[i + 3])
        {
            let val =
                (bytes[i + 1] - b'0') * 64 + (bytes[i + 2] - b'0') * 8 + (bytes[i + 3] - b'0');
            out.push(val);
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn makedev(major: u64, minor: u64) -> u64 {
    ((major & 0xfffff000) << 32)
        | ((major & 0x00000fff) << 8)
        | ((minor & 0xffffff00) << 12)
        | (minor & 0x000000ff)
}

pub fn safe_nonlink_mounts(roots: &[String], mountinfo: &Path) -> Vec<String> {
    let mut devices = BTreeSet::new();
    for r in roots {
        if let Ok(m) = fs::metadata(r) {
            devices.insert(m.dev());
        } else {
            return Vec::new();
        }
    }
    let Ok(text) = fs::read_to_string(mountinfo) else {
        return Vec::new();
    };
    let mut mounts = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let Some(sep_idx) = fields.iter().position(|&f| f == "-") else {
            return Vec::new();
        };
        if fields.len() <= sep_idx + 1 || fields.len() < 5 {
            return Vec::new();
        }
        let path = unescape_mount_path(fields[4]);
        let Some((maj_s, min_s)) = fields[2].split_once(':') else {
            return Vec::new();
        };
        let (Ok(maj), Ok(min)) = (maj_s.parse::<u64>(), min_s.parse::<u64>()) else {
            return Vec::new();
        };
        let fstype = fields[sep_idx + 1].to_string();
        mounts.push((path, fstype, makedev(maj, min)));
    }
    let mut safe = Vec::new();
    for (p, kind, dev) in &mounts {
        if kind == "exfat"
            && !devices.contains(dev)
            && !mounts.iter().any(|(q, _, _)| q != p && below(q, p))
        {
            safe.push(p.clone());
        }
    }
    safe
}

fn resolve_claim_path(raw: &str, mounts: &[String], same_namespace: bool) -> String {
    let has_parent_ref = Path::new(raw)
        .components()
        .any(|c| matches!(c, Component::ParentDir));
    if same_namespace && !has_parent_ref && mounts.iter().any(|m| below(raw, m)) {
        return normpath(raw);
    }
    fs::canonicalize(raw)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| normpath(raw))
}

fn shared_ancestors(root: &str) -> BTreeSet<String> {
    let p = Path::new(root);
    let mut out = BTreeSet::new();
    if is_shared_child(p) {
        if let Some(parent) = p.parent() {
            out.insert(parent.to_string_lossy().into_owned());
            if let Some(gp) = parent.parent() {
                out.insert(gp.to_string_lossy().into_owned());
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
enum ParentScope {
    None,
    ParentCwd,
    SharedAncestors(Vec<String>),
}

fn alias_claim(kind: &str, path: &str, alias: &str, scope: &ParentScope) -> bool {
    if below(path, alias) {
        return true;
    }
    if kind.starts_with("env:")
        && !matches!(kind, "env:PATH" | "env:PYTHONPATH" | "env:LD_LIBRARY_PATH")
        && below(alias, path)
    {
        return true;
    }
    match scope {
        ParentScope::ParentCwd => {
            if kind == "cwd" {
                if let Some(parent) = Path::new(alias).parent() {
                    if below(path, &parent.to_string_lossy()) {
                        return true;
                    }
                }
            }
        }
        ParentScope::SharedAncestors(ancs) => {
            if matches!(
                kind,
                "cwd" | "args" | "args_lexical" | "fd" | "maps" | "exe"
            ) && ancs.iter().any(|a| a == path)
            {
                return true;
            }
        }
        ParentScope::None => {}
    }
    false
}

fn docker_aliases(
    roots: &[String],
) -> Result<BTreeMap<String, BTreeMap<String, Vec<(String, ParentScope)>>>, String> {
    let mut stream = UnixStream::connect("/var/run/docker.sock")
        .map_err(|e| format!("connect docker.sock: {e}"))?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    stream
        .write_all(b"GET /containers/json HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .map_err(|e| format!("write docker.sock: {e}"))?;
    let mut resp = Vec::new();
    stream
        .read_to_end(&mut resp)
        .map_err(|e| format!("read docker.sock: {e}"))?;
    let text = String::from_utf8_lossy(&resp);
    let Some((headers, body_raw)) = text.split_once("\r\n\r\n") else {
        return Err("malformed HTTP response from docker.sock".to_string());
    };
    let first_line = headers.lines().next().unwrap_or("");
    if !first_line.contains(" 200 ") {
        return Err(format!("Docker mount inventory HTTP: {first_line}"));
    }
    let body = if headers
        .lines()
        .any(|l| l.to_ascii_lowercase().contains("transfer-encoding: chunked"))
    {
        decode_chunked(body_raw)?
    } else {
        body_raw.to_string()
    };
    let parsed = Json::parse(&body).map_err(|e| format!("parse docker JSON: {e}"))?;
    let Some(containers) = parsed.as_arr() else {
        return Err("docker /containers/json not an array".to_string());
    };

    let mut result = BTreeMap::new();
    for c in containers {
        let Some(cid) = c.get_str("Id") else {
            continue;
        };
        let Some(mounts) = c.get("Mounts").and_then(|v| v.as_arr()) else {
            continue;
        };
        let mut aliases: BTreeMap<String, Vec<(String, ParentScope)>> = BTreeMap::new();
        for m in mounts {
            let (Some(src_raw), Some(dst)) = (m.get_str("Source"), m.get_str("Destination")) else {
                continue;
            };
            let src = fs::canonicalize(src_raw)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| normpath(src_raw));
            for root in roots {
                if below(root, &src) {
                    let rel = root
                        .strip_prefix(&src)
                        .unwrap_or("")
                        .trim_start_matches('/');
                    let alias = if rel.is_empty() {
                        dst.to_string()
                    } else {
                        normpath(&format!("{dst}/{rel}"))
                    };
                    let mut scope = if !rel.is_empty()
                        && Path::new(root).file_name().and_then(|s| s.to_str()) == Some("target")
                    {
                        ParentScope::ParentCwd
                    } else {
                        ParentScope::None
                    };
                    if is_shared_child(Path::new(root)) {
                        let mut mapped = Vec::new();
                        for anc in shared_ancestors(root) {
                            if below(&anc, &src) {
                                let suf =
                                    anc.strip_prefix(&src).unwrap_or("").trim_start_matches('/');
                                let m_path = if suf.is_empty() {
                                    dst.to_string()
                                } else {
                                    normpath(&format!("{dst}/{suf}"))
                                };
                                mapped.push(m_path);
                            }
                        }
                        scope = if mapped.is_empty() {
                            ParentScope::None
                        } else {
                            ParentScope::SharedAncestors(mapped)
                        };
                    }
                    aliases.entry(root.clone()).or_default().push((alias, scope));
                } else if below(&src, root) {
                    aliases
                        .entry(root.clone())
                        .or_default()
                        .push((dst.to_string(), ParentScope::None));
                }
            }
        }
        if !aliases.is_empty() {
            result.insert(cid.to_string(), aliases);
        }
    }
    Ok(result)
}

fn decode_chunked(mut input: &str) -> Result<String, String> {
    let mut out = String::new();
    loop {
        let Some((len_hex, rest)) = input.split_once("\r\n") else {
            break;
        };
        let len_clean = len_hex.split(';').next().unwrap_or("").trim();
        if len_clean.is_empty() {
            break;
        }
        let n = usize::from_str_radix(len_clean, 16)
            .map_err(|_| format!("invalid chunk length '{len_clean}'"))?;
        if n == 0 {
            break;
        }
        if rest.len() < n + 2 {
            return Err("truncated HTTP chunk".to_string());
        }
        out.push_str(&rest[..n]);
        input = &rest[n + 2..];
    }
    Ok(out)
}

fn is_boundary_after(s: &str, pos: usize) -> bool {
    if pos >= s.len() {
        return true;
    }
    let b = s.as_bytes()[pos];
    matches!(
        b,
        b'/' | b' ' | b'\t' | b'\n' | b'\r' | b'"' | b'\'' | b'=' | b':' | b';' | b',' | b')' | b']'
    )
}

fn contains_explicit_root(arg: &str, root: &str, parent_refs: &BTreeSet<String>) -> bool {
    let mut start = 0;
    while let Some(idx) = arg[start..].find(root) {
        let abs_end = start + idx + root.len();
        if is_boundary_after(arg, abs_end) {
            return true;
        }
        start += idx + 1;
    }
    for anc in parent_refs {
        let mut s = 0;
        while let Some(idx) = arg[s..].find(anc.as_str()) {
            let mut abs_end = s + idx + anc.len();
            if arg.as_bytes().get(abs_end) == Some(&b'/') {
                abs_end += 1;
            }
            if abs_end >= arg.len()
                || matches!(
                    arg.as_bytes()[abs_end],
                    b' ' | b'\t' | b'\n' | b'\r' | b'"' | b'\'' | b'=' | b':' | b';' | b',' | b')' | b']'
                )
            {
                return true;
            }
            s += idx + 1;
        }
    }
    false
}

fn split_shell_words(arg: &str) -> Result<Vec<String>, ()> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut chars = arg.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    while let Some(c) = chars.next() {
        if in_single {
            if c == '\'' {
                in_single = false;
            } else {
                cur.push(c);
            }
        } else if in_double {
            if c == '"' {
                in_double = false;
            } else if c == '\\' {
                if let Some(next_c) = chars.next() {
                    cur.push(next_c);
                } else {
                    return Err(());
                }
            } else {
                cur.push(c);
            }
        } else {
            match c {
                '\'' => in_single = true,
                '"' => in_double = true,
                '\\' => {
                    if let Some(next_c) = chars.next() {
                        cur.push(next_c);
                    } else {
                        return Err(());
                    }
                }
                c if c.is_whitespace() || matches!(c, ';' | '&' | '|' | '(' | ')' | '<' | '>') => {
                    if !cur.is_empty() {
                        words.push(std::mem::take(&mut cur));
                    }
                }
                c => cur.push(c),
            }
        }
    }
    if in_single || in_double {
        return Err(());
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    Ok(words)
}

fn read_link_clean(p: &Path) -> Option<String> {
    fs::read_link(p).ok().map(|target| {
        let s = target.to_string_lossy();
        s.strip_suffix(" (deleted)").unwrap_or(&s).to_string()
    })
}

fn parse_stat_text(stat_text: &str) -> Option<(String, String, String)> {
    let open_paren = stat_text.find('(')?;
    let close_paren = stat_text.rfind(')')?;
    if close_paren <= open_paren {
        return None;
    }
    let comm = stat_text[open_paren + 1..close_paren].to_string();
    let fields: Vec<&str> = stat_text[close_paren + 1..].split_whitespace().collect();
    if fields.len() < 20 {
        return None;
    }
    let state = fields[0].to_string();
    let start_ticks = fields[19].to_string();
    Some((state, comm, start_ticks))
}

pub fn probe(roots: &[String], proc_root: &Path) -> Json {
    let mut claims: BTreeMap<String, Vec<Json>> =
        roots.iter().map(|r| (r.clone(), Vec::new())).collect();
    let parent_refs: BTreeMap<String, BTreeSet<String>> = roots
        .iter()
        .map(|r| (r.clone(), shared_ancestors(r)))
        .collect();
    let cwd_parents: BTreeMap<String, Option<String>> = roots
        .iter()
        .map(|r| {
            let p = Path::new(r);
            let parent = if p.file_name().and_then(|s| s.to_str()) == Some("target") {
                p.parent().map(|pp| pp.to_string_lossy().into_owned())
            } else {
                None
            };
            (r.clone(), parent)
        })
        .collect();

    let mut errors: Vec<Json> = Vec::new();
    let is_real_proc = proc_root == Path::new("/proc");
    let fast_mounts = if is_real_proc {
        safe_nonlink_mounts(roots, Path::new("/proc/self/mountinfo"))
    } else {
        Vec::new()
    };
    let host_namespace = if !fast_mounts.is_empty() {
        fs::read_link("/proc/self/ns/mnt").ok()
    } else {
        None
    };

    let aliases = if is_real_proc {
        match docker_aliases(roots) {
            Ok(a) => a,
            Err(e) => {
                errors.push(json_obj! {
                    "scope" => "Docker mount aliases",
                    "error" => e,
                });
                BTreeMap::new()
            }
        }
    } else {
        BTreeMap::new()
    };

    let Ok(proc_iter) = fs::read_dir(proc_root) else {
        return json_obj! {
            "complete" => false,
            "errors" => vec![json_obj!{"error" => "cannot read proc_root"}],
            "consumers" => Json::Obj(BTreeMap::new()),
        };
    };

    for entry in proc_iter.flatten() {
        let file_name = entry.file_name();
        let Some(name_str) = file_name.to_str() else {
            continue;
        };
        if !name_str.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(pid) = name_str.parse::<i64>() else {
            continue;
        };
        let proc_dir = entry.path();

        // Read /proc/<pid>/stat FIRST so we can detect D-state (uninterruptible sleep)
        // before touching cmdline, environ, or maps (which acquire mm->mmap_lock in kernel).
        let stat_text = match fs::read_to_string(proc_dir.join("stat")) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                errors.push(json_obj! { "pid" => pid, "error" => e.to_string() });
                continue;
            }
        };
        let Some((state, comm, start_ticks)) = parse_stat_text(&stat_text) else {
            continue;
        };

        let mut paths: Vec<(String, String)> = Vec::new();
        let cwd = read_link_clean(&proc_dir.join("cwd")).unwrap_or_default();
        let exe = read_link_clean(&proc_dir.join("exe")).unwrap_or_default();
        if !cwd.is_empty() {
            paths.push(("cwd".to_string(), cwd.clone()));
        }
        if !exe.is_empty() {
            paths.push(("exe".to_string(), exe.clone()));
        }

        if let Ok(fd_iter) = fs::read_dir(proc_dir.join("fd")) {
            for fd_entry in fd_iter.flatten() {
                if let Some(val) = read_link_clean(&fd_entry.path()) {
                    if val.starts_with('/') {
                        paths.push(("fd".to_string(), val));
                    }
                }
            }
        }

        // Only read cmdline, environ, and maps if the process is NOT stuck in D-state!
        if state != "D" {
            let cmdline_bytes = match fs::read(proc_dir.join("cmdline")) {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(_) => Vec::new(),
            };
            let args: Vec<String> = cmdline_bytes
                .split(|&b| b == 0)
                .filter(|s| !s.is_empty())
                .map(|s| String::from_utf8_lossy(s).into_owned())
                .collect();
            if args.is_empty() {
                continue;
            }

            let same_namespace = host_namespace
                .as_ref()
                .map(|hns| fs::read_link(proc_dir.join("ns/mnt")).ok().as_ref() == Some(hns))
                .unwrap_or(false);

            for arg in &args {
                let mut words = vec![arg.clone()];
                if arg.chars().any(|c| c.is_whitespace()) {
                    match split_shell_words(arg) {
                        Ok(extra) => words.extend(extra),
                        Err(()) => {
                            let prog = Path::new(&args[0])
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("");
                            if matches!(prog, "bash" | "sh" | "dash" | "zsh") {
                                errors.push(json_obj! {
                                    "pid" => pid,
                                    "error" => "unparseable shell",
                                });
                            }
                        }
                    }
                }
                for mut word in words {
                    if let Some((_, rhs)) = word.split_once('=') {
                        word = rhs.to_string();
                    }
                    if !word.is_empty()
                        && !word.starts_with('-')
                        && !word.chars().any(|c| c.is_whitespace())
                    {
                        let raw = if word.starts_with('/') {
                            word.clone()
                        } else {
                            format!("{cwd}/{word}")
                        };
                        paths.push(("args_lexical".to_string(), normpath(&raw)));
                        paths.push((
                            "args".to_string(),
                            resolve_claim_path(&raw, &fast_mounts, same_namespace),
                        ));
                    }
                }
                for root in roots {
                    if let Some(prefs) = parent_refs.get(root) {
                        if contains_explicit_root(arg, root, prefs) {
                            paths.push(("explicit_args".to_string(), root.clone()));
                        }
                    }
                }
            }

            if let Ok(env_bytes) = fs::read(proc_dir.join("environ")) {
                let mut env_map = BTreeMap::new();
                for item in env_bytes.split(|&b| b == 0) {
                    if let Some(pos) = item.iter().position(|&b| b == b'=') {
                        let k = String::from_utf8_lossy(&item[..pos]).into_owned();
                        let v = String::from_utf8_lossy(&item[pos + 1..]).into_owned();
                        env_map.insert(k, v);
                    }
                }
                let home_val = env_map.get("HOME").cloned().unwrap_or_default();
                for key in [
                    "CARGO_TARGET_DIR",
                    "CARGO_HOME",
                    "SCCACHE_DIR",
                    "TMPDIR",
                    "PATH",
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                ] {
                    if let Some(raw_val) = env_map.get(key) {
                        for part in raw_val.split(':') {
                            if part.is_empty() {
                                continue;
                            }
                            let expanded = if let Some(rest) = part.strip_prefix("~/") {
                                format!("{home_val}/{rest}")
                            } else {
                                part.to_string()
                            };
                            let joined = if expanded.starts_with('/') {
                                expanded
                            } else {
                                format!("{cwd}/{expanded}")
                            };
                            let kind = format!("env:{key}");
                            paths.push((kind.clone(), normpath(&joined)));
                            paths.push((
                                kind,
                                resolve_claim_path(&joined, &fast_mounts, same_namespace),
                            ));
                        }
                    }
                }
            }

            if let Ok(maps_text) = fs::read_to_string(proc_dir.join("maps")) {
                for line in maps_text.lines() {
                    let mut cols = line.split_whitespace();
                    let _ = (
                        cols.next(),
                        cols.next(),
                        cols.next(),
                        cols.next(),
                        cols.next(),
                    );
                    if let Some(path_part) = cols.next() {
                        if path_part.starts_with('/') {
                            let clean = path_part
                                .strip_suffix(" (deleted)")
                                .unwrap_or(path_part)
                                .to_string();
                            paths.push(("maps".to_string(), clean));
                        }
                    }
                }
            }
        }

        let (latest_state, latest_comm, latest_ticks) = fs::read_to_string(proc_dir.join("stat"))
            .ok()
            .and_then(|s| parse_stat_text(&s))
            .unwrap_or((state, comm, start_ticks));
        let latest_cwd = read_link_clean(&proc_dir.join("cwd")).unwrap_or(cwd);
        let latest_exe = read_link_clean(&proc_dir.join("exe")).unwrap_or(exe);
        if !latest_cwd.is_empty() {
            paths.push(("cwd".to_string(), latest_cwd.clone()));
        }
        if !latest_exe.is_empty() {
            paths.push(("exe".to_string(), latest_exe.clone()));
        }

        for root in roots {
            let mut reasons = BTreeSet::new();
            let cwd_p = cwd_parents.get(root).and_then(|x| x.as_deref());
            let prefs = parent_refs.get(root);
            for (kind, p) in &paths {
                let matched = below(p, root)
                    || (kind.starts_with("env:")
                        && !matches!(
                            kind.as_str(),
                            "env:PATH" | "env:PYTHONPATH" | "env:LD_LIBRARY_PATH"
                        )
                        && below(root, p))
                    || (kind == "cwd" && cwd_p.is_some_and(|cp| below(p, cp)))
                    || (matches!(
                        kind.as_str(),
                        "cwd" | "args" | "args_lexical" | "fd" | "maps" | "exe"
                    ) && prefs.is_some_and(|pr| pr.contains(p)));
                if matched {
                    reasons.insert(kind.clone());
                }
            }
            if !reasons.is_empty() {
                let reason_vec: Vec<String> = reasons.into_iter().collect();
                if let Some(list) = claims.get_mut(root) {
                    list.push(json_obj! {
                        "pid" => pid,
                        "state" => latest_state.clone(),
                        "start_ticks" => latest_ticks.clone(),
                        "comm" => latest_comm.clone(),
                        "exe" => latest_exe.clone(),
                        "cwd" => latest_cwd.clone(),
                        "reasons" => reason_vec,
                    });
                }
            }
        }

        if !aliases.is_empty() {
            if let Ok(cgroup) = fs::read_to_string(proc_dir.join("cgroup")) {
                for (cid, mappings) in &aliases {
                    if !cgroup.contains(cid) {
                        continue;
                    }
                    let short_cid: String = cid.chars().take(12).collect();
                    for (root, mapped) in mappings {
                        let mut reasons = BTreeSet::new();
                        for (alias, scope) in mapped {
                            for (kind, path) in &paths {
                                if alias_claim(kind, path, alias, scope) {
                                    reasons.insert(format!("docker_alias:{kind}"));
                                }
                            }
                        }
                        if !reasons.is_empty() {
                            let reason_vec: Vec<String> = reasons.into_iter().collect();
                            if let Some(list) = claims.get_mut(root) {
                                list.push(json_obj! {
                                    "pid" => pid,
                                    "state" => latest_state.clone(),
                                    "start_ticks" => latest_ticks.clone(),
                                    "comm" => latest_comm.clone(),
                                    "exe" => latest_exe.clone(),
                                    "cwd" => latest_cwd.clone(),
                                    "container" => short_cid.clone(),
                                    "reasons" => reason_vec,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    let consumers_obj: BTreeMap<String, Json> = claims
        .into_iter()
        .map(|(k, v)| (k, Json::Arr(v)))
        .collect();
    json_obj! {
        "complete" => errors.is_empty(),
        "errors" => Json::Arr(errors),
        "consumers" => Json::Obj(consumers_obj),
    }
}

pub fn cli_probe_stdin() -> i32 {
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        eprintln!("failed to read stdin");
        return 1;
    }
    let Ok(parsed) = Json::parse(&input) else {
        eprintln!("invalid JSON on stdin");
        return 1;
    };
    let roots: Vec<String> = parsed
        .get("roots")
        .and_then(|v| v.as_arr())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let out = probe(&roots, Path::new("/proc"));
    println!("{}", out.to_compact());
    0
}
