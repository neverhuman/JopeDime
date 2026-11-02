use super::*;

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

pub(crate) fn mtime_ms(p: &Path) -> Result<i64> {
    let md = fs::metadata(p).with_context(|| format!("stat {}", p.display()))?;
    Ok(md.modified()?.duration_since(UNIX_EPOCH)?.as_millis() as i64)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

pub(crate) fn f(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| x.as_f64())
}
pub(crate) fn i(v: Option<&Value>) -> Option<i64> {
    v.and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|y| y as i64)))
}
pub(crate) fn s(v: Option<&Value>) -> Option<String> {
    v.and_then(|x| x.as_str().map(|y| y.to_string()))
}
pub(crate) fn b(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Bool(x)) => Some(*x as i64),
        Some(Value::Object(_)) => Some(1), // anti_overfit_experiment is an object when enabled
        Some(Value::Null) | None => None,
        Some(Value::Number(n)) => n.as_i64(),
        _ => None,
    }
}

// ---------------------------------------------------------------- discovery

#[derive(Debug, Clone)]
pub(crate) struct SourceSpec {
    pub(crate) lane: String,
    pub(crate) lane_kind: String,
    pub(crate) in_policy_curve: bool,
    pub(crate) run_id: String,
    pub(crate) attempt: i64,
    pub(crate) watermark: String,
    pub(crate) node: Option<String>,
    pub(crate) generation: Option<i64>,
    pub(crate) model_version_hint: Option<String>,
    /// File actually read.
    pub(crate) path: PathBuf,
    /// Where the bytes originally lived (may be a remote node).
    pub(crate) origin: String,
    pub(crate) sidecar_result: Option<PathBuf>,
    pub(crate) sidecar_kind: Option<String>,
    pub(crate) notes: Option<String>,
    /// Extra physical copies with identical content.
    pub(crate) aliases: Vec<(String, String)>,
}

impl SourceSpec {
    pub(crate) fn run_key(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.lane, self.run_id, self.attempt, self.watermark
        )
    }
}

/// scq run dirs, preferring WS-B's durable copy when it has at least as many rows.
pub(crate) fn discover_scq(root: &Path) -> Result<Vec<SourceSpec>> {
    let mut by_key: BTreeMap<(String, i64, String), (PathBuf, String, usize)> = BTreeMap::new();

    let mut consider = |p: PathBuf, origin: String, run_id: String, attempt: i64, wm: String| {
        let rows = match fs::read_to_string(&p) {
            Ok(t) => t.lines().filter(|l| !l.trim().is_empty()).count(),
            Err(_) => return,
        };
        let key = (run_id, attempt, wm);
        let better = match by_key.get(&key) {
            Some((_, _, prev)) => rows > *prev,
            None => true,
        };
        if better {
            by_key.insert(key, (p, origin, rows));
        }
    };

    // (a) WS-B archive: runs/<run_id>/<attempt>/<watermark>/train_metrics.jsonl
    let wsb = root.join("runs");
    if wsb.is_dir() {
        for run in fs::read_dir(&wsb)? {
            let run = run?;
            if !run.file_type()?.is_dir() {
                continue;
            }
            let run_id = run.file_name().to_string_lossy().to_string();
            for att in fs::read_dir(run.path())? {
                let att = att?;
                if !att.file_type()?.is_dir() {
                    continue;
                }
                let attempt: i64 = att.file_name().to_string_lossy().parse().unwrap_or(0);
                for wmd in fs::read_dir(att.path())? {
                    let wmd = wmd?;
                    if !wmd.file_type()?.is_dir() {
                        continue;
                    }
                    let wm = wmd
                        .file_name()
                        .to_string_lossy()
                        .trim_start_matches("watermark-")
                        .to_string();
                    let p = wmd.path().join("train_metrics.jsonl");
                    if p.is_file() {
                        let origin = format!("xbabe2:{}", p.display());
                        consider(p, origin, run_id.clone(), attempt, wm);
                    }
                }
            }
        }
    }

    // (b) live scq run dirs (authoritative while they exist; may hold newer rows)
    let scq = Path::new(scq_runs_str());
    if scq.is_dir() {
        for run in fs::read_dir(scq)? {
            let run = run?;
            let run_id = run.file_name().to_string_lossy().to_string();
            if !run_id.contains("train") || !run.file_type()?.is_dir() {
                continue;
            }
            let attempts = run.path().join("attempts");
            if !attempts.is_dir() {
                continue;
            }
            for att in fs::read_dir(&attempts)? {
                let att = att?;
                let attempt: i64 = att.file_name().to_string_lossy().parse().unwrap_or(0);
                let tdir = att.path().join("outputs/training");
                if !tdir.is_dir() {
                    continue;
                }
                for wmd in fs::read_dir(&tdir)? {
                    let wmd = wmd?;
                    let wm = wmd
                        .file_name()
                        .to_string_lossy()
                        .trim_start_matches("watermark-")
                        .to_string();
                    let p = wmd.path().join("train_metrics.jsonl");
                    if p.is_file() {
                        let origin = format!("xbabe2:{}", p.display());
                        consider(p, origin, run_id.clone(), attempt, wm);
                    }
                }
            }
        }
    }

    Ok(by_key
        .into_iter()
        .map(
            |((run_id, attempt, watermark), (path, origin, _))| SourceSpec {
                lane: "scq".into(),
                lane_kind: "epoch".into(),
                in_policy_curve: true,
                run_id,
                attempt,
                watermark,
                node: None, // resolved from scq events
                generation: None,
                model_version_hint: None,
                path,
                origin,
                sidecar_result: None, // resolved below
                sidecar_kind: Some("train_result".into()),
                notes: None,
                aliases: vec![],
            },
        )
        .collect())
}

/// Glob with a single '*' wildcard: "gen*_train_metrics.jsonl".
pub(crate) fn glob1(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        Some((a, b2)) => {
            name.len() >= a.len() + b2.len() && name.starts_with(a) && name.ends_with(b2)
        }
        None => pattern == name,
    }
}

/// Mirror `src` into the archive (preserving mtime) unless an identical copy is
/// already there. The archive must never depend on a directory it does not own.
pub(crate) fn mirror(src: &Path, dst: &Path) -> Result<()> {
    if let Some(p) = dst.parent() {
        fs::create_dir_all(p)?;
    }
    let bytes = fs::read(src)?;
    let same = fs::read(dst).map(|d| d == bytes).unwrap_or(false);
    if !same {
        let tmp = dst.with_extension("tmp-ingest");
        fs::write(&tmp, &bytes)?;
        fs::rename(&tmp, dst)?;
    }
    // keep the source mtime: it is the only wall-clock anchor these lanes have
    let mt = fs::metadata(src)?.modified()?;
    let ft = fs::File::options().write(true).open(dst)?;
    ft.set_modified(mt)?;
    Ok(())
}

/// Walk each `lane_roots` directory, mirror what it finds into the archive and
/// return one SourceSpec per file. run_id is derived from the file name so that
/// re-running is stable.
pub(crate) fn discover_lane_roots(root: &Path, doc: &Value) -> Result<Vec<SourceSpec>> {
    let mut out = vec![];
    for lr in doc["lane_roots"].as_array().cloned().unwrap_or_default() {
        let dir = PathBuf::from(lr["dir"].as_str().unwrap_or_default());
        if !dir.is_dir() {
            eprintln!("WARN lane_root missing: {}", dir.display());
            continue;
        }
        let pattern = lr["pattern"].as_str().unwrap_or("*");
        let lane = lr["lane"].as_str().unwrap_or("legacy").to_string();
        let subdir = lr["archive_subdir"].as_str().unwrap_or(&lane).to_string();
        let node = lr["node"].as_str().unwrap_or("xbabe2").to_string();
        let sidecar_pattern = lr["sidecar_pattern"].as_str().map(|x| x.to_string());
        let mut names: Vec<String> = fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| glob1(pattern, n))
            .collect();
        names.sort();
        for name in names {
            let src = dir.join(&name);
            let dst = root.join("db/imported").join(&subdir).join(&name);
            mirror(&src, &dst)?;
            // gen17_train_metrics.jsonl -> gen17 ; train_metrics.pre_logging_patch.jsonl -> pre_logging_patch
            let stem = name
                .trim_end_matches(".jsonl")
                .replace("_train_metrics", "")
                .replace("train_metrics.", "")
                .replace("train_metrics", "")
                .trim_matches(['_', '.', '-'])
                .to_string();
            // archive_subdir's last component is unique per lane root, and file
            // names are unique within a directory, so this run_id cannot collide.
            let base = subdir
                .rsplit('/')
                .next()
                .unwrap_or(&lane)
                .trim_start_matches("jope-runs-")
                .trim_start_matches("jope-prime-")
                .to_string();
            let run_id = if stem.is_empty() {
                base
            } else {
                format!("{base}-{stem}")
            };
            let generation = stem.strip_prefix("gen").and_then(|g| g.parse::<i64>().ok());
            // sidecar with the same generation prefix, if the lane declares one
            let sidecar = sidecar_pattern.as_ref().and_then(|sp| {
                let want = sp.replacen('*', stem.strip_prefix("gen").unwrap_or(&stem), 1);
                let s = dir.join(&want);
                if s.is_file() {
                    let d = root.join("db/imported").join(&subdir).join(&want);
                    mirror(&s, &d).ok()?;
                    Some(d)
                } else {
                    None
                }
            });
            out.push(SourceSpec {
                lane: lane.clone(),
                lane_kind: lr["lane_kind"].as_str().unwrap_or("epoch").to_string(),
                in_policy_curve: lr["in_policy_curve"].as_bool().unwrap_or(false),
                run_id,
                attempt: 0,
                watermark: "none".into(),
                node: Some(node.clone()),
                generation,
                model_version_hint: lr["model_version_hint"].as_str().map(|x| x.to_string()),
                path: dst,
                origin: format!("{node}:{}", src.display()),
                sidecar_result: sidecar,
                sidecar_kind: lr["sidecar_kind"].as_str().map(|x| x.to_string()),
                notes: lr["notes"].as_str().map(|x| x.to_string()),
                aliases: vec![],
            });
        }
    }
    Ok(out)
}

pub(crate) fn discover_imported(root: &Path) -> Result<(Vec<SourceSpec>, Value)> {
    let manifest_path = root.join("db/imported/sources.v1.json");
    if !manifest_path.is_file() {
        return Ok((vec![], Value::Null));
    }
    let txt = fs::read_to_string(&manifest_path)?;
    let doc: Value =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", manifest_path.display()))?;
    let mut out = discover_lane_roots(root, &doc)?;
    for src in doc["sources"].as_array().cloned().unwrap_or_default() {
        let local = src["local_path"].as_str().unwrap_or_default();
        let path = root.join("db").join(local);
        if !path.is_file() {
            eprintln!("WARN missing imported source {}", path.display());
            continue;
        }
        let node = src["node"].as_str().unwrap_or("unknown").to_string();
        out.push(SourceSpec {
            lane: src["lane"].as_str().unwrap_or("imported").to_string(),
            lane_kind: src["lane_kind"].as_str().unwrap_or("epoch").to_string(),
            in_policy_curve: src["in_policy_curve"].as_bool().unwrap_or(false),
            run_id: src["run_id"].as_str().unwrap_or("unknown").to_string(),
            attempt: src["attempt"].as_i64().unwrap_or(0),
            watermark: src["watermark"].as_str().unwrap_or("none").to_string(),
            node: Some(node.clone()),
            generation: src["generation"].as_i64(),
            model_version_hint: src["model_version_hint"].as_str().map(|x| x.to_string()),
            path,
            origin: format!("{}:{}", node, src["origin_path"].as_str().unwrap_or("?")),
            sidecar_result: src["sidecar_result"]
                .as_str()
                .map(|p| root.join("db").join(p)),
            sidecar_kind: src["sidecar_kind"].as_str().map(|x| x.to_string()),
            notes: src["notes"].as_str().map(|x| x.to_string()),
            aliases: vec![],
        });
    }
    Ok((out, doc))
}

// ------------------------------------------------------------- scq metadata
