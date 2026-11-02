use super::*;

pub(super) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

pub(super) fn mtime_ms(p: &Path) -> Result<i64> {
    let md = fs::metadata(p).with_context(|| format!("stat {}", p.display()))?;
    Ok(md.modified()?.duration_since(UNIX_EPOCH)?.as_millis() as i64)
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

pub(super) fn f(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| x.as_f64())
}
pub(super) fn i(v: Option<&Value>) -> Option<i64> {
    v.and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|y| y as i64)))
}
pub(super) fn s(v: Option<&Value>) -> Option<String> {
    v.and_then(|x| x.as_str().map(|y| y.to_string()))
}
pub(super) fn b(v: Option<&Value>) -> Option<i64> {
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
pub(super) struct SourceSpec {
    pub(super) lane: String,
    pub(super) lane_kind: String,
    pub(super) in_policy_curve: bool,
    pub(super) run_id: String,
    pub(super) attempt: i64,
    pub(super) watermark: String,
    pub(super) node: Option<String>,
    pub(super) generation: Option<i64>,
    pub(super) model_version_hint: Option<String>,
    /// File actually read.
    pub(super) path: PathBuf,
    /// Where the bytes originally lived (may be a remote node).
    pub(super) origin: String,
    pub(super) sidecar_result: Option<PathBuf>,
    pub(super) sidecar_kind: Option<String>,
    pub(super) notes: Option<String>,
    /// Extra physical copies with identical content.
    pub(super) aliases: Vec<(String, String)>,
}

impl SourceSpec {
    pub(super) fn run_key(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.lane, self.run_id, self.attempt, self.watermark
        )
    }
}

/// scq run dirs, preferring WS-B's durable copy when it has at least as many rows.
pub(super) fn discover_scq(root: &Path) -> Result<Vec<SourceSpec>> {
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
pub(super) fn glob1(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        Some((a, b2)) => {
            name.len() >= a.len() + b2.len() && name.starts_with(a) && name.ends_with(b2)
        }
        None => pattern == name,
    }
}

/// Mirror `src` into the archive (preserving mtime) unless an identical copy is
/// already there. The archive must never depend on a directory it does not own.
pub(super) fn mirror(src: &Path, dst: &Path) -> Result<()> {
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
pub(super) fn discover_lane_roots(root: &Path, doc: &Value) -> Result<Vec<SourceSpec>> {
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

pub(super) fn discover_imported(root: &Path) -> Result<(Vec<SourceSpec>, Value)> {
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

#[derive(Default, Debug, Clone)]
pub(super) struct ScqTiming {
    pub(super) node: Option<String>,
    pub(super) submit_ms: Option<i64>,
    pub(super) running_ms: Option<i64>,
    pub(super) end_ms: Option<i64>,
    pub(super) state: Option<String>,
}

pub(super) fn parse_ts_ms(ts: &str) -> Option<i64> {
    // "2026-07-28T00:19:58.123456789+00:00" -> epoch ms, UTC offsets only.
    let bytes = ts.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    let y: i64 = ts.get(0..4)?.parse().ok()?;
    let mo: i64 = ts.get(5..7)?.parse().ok()?;
    let d: i64 = ts.get(8..10)?.parse().ok()?;
    let h: i64 = ts.get(11..13)?.parse().ok()?;
    let mi: i64 = ts.get(14..16)?.parse().ok()?;
    let sec: i64 = ts.get(17..19)?.parse().ok()?;
    let frac = if bytes.get(19) == Some(&b'.') {
        let mut ms = 0i64;
        let mut mult = 100i64;
        for c in bytes[20..].iter().take(3) {
            if !c.is_ascii_digit() {
                break;
            }
            ms += ((c - b'0') as i64) * mult;
            mult /= 10;
        }
        ms
    } else {
        0
    };
    // days since epoch (civil_from_days inverse, Howard Hinnant's algorithm)
    let yy = if mo <= 2 { y - 1 } else { y };
    let era = if yy >= 0 { yy } else { yy - 399 } / 400;
    let yoe = yy - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(((days * 86400 + h * 3600 + mi * 60 + sec) * 1000) + frac)
}

pub(super) fn scq_timings() -> Result<BTreeMap<String, ScqTiming>> {
    let mut out: BTreeMap<String, ScqTiming> = BTreeMap::new();
    let path = Path::new(scq_events_str());
    if !path.is_file() {
        return Ok(out);
    }
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    for line in txt.lines() {
        if !line.contains("train") {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let job = match v["job"].as_str() {
            Some(j) if j.contains("train") => j.to_string(),
            _ => continue,
        };
        let kind = v["kind"].as_str().unwrap_or("");
        let ts = v["ts"].as_str().and_then(parse_ts_ms);
        let e = out.entry(job).or_default();
        if let Some(n) = v["node"].as_str() {
            e.node = Some(n.to_string());
        }
        match kind {
            "submit" => e.submit_ms = e.submit_ms.or(ts),
            "running" => {
                if e.running_ms.is_none() {
                    e.running_ms = ts;
                }
            }
            "exit" | "cancel" | "fail" => {
                e.end_ms = ts;
                e.state = Some(kind.to_string());
            }
            _ => {}
        }
        if e.state.is_none() && kind == "running" {
            e.state = Some("running".into());
        }
    }
    Ok(out)
}

#[derive(Default, Debug, Clone)]
pub(super) struct SpecInfo {
    pub(super) recipe: Option<String>,
    pub(super) seed: Option<String>,
    pub(super) corpus_snapshot: Option<String>,
    pub(super) sampler: Option<String>,
    pub(super) warm_start_sha256: Option<String>,
    pub(super) trainer_source_sha256: Option<String>,
    pub(super) train_script_sha256: Option<String>,
    pub(super) path: Option<String>,
}

pub(super) fn read_spec(run_id: &str) -> SpecInfo {
    let mut info = SpecInfo::default();
    let p = PathBuf::from(specs_dir_str()).join(format!("{run_id}.toml"));
    let txt = match fs::read_to_string(&p) {
        Ok(t) => t,
        Err(_) => return info,
    };
    info.path = Some(p.display().to_string());
    // args = [... "--recipe", "a", "--seed", "20260726" ...]
    let toks: Vec<&str> = txt.split('"').collect();
    for w in toks.windows(3) {
        match w[0] {
            "--recipe" => info.recipe = Some(w[2].to_string()),
            "--seed" => info.seed = Some(w[2].to_string()),
            _ => {}
        }
    }
    // [[input]] blocks: name / source / target / sha256
    let mut cur_name: Option<String> = None;
    let mut cur_source: Option<String> = None;
    for line in txt.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("name = ") {
            cur_name = Some(rest.trim_matches('"').to_string());
            cur_source = None;
        } else if let Some(rest) = l.strip_prefix("source = ") {
            cur_source = Some(rest.trim_matches('"').to_string());
        } else if let Some(rest) = l.strip_prefix("sha256 = ") {
            let sha = rest.trim_matches('"').to_string();
            match cur_name.as_deref() {
                Some("checkpoint") => info.warm_start_sha256 = Some(sha),
                Some("trainer_source") => info.trainer_source_sha256 = Some(sha),
                Some("script") => info.train_script_sha256 = Some(sha),
                Some("manifest") => {
                    if let Some(src) = &cur_source {
                        if let Some(idx) = src.find("/training/snapshot-") {
                            let tail = &src[idx + "/training/".len()..];
                            let snap = tail.split('/').next().unwrap_or(tail).to_string();
                            if let Some(pos) = snap.rfind("-eq") {
                                info.sampler = Some(snap[pos + 1..].to_string());
                            }
                            info.corpus_snapshot = Some(snap);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    info
}

/// "warm-started 130/152 tensors from `<path>`"
pub(super) fn read_warm_start(run_id: &str, attempt: i64) -> Option<(i64, i64, String)> {
    let p = PathBuf::from(scq_runs_str())
        .join(run_id)
        .join("attempts")
        .join(attempt.to_string())
        .join("stdout.log");
    let txt = fs::read_to_string(p).ok()?;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("warm-started ") {
            let mut it = rest.split_whitespace();
            let frac = it.next()?;
            let (a, bb) = frac.split_once('/')?;
            let _tensors = it.next()?; // "tensors"
            let _from = it.next()?; // "from"
            let path = it.next()?.trim_end_matches(';').to_string();
            return Some((a.parse().ok()?, bb.parse().ok()?, path));
        }
    }
    None
}

pub(super) fn find_train_result(spec: &SourceSpec) -> Option<PathBuf> {
    if let Some(p) = &spec.sidecar_result {
        if p.is_file() {
            return Some(p.clone());
        }
    }
    if spec.lane != "scq" {
        return None;
    }
    let live = spec.path.parent()?.join("train-result.json");
    if live.is_file() {
        return Some(live);
    }
    None
}

// ------------------------------------------------------------------ ingest

pub(super) struct RowOut {
    pub(super) epoch: i64,
    pub(super) phase: String,
    pub(super) row_ord: i64,
    pub(super) obj: Map<String, Value>,
    pub(super) raw: String,
}

pub(super) fn parse_rows(text: &str, lane_kind: &str) -> (Vec<RowOut>, Vec<String>) {
    let mut rows = vec![];
    let mut problems = vec![];
    let mut seen: BTreeSet<(i64, String)> = BTreeSet::new();
    for (idx, line) in text.lines().enumerate() {
        let raw = line.trim_end_matches(['\r', '\n']);
        if raw.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(e) => {
                problems.push(format!("line {}: unparseable ({e})", idx + 1));
                continue;
            }
        };
        let obj = match v {
            Value::Object(m) => m,
            _ => {
                problems.push(format!("line {}: not an object", idx + 1));
                continue;
            }
        };
        let epoch = i(obj.get("epoch"))
            .or_else(|| i(obj.get("step")))
            .or_else(|| i(obj.get("seq")))
            .unwrap_or(idx as i64 + 1);
        let mut phase = obj
            .get("phase")
            .and_then(|x| x.as_str())
            .map(|x| x.to_string())
            .unwrap_or_else(|| {
                if lane_kind == "step" {
                    "STEP".into()
                } else {
                    "?".into()
                }
            });
        // Never silently overwrite: disambiguate a repeated (epoch, phase).
        if seen.contains(&(epoch, phase.clone())) {
            let disambiguated = format!("{phase}#{}", idx + 1);
            problems.push(format!(
                "line {}: duplicate (epoch={epoch}, phase={phase}) -> stored as phase '{disambiguated}'",
                idx + 1
            ));
            phase = disambiguated;
        }
        seen.insert((epoch, phase.clone()));
        rows.push(RowOut {
            epoch,
            phase,
            row_ord: idx as i64,
            obj,
            raw: raw.to_string(),
        });
    }
    (rows, problems)
}
