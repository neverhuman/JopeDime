use super::*;

#[derive(Default, Debug, Clone)]
pub(crate) struct ScqTiming {
    pub(crate) node: Option<String>,
    pub(crate) submit_ms: Option<i64>,
    pub(crate) running_ms: Option<i64>,
    pub(crate) end_ms: Option<i64>,
    pub(crate) state: Option<String>,
}

pub(crate) fn parse_ts_ms(ts: &str) -> Option<i64> {
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

pub(crate) fn scq_timings() -> Result<BTreeMap<String, ScqTiming>> {
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
pub(crate) struct SpecInfo {
    pub(crate) recipe: Option<String>,
    pub(crate) seed: Option<String>,
    pub(crate) corpus_snapshot: Option<String>,
    pub(crate) sampler: Option<String>,
    pub(crate) warm_start_sha256: Option<String>,
    pub(crate) trainer_source_sha256: Option<String>,
    pub(crate) train_script_sha256: Option<String>,
    pub(crate) path: Option<String>,
}

pub(crate) fn read_spec(run_id: &str) -> SpecInfo {
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
pub(crate) fn read_warm_start(run_id: &str, attempt: i64) -> Option<(i64, i64, String)> {
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

pub(crate) fn find_train_result(spec: &SourceSpec) -> Option<PathBuf> {
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

pub(crate) struct RowOut {
    pub(crate) epoch: i64,
    pub(crate) phase: String,
    pub(crate) row_ord: i64,
    pub(crate) obj: Map<String, Value>,
    pub(crate) raw: String,
}

pub(crate) fn parse_rows(text: &str, lane_kind: &str) -> (Vec<RowOut>, Vec<String>) {
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
