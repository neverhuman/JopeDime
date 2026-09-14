//! Read-only, restartable loss history. The trainer's saved logs are authoritative;
//! the observer cache only avoids reparsing unchanged files.
use ratatui::{layout::{Constraint, Layout, Rect}, style::{Color, Style, Stylize},
    symbols, text::Line, widgets::{Axis, Block, Chart, Dataset, GraphType, List, ListItem, Paragraph, Wrap}, Frame};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::{BTreeMap, BTreeSet}, fs::{self, File}, io::{BufRead, BufReader, Seek, SeekFrom},
    os::unix::fs::MetadataExt, path::{Path, PathBuf}, sync::{atomic::{AtomicBool, Ordering}, mpsc, Arc},
    thread, time::{Duration, SystemTime}};

#[derive(Clone, Debug, Deserialize)]
pub struct Validation {
    pub sequence_nll: f64,
    pub ranking_bt: f64,
    pub evaluated_step: Option<u64>,
    pub panel_sha256: Option<String>,
}
#[derive(Deserialize)]
struct Metric {
    schema: String, rank: u64, run: PathBuf, step: u64, timestamp: u64,
    corpus_sha256: String, sequence_nll: f64,
    validation: Option<Validation>, initialization: Option<Value>,
    #[serde(default)] invalidated: bool,
    #[serde(default)] batch_qualification: bool,
}
#[derive(Clone, Debug)]
pub struct Point {
    pub step: u64, pub timestamp: u64, pub train: f64,
    pub validation: Option<Validation>, pub attempt: usize,
}
#[derive(Clone, Debug, Default)]
pub struct Run {
    pub path: PathBuf, pub points: Vec<Point>, pub checkpoint: String,
    pub parent: String, pub corpora: BTreeSet<String>, pub status: String,
    pub malformed: usize, pub unbound_validation: usize, pub pending_line: bool,
    pub read_error: Option<String>, pub attempts: usize,
}
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub runs: Vec<Run>, pub ready: bool, pub errors: Vec<String>,
}
impl Snapshot {
    pub fn report(&self) -> Value {
        serde_json::json!({"schema":"parallel.learning_history.v1","read_only":true,"ready":self.ready,
            "records":self.runs.iter().map(|r|r.points.len()).sum::<usize>(),"errors":self.errors,
            "scope":"All persisted campaign, auxiliary and standalone training records; branches and restarts retained. Diagnostic batch runs excluded.",
            "runs":self.runs.iter().map(|r|serde_json::json!({"path":r.path,"status":r.status,
                "records":r.points.len(),"attempts":r.attempts,"parent":r.parent,"checkpoint":r.checkpoint,
                "first_timestamp":r.points.first().map(|p|p.timestamp),"last_timestamp":r.points.last().map(|p|p.timestamp),
                "first_step":r.points.first().map(|p|p.step),"last_step":r.points.last().map(|p|p.step),
                "first_training_nll":r.points.first().map(|p|p.train),"last_training_nll":r.points.last().map(|p|p.train),
                "bound_validation_points":r.points.iter().filter(|p|p.validation.is_some()).count(),
                "corpora":r.corpora,"malformed_records":r.malformed,"unbound_validation_rows":r.unbound_validation,
                "incomplete_final_line":r.pending_line,"read_error":r.read_error})).collect::<Vec<_>>()})
    }
}
#[derive(Default)]
struct FileCache {
    run: Run, offset: u64, inode: Option<(u64, u64)>, modified: Option<SystemTime>, length: u64,
}
#[derive(Default)]
pub struct Cache { files: BTreeMap<PathBuf, FileCache> }

fn json(path: &Path) -> Value {
    fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}
fn safe_path(root: &Path, path: &Path) -> Option<PathBuf> {
    let path = root.join(path);
    if !path.starts_with(root.join("runs")) { return None; }
    for ancestor in path.ancestors().take_while(|p| *p != root) {
        if ancestor.is_symlink() { return None; }
    }
    let canonical = path.canonicalize().ok()?;
    canonical.starts_with(root.join("runs")).then_some(canonical)
}
fn discover(root: &Path) -> (BTreeSet<PathBuf>, Vec<String>) {
    let mut cycles = BTreeSet::new(); let mut runs = BTreeSet::new(); let mut errors = Vec::new();
    let mut campaigns = BTreeSet::new();
    let campaign = json(&root.join("runs/campaign-progress.json"));
    let current_parent = campaign["current_cycle"].as_str().map(|p| root.join(p)).and_then(|p| p.parent().map(Path::to_owned));
    let entries = match fs::read_dir(root.join("runs")) {
        Ok(entries) => entries,
        Err(error) => return (runs, vec![format!("{}: {error}", root.join("runs").display())]),
    };
    for entry in entries.flatten() {
        let Some(path) = safe_path(root, &entry.path()) else { continue; };
        if !path.is_dir() || path.file_name().is_some_and(|n| n == "parity") { continue; }
        if path.join("metrics.jsonl").is_file() { runs.insert(path.clone()); }
        if path.join("cycle.json").is_file() { cycles.insert(path.clone()); }
        let manifest = path.join("completed-cycles.json");
        if manifest.is_file() || current_parent.as_ref() == Some(&path) {
            campaigns.insert(path.clone());
        }
        if manifest.is_file() {
            let value = json(&manifest);
            if let Some(rows) = value["cycles"].as_array() {
                for row in rows {
                    if let Some(p) = row["cycle"].as_str().and_then(|s| safe_path(root, Path::new(s))) {
                        cycles.insert(p);
                    } else { errors.push(format!("{}: missing or indirect cycle {}", manifest.display(), row["cycle"])); }
                }
            } else { errors.push(format!("{}: invalid cycle inventory", manifest.display())); }
        }
    }
    if let Some(path) = current_parent.and_then(|p|safe_path(root,&p)) {
        if path != root.join("runs") { campaigns.insert(path); }
    }
    // Expansions have their own nested cycle namespace. Walk only campaign
    // containers and stop at each cycle; never traverse GP candidate trees.
    // Keep old expansion branches after the active campaign moves elsewhere.
    let mut visited = BTreeSet::new();
    while let Some(path) = campaigns.pop_first() {
        if !visited.insert(path.clone()) { continue; }
        let children = match fs::read_dir(&path) {
            Ok(children) => children,
            Err(error) => { errors.push(format!("{}: {error}",path.display())); continue; }
        };
        for child in children.flatten() {
            let Some(child) = safe_path(root, &child.path()) else { continue; };
            if !child.is_dir() { continue; }
            if child.join("cycle.json").is_file() { cycles.insert(child); }
            else if !child.file_name().is_some_and(|n|n.to_string_lossy().starts_with("cycle-"))
                && !child.join("metrics.jsonl").is_file() { campaigns.insert(child); }
        }
    }
    if let Some(path) = json(&root.join("runs/campaign-progress.json"))["current_cycle"].as_str() {
        if let Some(path) = safe_path(root, Path::new(path)) { cycles.insert(path); }
        else { errors.push(format!("Missing or indirect current cycle: {path}")); }
    }
    // Continuous blocks have their own namespace and retain every intermediate run.
    if let Some(path) = json(&root.join("runs/continuous-progress.json"))["output"].as_str()
        .and_then(|p|safe_path(root, Path::new(p))) {
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("cycle-") {
                    if let Some(path) = safe_path(root, &entry.path().join("train-2")) { runs.insert(path); }
                }
            }
        }
    }
    // Include active standalone training before its first checkpoint exists.
    if let Some(path) = json(&root.join("runs/neural-service.json"))["run"].as_str() {
        if !Path::new(path).components().any(|p| p.as_os_str() == "parity") {
            if let Some(path) = safe_path(root, Path::new(path)) { runs.insert(path); }
        }
    }
    for cycle in cycles {
        for block in ["train-1", "train-2", "auxiliary/train", "auxiliary-pack/train"] {
            if let Some(path) = safe_path(root, &cycle.join(block)) { runs.insert(path); }
            else if matches!(block,"train-1"|"train-2") && json(&cycle.join("complete.json"))["dl_gp_dl_complete"] == true {
                errors.push(format!("Missing or indirect saved training block: {}", cycle.join(block).display()));
            }
        }
    }
    (runs, errors)
}
impl FileCache {
    fn refresh(&mut self, root: &Path, path: &Path, live: &Value) {
        self.run.path = path.to_owned(); self.run.read_error = None;
        let result = self.read(root, path);
        if let Err(error) = result { self.run.read_error = Some(error.to_string()); }
        self.run.checkpoint = json(&path.join("latest.json"))["checkpoint"].as_str().unwrap_or("").to_owned();
        self.run.status = if path.join("invalidated.json").exists() { "INVALID" }
            // Resumed auxiliary blocks retain the preceding block's completion receipt.
            else if live["phase"] == "training on real GP teaching corpus"
                && live["run"].as_str().is_some_and(|p| root.join(p) == path)
                && live["timestamp"].as_u64().is_some_and(|t| t <= crate::telemetry::now() && crate::telemetry::now()-t <= 5) { "LIVE" }
            else if path.join("complete.json").exists() { "SAVED" }
            else { "PAUSED" }.into();
    }
    fn read(&mut self, root: &Path, path: &Path) -> std::io::Result<()> {
        let file_path = path.join("metrics.jsonl");
        if file_path.is_symlink() { return Err(std::io::Error::other("Indirect metric log rejected")); }
        let mut file = File::open(file_path)?; let meta = file.metadata()?;
        let inode = (meta.dev(), meta.ino()); let modified = meta.modified().ok();
        if self.inode != Some(inode) || meta.len() < self.length
            || (meta.len() == self.length && modified != self.modified) {
            *self = Self::default(); self.run.path = path.to_owned();
        }
        self.inode = Some(inode);
        if meta.len() == self.length && modified == self.modified { return Ok(()); }
        file.seek(SeekFrom::Start(self.offset))?;
        let mut reader = BufReader::new(file); let mut bytes = Vec::new();
        self.run.pending_line = false;
        loop {
            bytes.clear(); let count = reader.read_until(b'\n', &mut bytes)?;
            if count == 0 { break; }
            if bytes.last() != Some(&b'\n') { self.run.pending_line = true; break; }
            self.offset += count as u64;
            let metric = match serde_json::from_slice::<Metric>(&bytes) {
                Ok(metric) => metric,
                Err(_) => { self.run.malformed += 1; continue; }
            };
            if metric.schema != "parallel.neural_training.v1" || metric.rank != 0
                || metric.invalidated || metric.batch_qualification { continue; }
            if root.join(&metric.run) != path || metric.corpus_sha256.is_empty()
                || metric.timestamp == 0 || metric.step == 0
                || !metric.sequence_nll.is_finite() || metric.sequence_nll < 0. {
                self.run.malformed += 1; continue;
            }
            if self.run.points.last().is_none_or(|p| metric.step <= p.step) { self.run.attempts += 1; }
            self.run.corpora.insert(metric.corpus_sha256);
            if self.run.parent.is_empty() {
                if let Some(initialization) = metric.initialization {
                    self.run.parent = initialization["checkpoint"].as_str()
                        .or_else(|| initialization["kind"].as_str()).unwrap_or("unknown").to_owned();
                }
            }
            let validation = metric.validation.filter(|v| {
                if v.evaluated_step.is_none() || v.panel_sha256.as_ref().is_none_or(String::is_empty) {
                    self.run.unbound_validation += 1; return false;
                }
                if !v.sequence_nll.is_finite() || v.sequence_nll < 0. || !v.ranking_bt.is_finite() || v.ranking_bt < 0. {
                    self.run.malformed += 1; return false;
                }
                // A checkpoint value repeated on later optimizer rows is one observation.
                v.evaluated_step == Some(metric.step)
            });
            self.run.points.push(Point { step: metric.step, timestamp: metric.timestamp,
                train: metric.sequence_nll, validation, attempt: self.run.attempts });
        }
        self.length = meta.len(); self.modified = modified;
        Ok(())
    }
}
impl Cache {
    pub fn refresh(&mut self, root: &Path) -> Snapshot {
        let root = match root.canonicalize() { Ok(root) => root, Err(error) => return Snapshot { ready: true, errors: vec![error.to_string()], ..Default::default() } };
        let (paths, mut errors) = discover(&root); let live = json(&root.join("runs/neural-service.json"));
        for path in &paths { self.files.entry(path.clone()).or_default().refresh(&root, path, &live); }
        // Disappearance does not erase already observed history from a live TUI.
        for (path, cached) in &mut self.files {
            if !paths.contains(path) { cached.run.read_error = Some("Run disappeared; displaying last saved observation".into()); }
        }
        let mut runs = self.files.values().map(|c| c.run.clone()).collect::<Vec<_>>();
        runs.retain(|r| !r.points.is_empty() || !r.checkpoint.is_empty() || r.read_error.is_some());
        runs.sort_by(|a, b| (a.points.first().map(|p| p.timestamp).unwrap_or(0), &a.path)
            .cmp(&(b.points.first().map(|p| p.timestamp).unwrap_or(0), &b.path)));
        for run in &runs {
            if let Some(error) = &run.read_error { errors.push(format!("{}: {error}", run.path.join("metrics.jsonl").display())); }
        }
        Snapshot { runs, ready: true, errors }
    }
}

pub struct Watcher { receiver: mpsc::Receiver<Arc<Snapshot>>, stop: Arc<AtomicBool> }
impl Watcher {
    pub fn start(root: PathBuf) -> Self {
        let (sender, receiver) = mpsc::sync_channel(1); let stop = Arc::new(AtomicBool::new(false)); let ending = stop.clone();
        thread::spawn(move || {
            let mut cache = Cache::default();
            while !ending.load(Ordering::Relaxed) {
                let snapshot = Arc::new(cache.refresh(&root));
                if let Err(mpsc::TrySendError::Disconnected(_)) = sender.try_send(snapshot) { break; }
                for _ in 0..50 { if ending.load(Ordering::Relaxed) { return; } thread::sleep(Duration::from_millis(100)); }
            }
        });
        Self { receiver, stop }
    }
    pub fn latest(&self) -> Option<Arc<Snapshot>> { self.receiver.try_iter().last() }
}
impl Drop for Watcher { fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); } }

#[derive(Default)]
pub struct Navigation { pub selected: Option<PathBuf> }
impl Navigation {
    pub fn move_by(&mut self, snapshot: &Snapshot, delta: isize) {
        if snapshot.runs.is_empty() { return; }
        let current = self.selected.as_ref().and_then(|path| snapshot.runs.iter().position(|r| &r.path == path));
        let index = match current { Some(i) => i.saturating_add_signed(delta).min(snapshot.runs.len()-1),
            None if delta < 0 => snapshot.runs.len()-1, None => 0 };
        self.selected = Some(snapshot.runs[index].path.clone());
    }
    pub fn edge(&mut self, snapshot: &Snapshot, last: bool) {
        self.selected = if last { snapshot.runs.last() } else { snapshot.runs.first() }.map(|r| r.path.clone());
    }
}
fn title(text: &str) -> Block<'_> { Block::bordered().title(text).title_style(Style::default().fg(Color::Cyan).bold()) }
fn short(path: &Path) -> String {
    let parts = path.components().map(|p| p.as_os_str().to_string_lossy()).collect::<Vec<_>>();
    parts[parts.len().saturating_sub(if path.ends_with("auxiliary/train") || path.ends_with("auxiliary-pack/train") { 3 } else { 2 })..].join("/")
}
fn one_line(text: &str, width: u16) -> String {
    let chars=text.chars().collect::<Vec<_>>();
    if chars.len() <= width as usize { return text.into(); }
    let keep=(width as usize).saturating_sub(1)/2;
    format!("{}…{}",chars[..keep].iter().collect::<String>(),chars[chars.len()-keep..].iter().collect::<String>())
}
/// Keep first/last and each horizontal bucket's extrema; never discard a spike
/// merely because the full recorded history exceeds the terminal width.
pub fn envelope(points: &[(f64, f64)], columns: usize, bounds: [f64; 2]) -> Vec<(f64, f64)> {
    if points.len() <= columns.saturating_mul(2) { return points.to_vec(); }
    let mut bins = vec![None::<((f64, f64), (f64, f64))>; columns.max(1)];
    for &(x, y) in points {
        let index = (((x-bounds[0])/(bounds[1]-bounds[0]).max(1.) * bins.len() as f64) as usize).min(bins.len()-1);
        match &mut bins[index] { Some((low, high)) => { if y < low.1 { *low=(x,y); } if y > high.1 { *high=(x,y); } }, None => bins[index]=Some(((x,y),(x,y))) }
    }
    let mut result = bins.into_iter().flatten().flat_map(|(a,b)| [a,b]).collect::<Vec<_>>();
    if let Some(p) = points.first() { result.push(*p); } if let Some(p) = points.last() { result.push(*p); }
    result.sort_by(|a,b| a.0.total_cmp(&b.0)); result.dedup(); result
}
fn chart(f: &mut Frame, area: Rect, name: &str, points: &[(f64,f64)], bounds: [f64;2], axis: &str, color: Color) {
    if points.is_empty() { f.render_widget(Paragraph::new("No bound saved observations yet").block(title(name)), area); return; }
    let low = points.iter().map(|p|p.1).fold(f64::INFINITY,f64::min); let high = points.iter().map(|p|p.1).fold(f64::NEG_INFINITY,f64::max);
    let pad = ((high-low)*0.08).max(0.001); let data = envelope(points, area.width.saturating_sub(10) as usize, bounds);
    f.render_widget(Chart::new(vec![Dataset::default().marker(symbols::Marker::Braille).graph_type(GraphType::Scatter)
        .style(Style::default().fg(color)).data(&data)]).block(title(name))
        .x_axis(Axis::default().title(axis).bounds(bounds).labels([format!("{:.1}",bounds[0]),format!("{:.1}",bounds[1])]))
        .y_axis(Axis::default().bounds([(low-pad).max(0.),high+pad]).labels([format!("{low:.3}"),format!("{high:.3}")])),area);
}
pub fn render(f: &mut Frame, area: Rect, snapshot: &Snapshot, navigation: &Navigation, run: &Value) {
    let selected = navigation.selected.as_ref().and_then(|p|snapshot.runs.iter().position(|r| &r.path==p));
    let count: usize = snapshot.runs.iter().map(|r|r.points.len()).sum();
    let missing: usize = snapshot.runs.iter().map(|r|r.malformed).sum();
    let unbound: usize = snapshot.runs.iter().map(|r|r.unbound_validation).sum();
    let pending = snapshot.runs.iter().filter(|r|r.pending_line).count();
    let compact = area.height < 18;
    let header = if !snapshot.ready { "Loading FULL saved history in background…".into() }
        else if compact { format!("Deep learning · {} runs · {count} records",snapshot.runs.len()) }
        else { format!("FULL saved history · {} runs · {count} optimizer records · {} missing sources · {missing} malformed records", snapshot.runs.len(), snapshot.errors.len()) };
    let layout = Layout::vertical([Constraint::Length(if compact{1}else{3}),Constraint::Min(1),Constraint::Length(if compact{2}else{6})]).split(area);
    let header = Paragraph::new(header).wrap(Wrap{trim:true});
    f.render_widget(if compact{header}else{header.block(title(" Deep learning loss · restored after restart "))},layout[0]);
    let mut content = layout[1];
    if area.width >= 100 && area.height >= 24 {
        let split = Layout::horizontal([Constraint::Length(35),Constraint::Min(1)]).split(content); content=split[1];
        let visible = split[0].height.saturating_sub(2) as usize;
        let end = selected.map_or(snapshot.runs.len(), |i|(i+1).max(visible).min(snapshot.runs.len()));
        let start = end.saturating_sub(visible);
        let items = snapshot.runs[start..end].iter().enumerate().map(|(i,r)| {
            ListItem::new(format!("{} {:>3} {} {}",if selected==Some(start+i){">"}else{" "},start+i+1,r.status,short(&r.path)))
                .style(Style::default().fg(if selected==Some(start+i){Color::Cyan}else{Color::Gray}))
        }).collect::<Vec<_>>();
        f.render_widget(List::new(items).block(title(" Runs · ↑↓ / PgUp PgDn · Home End ")),split[0]);
    }
    let runs = selected.map(|i|vec![&snapshot.runs[i]]).unwrap_or_else(||snapshot.runs.iter().filter(|r|r.status!="INVALID").collect());
    let first = runs.iter().flat_map(|r|r.points.iter()).map(|p|p.timestamp).min().unwrap_or(0);
    let mut train=Vec::new(); let mut validation=Vec::new(); let mut ranking=Vec::new(); let mut panels=BTreeSet::new();
    for r in &runs { for p in &r.points {
        let x=p.timestamp.saturating_sub(first) as f64/if selected.is_some(){60.}else{3600.};
        train.push((x,p.train));
        if let Some(v)=&p.validation { validation.push((x,v.sequence_nll));ranking.push((x,v.ranking_bt));panels.insert(v.panel_sha256.as_deref().unwrap_or("")); }
    }}
    let xmax=train.iter().map(|p|p.0).fold(1.,f64::max);let bounds=[0.,xmax];
    let axis=if selected.is_some(){"minutes since run began · attempts retained"}else{"hours since first saved training record"};
    if content.height >= 18 {
        let plots=Layout::vertical([Constraint::Ratio(1,3);3]).split(content);
        chart(f,plots[0]," Training Sequence NLL · lower is better ",&train,bounds,axis,Color::Cyan);
        chart(f,plots[1]," Validation Sequence NLL · lower is better ",&validation,bounds,axis,Color::Yellow);
        chart(f,plots[2]," Validation pairwise loss · lower is better ",&ranking,bounds,axis,Color::Magenta);
    } else {
        let last=runs.iter().flat_map(|r|r.points.iter()).max_by_key(|p|p.timestamp);
        let last_val=runs.iter().flat_map(|r|r.points.iter()).filter(|p|p.validation.is_some())
            .max_by_key(|p|p.timestamp).and_then(|p|p.validation.as_ref());
        let text=format!("Training Sequence NLL: {}\nValidation Sequence NLL: {}\nValidation pairwise loss: {}\n{}\nEnlarge terminal for three full-history charts.",last.map(|p|format!("{:.5}",p.train)).unwrap_or_else(||"unavailable".into()),last_val.map(|v|format!("{:.5}",v.sequence_nll)).unwrap_or_else(||"unavailable".into()),last_val.map(|v|format!("{:.5}",v.ranking_bt)).unwrap_or_else(||"unavailable".into()),selected.map(|i|short(&snapshot.runs[i].path)).unwrap_or_else(||"All recorded campaign and auxiliary runs".into()));
        let widget=Paragraph::new(text).wrap(Wrap{trim:true});
        f.render_widget(if compact{widget}else{widget.block(title(" Loss history "))},content);
    }
    let detail=selected.map(|i| {let r=&snapshot.runs[i];format!("{} · {} records · {} attempts",r.path.display(),r.points.len(),r.attempts)})
        .unwrap_or_else(||format!("All runs, including auxiliary branches · {} validation panels; panels/corpora may differ",panels.len()));
    let checkpoint=run["selected_checkpoint"]["checkpoint"].as_str().unwrap_or("unavailable");
    let notice=snapshot.errors.first().cloned().unwrap_or_else(||format!("{unbound} unbound validation rows not plotted · {pending} incomplete final lines pending; saved logs remain unchanged"));
    let lines=if compact{vec![Line::raw("↑↓ runs · Home/End · a full · d details"),Line::raw(detail)]}else{vec![Line::raw("↑↓ select run · PgUp/PgDn page · Home oldest · End newest · a FULL history · d optimizer details"),
        Line::raw(one_line(&detail,area.width)),Line::raw(one_line(&format!("Saved checkpoint: {checkpoint}"),area.width)),
        Line::styled("Training fit is not generalization. Validation is adaptive; compare the same panel.",Style::default().fg(Color::Yellow)),
        Line::raw(one_line(&selected.map(|i|format!("Parent: {} · {} · restart boundaries retained",snapshot.runs[i].parent,snapshot.runs[i].status))
            .unwrap_or_else(||"Restart boundaries stay recorded. INVALID runs are excluded from the overview; select them to inspect.".into()),area.width)),
        Line::raw(one_line(&notice,area.width))]};
    f.render_widget(Paragraph::new(lines)
        .wrap(Wrap{trim:true}),layout[2]);
}
