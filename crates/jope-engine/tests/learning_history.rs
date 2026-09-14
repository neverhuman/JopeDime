use jope_parallel::{learning_history::{Cache, Navigation, column_line, envelope, moving_average, trend_row}, dashboard::{self, History}, telemetry::Hardware};
use ratatui::{backend::TestBackend, Terminal};
use serde_json::{json, Value};
use std::{fs, io::Write, path::{Path, PathBuf}, sync::Arc};

fn write(path: &Path, value: Value) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}
fn row(path: &str, step: u64, timestamp: u64) -> Value {
    json!({"schema":"parallel.neural_training.v1","run":path,"rank":0,"step":step,"timestamp":timestamp,
        "sequence_nll":1.0/step as f64,"corpus_sha256":"frozen-corpus","initialization":{"kind":"warm_start","checkpoint":"runs/parent/checkpoints/step-3000"},
        "validation":{"sequence_nll":0.9,"ranking_bt":0.6,"evaluated_step":step,"panel_sha256":"fixed-panel"}})
}
fn append(root: &Path, path: &str, values: &[Value]) -> PathBuf {
    let file=root.join(path).join("metrics.jsonl");fs::create_dir_all(file.parent().unwrap()).unwrap();
    let mut stream=fs::OpenOptions::new().create(true).append(true).open(&file).unwrap();
    for value in values { writeln!(stream,"{value}").unwrap(); }
    file
}

#[test]
fn resumed_auxiliary_is_live_only_with_fresh_bound_training_receipt() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();let path="runs/training";
    let now=jope_parallel::telemetry::now();
    let file=append(root,path,&[row(path,300,now-10),row(path,301,now)]);
    write(&root.join(path).join("complete.json"),json!({"step":300}));
    let original=fs::read(&file).unwrap();let mut cache=Cache::default();
    let live=json!({"run":path,"phase":"training on real GP teaching corpus","timestamp":now});
    for (key,value,expected) in [
        ("phase",json!("training on real GP teaching corpus"),"LIVE"),
        ("phase",json!("complete"),"SAVED"),
        ("timestamp",json!(now-60),"SAVED"),
        ("timestamp",json!(now+60),"SAVED"),
        ("run",json!("runs/unrelated"),"SAVED"),
    ] {
        let mut receipt=live.clone();receipt[key]=value;
        write(&root.join("runs/neural-service.json"),receipt);
        let warm=cache.refresh(root);let cold=Cache::default().refresh(root);
        assert_eq!(warm.runs[0].status,expected);assert_eq!(cold.runs[0].status,expected);
        assert_eq!(warm.runs[0].points.len(),2);assert_eq!(cold.runs[0].points.len(),2);
    }
    write(&root.join("runs/neural-service.json"),live);
    write(&root.join(path).join("invalidated.json"),json!({"reason":"withdrawn"}));
    assert_eq!(cache.refresh(root).runs[0].status,"INVALID");
    assert_eq!(Cache::default().refresh(root).runs[0].status,"INVALID");
    assert_eq!(fs::read(file).unwrap(),original);
}

#[test]
fn full_history_survives_restart_and_includes_old_auxiliary_and_unlisted_cycles() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();let mut cycles=Vec::new();
    for i in 0..24 {
        let cycle=format!("runs/campaign/cycle-{i:04}");
        write(&root.join(&cycle).join("cycle.json"),json!({}));
        let path=format!("{cycle}/train-1");append(root,&path,&[row(&path,1,100+i)]);
        if i<23 {cycles.push(json!({"cycle":cycle}));}
    }
    write(&root.join("runs/campaign/completed-cycles.json"),json!({"cycles":cycles}));
    let auxiliary="runs/campaign/cycle-0000/auxiliary/train";append(root,auxiliary,&[row(auxiliary,1,101)]);
    let packing="runs/campaign/cycle-0000/auxiliary-pack/train";append(root,packing,&[row(packing,1,102)]);
    let first=Cache::default().refresh(root);let restart=Cache::default().refresh(root);
    assert_eq!(first.runs.len(),26);assert_eq!(format!("{first:?}"),format!("{restart:?}"));
    assert_eq!(first.report(),restart.report());assert_eq!(first.report()["records"],26);assert_eq!(first.report()["read_only"],true);
    assert!(first.runs.iter().any(|r|r.path.ends_with("cycle-0000/train-1")));
    assert!(first.runs.iter().any(|r|r.path.ends_with("cycle-0023/train-1")));
    assert!(first.runs.iter().any(|r|r.path.ends_with("auxiliary/train")));
    assert!(first.runs.iter().any(|r|r.path.ends_with("auxiliary-pack/train")));
}

#[test]
fn complete_log_is_not_truncated_to_a_recent_byte_tail() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();let path="runs/training";
    let mut values=Vec::new();
    for i in 1..=9000 {let mut v=row(path,i,100+i);v["unrelated_debug_detail"]="x".repeat(1000).into();values.push(v);}
    let file=append(root,path,&values);assert!(file.metadata().unwrap().len()>8*1024*1024);
    let history=Cache::default().refresh(root);assert_eq!(history.runs[0].points.len(),9000);
    assert_eq!(history.runs[0].points[0].step,1);assert_eq!(history.runs[0].points.last().unwrap().step,9000);
}

#[test]
fn nested_expansion_history_survives_switching_campaigns_and_cold_restart() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();
    write(&root.join("runs/campaign/completed-cycles.json"),json!({"cycles":[]}));
    for branch in ["expansion-old","expansion-current"] {
        for i in 0..3 {
            let cycle=format!("runs/campaign/{branch}/cycle-{i:04}");
            write(&root.join(&cycle).join("cycle.json"),json!({}));
            let path=format!("{cycle}/train-1");append(root,&path,&[row(&path,1,100+i)]);
        }
    }
    // A candidate named like a training run must never be discovered.
    let candidate="runs/campaign/expansion-old/cycle-0000/gp/candidate/train-1";
    append(root,candidate,&[row(candidate,1,1)]);
    write(&root.join("runs/campaign-progress.json"),json!({"current_cycle":"runs/campaign/expansion-current/cycle-0002"}));
    let first=Cache::default().refresh(root);assert_eq!(first.runs.len(),6);
    write(&root.join("runs/campaign-progress.json"),json!({"current_cycle":"runs/different/cycle-0000"}));
    let cold=Cache::default().refresh(root);assert_eq!(cold.runs.len(),6);
    assert_eq!(cold.report()["runs"],first.report()["runs"]);
}

#[test]
fn standalone_current_cycle_does_not_scan_unrelated_gp_trees() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();
    let cycle="runs/standalone-cycle";
    write(&root.join(cycle).join("cycle.json"),json!({}));
    let run=format!("{cycle}/train-1");append(root,&run,&[row(&run,1,100)]);
    let fake="runs/gp-root/candidates/deep/cycle";
    write(&root.join(fake).join("cycle.json"),json!({}));
    let candidate=format!("{fake}/train-1");append(root,&candidate,&[row(&candidate,1,101)]);
    write(&root.join("runs/campaign-progress.json"),json!({"current_cycle":cycle}));
    let history=Cache::default().refresh(root);
    assert_eq!(history.runs.len(),1);assert_eq!(history.runs[0].path,root.join(run));
}

#[test]
fn compact_overview_uses_latest_validation_timestamp_across_overlapping_runs() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();
    let mut latest=row("runs/earlier-start",2,300);latest["validation"]["sequence_nll"]=0.12345.into();
    append(root,"runs/earlier-start",&[row("runs/earlier-start",1,100),latest]);
    append(root,"runs/later-start",&[row("runs/later-start",1,200),row("runs/later-start",2,220)]);
    let mut history=History::default();history.learning=Arc::new(Cache::default().refresh(root));
    let mut terminal=Terminal::new(TestBackend::new(48,15)).unwrap();
    terminal.draw(|f|dashboard::render(f,2,&Hardware::default(),&Value::Null,&Value::Null,&json!({}),"",&history)).unwrap();
    let text=terminal.backend().buffer().content.iter().map(|c|c.symbol()).collect::<String>();
    assert!(text.contains("Validation Sequence NLL: 0.12345"));
}

#[test]
fn partial_append_and_corrupt_complete_records_are_visible_without_repair_writes() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();let path="runs/training";
    let file=append(root,path,&[row(path,1,100)]);let mut stream=fs::OpenOptions::new().append(true).open(&file).unwrap();
    writeln!(stream,"broken complete record").unwrap();let second=row(path,2,101).to_string();
    stream.write_all(second[..20].as_bytes()).unwrap();let original=fs::read(&file).unwrap();
    let mut cache=Cache::default();let first=cache.refresh(root);
    assert_eq!(first.runs[0].points.len(),1);assert_eq!(first.runs[0].malformed,1);assert!(first.runs[0].pending_line);
    assert_eq!(fs::read(&file).unwrap(),original);
    stream.write_all(second[20..].as_bytes()).unwrap();stream.write_all(b"\n").unwrap();
    let second=cache.refresh(root);assert_eq!(second.runs[0].points.len(),2);assert!(!second.runs[0].pending_line);
    assert_eq!(second.runs[0].malformed,1);assert_eq!(format!("{second:?}"),format!("{:?}",Cache::default().refresh(root)));
}

#[test]
fn resumed_steps_remain_attempts_and_validation_values_are_not_repeated_or_invented() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();let path="runs/training";
    let a=row(path,1,100);let mut b=row(path,2,101);b["validation"]["evaluated_step"]=1.into();
    let mut c=row(path,1,102);c["validation"].as_object_mut().unwrap().remove("evaluated_step");
    append(root,path,&[a,b,c,row(path,2,103)]);
    let history=Cache::default().refresh(root);let r=&history.runs[0];
    assert_eq!(r.points.len(),4);assert_eq!(r.attempts,2);assert_eq!(r.points[2].attempt,2);
    assert_eq!(r.points.iter().filter(|p|p.validation.is_some()).count(),2);assert_eq!(r.unbound_validation,1);
}

#[test]
fn replaced_log_reloads_and_missing_source_retains_explicit_last_observation() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();let path="runs/training";
    let file=append(root,path,&[row(path,1,100),row(path,2,101)]);let mut cache=Cache::default();cache.refresh(root);
    let replacement=file.with_extension("replacement");fs::write(&replacement,format!("{}\n",row(path,3,102))).unwrap();fs::rename(replacement,&file).unwrap();
    let changed=cache.refresh(root);assert_eq!(changed.runs[0].points.len(),1);assert_eq!(changed.runs[0].points[0].step,3);
    fs::remove_file(file).unwrap();let missing=cache.refresh(root);assert_eq!(missing.runs[0].points.len(),1);assert_eq!(missing.errors.len(),1);
}

#[test]
fn unrelated_or_indirect_metrics_do_not_become_production_learning() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();let path="runs/training";
    let mut wrong=row("runs/another-model",1,100);let mut probe=row(path,2,101);probe["batch_qualification"]=true.into();
    let mut rank=row(path,3,102);rank["rank"]=1.into();append(root,path,&[wrong.clone(),probe,rank,row(path,4,103)]);
    let outside=tempfile::tempdir().unwrap();std::os::unix::fs::symlink(outside.path(),root.join("runs/indirect")).unwrap();
    let history=Cache::default().refresh(root);assert_eq!(history.runs.len(),1);assert_eq!(history.runs[0].points.len(),1);assert_eq!(history.runs[0].malformed,1);
    wrong["run"]=path.into();wrong["rank"]=true.into();append(root,path,&[wrong]);
    assert_eq!(Cache::default().refresh(root).runs[0].malformed,2);
}

#[test]
fn navigation_reaches_both_ends_and_all_view_sizes_remain_usable() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();
    for i in 0..30 {let p=format!("runs/train-{i:02}");append(root,&p,&[row(&p,1,100+i),row(&p,2,101+i)]);}
    let snapshot=Cache::default().refresh(root);let mut nav=Navigation::default();
    nav.edge(&snapshot,false);assert_eq!(nav.selected.as_ref(),Some(&snapshot.runs[0].path));
    nav.move_by(&snapshot,-20);assert_eq!(nav.selected.as_ref(),Some(&snapshot.runs[0].path));
    nav.edge(&snapshot,true);assert_eq!(nav.selected.as_ref(),Some(&snapshot.runs[29].path));
    nav.move_by(&snapshot,20);assert_eq!(nav.selected.as_ref(),Some(&snapshot.runs[29].path));
    let mut history=History::default();history.learning=Arc::new(snapshot);history.learning_navigation=nav;
    for (width,height) in [(160,55),(100,36),(48,15),(20,8)] {
        let mut terminal=Terminal::new(TestBackend::new(width,height)).unwrap();
        terminal.draw(|f|dashboard::render(f,2,&Hardware::default(),&Value::Null,&Value::Null,&json!({}),"",&history)).unwrap();
        let text=terminal.backend().buffer().content.iter().map(|c|c.symbol()).collect::<String>();
        if width>=48 {assert!(text.contains("Sequence NLL"));assert!(text.contains("Deep learning"));}
        if width>=100 {assert!(text.contains("Validation is adaptive"));assert!(text.contains("unbound validation rows"));}
    }
}

#[test]
fn full_history_plot_retains_spikes_and_both_endpoints() {
    let mut points=(0..10000).map(|i|(i as f64,1.)).collect::<Vec<_>>();points[7133].1=100.;points[9999].1=0.1;
    let plot=envelope(&points,80,[0.,9999.]);assert!(plot.len()<=162);
    assert!(plot.contains(&points[0]));assert!(plot.contains(&points[7133]));assert!(plot.contains(&points[9999]));
}

#[test]
fn moving_average_follows_the_global_trend_and_damps_a_spike() {
    let mut points=(0..100).map(|i|(i as f64, 10.0 - i as f64 * 0.05)).collect::<Vec<_>>();
    points[50].1 = 80.0;
    let smooth=moving_average(&points);
    assert_eq!(smooth.first().map(|p|p.0), Some(0.0));
    assert!((smooth.first().unwrap().1 - 10.0).abs() < 1.0);
    assert!((smooth.last().unwrap().1 - points.last().unwrap().1).abs() < 1.0);
    let mid=smooth.iter().find(|p| (p.0-50.0).abs()<0.1).unwrap();
    assert!(mid.1 < 20.0, "spike should not dominate the global line: {}", mid.1);
    assert!(smooth.windows(2).all(|w| w[0].0 <= w[1].0));
}

#[test]
fn selected_run_keeps_the_three_global_history_charts() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path();
    let mut cycles=Vec::new();
    for i in 0..4 {
        let cycle=format!("runs/campaign/cycle-{i:04}");
        write(&root.join(&cycle).join("cycle.json"),json!({}));
        let path=format!("{cycle}/train-2");
        write(&root.join(&path).join("complete.json"),json!({"step":2}));
        append(root,&path,&[row(&path,1,1_000+i*3_600),row(&path,2,1_100+i*3_600)]);
        cycles.push(json!({"cycle":cycle}));
    }
    write(&root.join("runs/campaign/completed-cycles.json"),json!({"cycles":cycles}));
    let snapshot=Cache::default().refresh(root);let mut nav=Navigation::default();
    nav.edge(&snapshot,true);
    let mut history=History::default();history.learning=Arc::new(snapshot);history.learning_navigation=nav;
    let mut terminal=Terminal::new(TestBackend::new(160,48)).unwrap();
    terminal.draw(|f|dashboard::render(f,2,&Hardware::default(),&Value::Null,&Value::Null,&json!({}),"",&history)).unwrap();
    let text=terminal.backend().buffer().content.iter().map(|c|c.symbol()).collect::<String>();
    assert!(text.contains("global SMA"));
    assert!(text.contains("hours since first saved"));
    assert!(!text.contains("minutes since run began"));
    assert!(text.contains("Training Sequence NLL"));
    assert!(text.contains("Validation Sequence NLL"));
    assert!(text.contains("Validation pairwise loss"));
    let line = text.chars().filter(|c| "─│╭╮╰╯╱╲".contains(*c)).count();
    assert!(line >= 40, "expected a connected trend line, found {line} stroke cells in:\n{text}");
}

#[test]
fn trend_row_uses_the_chart_height_for_a_large_loss_drop() {
    let height = 10;
    let top = trend_row(8.0, 0.4, 8.0, height, true);
    let bot = trend_row(0.4, 0.4, 8.0, height, true);
    assert_eq!(top, 0);
    assert_eq!(bot, 9);
    assert!(trend_row(1.0, 0.4, 8.0, height, true) < 8);
}

#[test]
fn column_line_covers_every_horizontal_cell() {
    let points = (0..40).map(|i|(i as f64, 8.0 - i as f64 * 0.1)).collect::<Vec<_>>();
    let line = column_line(&points, 20, [0., 39.]);
    assert_eq!(line.len(), 20);
    assert!((line[0].0 - 0.0).abs() < 1e-9);
    assert!((line[19].0 - 39.0).abs() < 1e-9);
    assert!(line.windows(2).all(|w| w[0].0 < w[1].0));
    assert!(line[0].1 > line[19].1);
}
