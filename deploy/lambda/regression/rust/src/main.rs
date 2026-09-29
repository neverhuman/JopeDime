use feat_math::{GenomeIr, Model, Rows, V3Candidate};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn hash(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
fn now() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64 }
fn hex(s: &str) -> bool { s.len()==64 && s.bytes().all(|c|c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) }
fn memory() -> String { fs::read_to_string("/proc/self/status").unwrap_or_default().lines().filter(|l|l.starts_with("VmHWM:")||l.starts_with("VmRSS:")).collect::<Vec<_>>().join(" ") }
fn write_json(path: &Path, v: &Value) -> Result<(), Error> { let mut f=File::create(path)?; f.write_all(&serde_json::to_vec(v)?)?; f.sync_all()?; Ok(()) }

// A bounded numeric format: magic, row/feature counts, three-fold flags, then row-major f64 X,y.
// The byte hash binds every row, original membership manifest is held in the context closure.
struct Data { x: Rows, y: Vec<f64>, folds: Vec<i32>, ordered: bool }
fn data(raw: &[u8]) -> Result<Data, Error> {
    if raw.len()<17 || &raw[..4]!=b"JRG1" {return Err("invalid JRG1 header".into())}
    let n=u64::from_le_bytes(raw[4..12].try_into()?) as usize;
    let p=u32::from_le_bytes(raw[12..16].try_into()?) as usize;
    if n<12 || n>1_100_000 || p==0 || p>512 || n.checked_mul(p).unwrap_or(usize::MAX)>16_000_000 || raw[16]>1 {return Err("unsupported complete input shape".into())}
    let expected=17usize.checked_add(n.checked_mul(4+(p+1)*8).ok_or("shape overflow")?).ok_or("length overflow")?;
    if raw.len()!=expected {return Err("JRG1 byte count mismatch".into())}
    let mut pos=17; let mut folds=Vec::with_capacity(n);
    for _ in 0..n {let f=i32::from_le_bytes(raw[pos..pos+4].try_into()?);pos+=4;if !(-1..=2).contains(&f){return Err("invalid fold".into())}folds.push(f)}
    let ordered=raw[16]==1;
    if !ordered && folds.iter().any(|f|*f<0) {return Err("IID fit-only rows forbidden".into())}
    for k in 0..3 {if folds.iter().filter(|f|**f==k).count()<2 || folds.iter().filter(|f|if ordered {**f<k}else{**f!=k}).count()<5 {return Err("infeasible fold".into())}}
    let mut x=Vec::with_capacity(n);let mut y=Vec::with_capacity(n);
    for _ in 0..n {let mut row=Vec::with_capacity(p);for j in 0..=p {let v=f64::from_le_bytes(raw[pos..pos+8].try_into()?);pos+=8;if !v.is_finite(){return Err("nonfinite input".into())}if j<p{row.push(v)}else{y.push(v)}}x.push(row)}
    Ok(Data{x,y,folds,ordered})
}
fn metrics(y:&[f64],p:&[f64])->Result<Value,Error>{
    if y.is_empty()||y.len()!=p.len()||p.iter().any(|v|!v.is_finite()){return Err("invalid prediction".into())}
    let n=y.len() as f64;let mean=y.iter().sum::<f64>()/n;let sse=y.iter().zip(p).map(|(a,b)|(a-b).powi(2)).sum::<f64>();let tss=y.iter().map(|a|(a-mean).powi(2)).sum::<f64>();
    if !sse.is_finite() || !tss.is_finite(){return Err("metric overflow".into())}
    let mae=y.iter().zip(p).map(|(a,b)|(a-b).abs()).sum::<f64>()/n;
    let mape=if y.iter().all(|v|v.abs()>1e-12){Some(y.iter().zip(p).map(|(a,b)|((a-b)/a).abs()).sum::<f64>()/n)}else{None};
    let rmsle=if y.iter().chain(p.iter()).all(|v|*v>=0.){Some((y.iter().zip(p).map(|(a,b)|(a.ln_1p()-b.ln_1p()).powi(2)).sum::<f64>()/n).sqrt())}else{None};
    Ok(json!({"rmse":(sse/n).sqrt(),"mae":mae,"r2":if tss>0.{Some(1.-sse/tss)}else{None},"mape_fraction_nonzero_targets":mape,"rmsle_nonnegative_only":rmsle,"rows":y.len(),"r2_null_reason":if tss==0.{Some("constant_validation_target")}else{None}}))
}
#[derive(Clone,Debug,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Candidate { dsl:String, program_sha256:String }
fn genome(c:&Candidate)->Result<GenomeIr,Error>{
    if c.dsl.len()>32768||!hex(&c.program_sha256)||hash(c.dsl.as_bytes())!=c.program_sha256{return Err("program identity mismatch".into())}
    let g=GenomeIr::parse(&c.dsl)?;
    if g.canonical()!=c.dsl{return Err("noncanonical program".into())}
    if let GenomeIr::Pipeline(p)=&g {if p.stages.len()>9{return Err("pipeline exceeds8 transforms".into())}}
    Ok(g)
}
#[derive(Clone,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Job {context_sha256:String, input_sha256:String, input_url:String, input_bytes:usize, candidates:Vec<Candidate>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {schema:String, attempt_id:String, output_url:String, jobs:Vec<Job>, candidate_timeout_ms:u64}
fn allowed_url(url:&str,output:bool)->bool{
    let base="https://veox-jopedime-learning-use2.s3.us-east-2.amazonaws.com/experiments/regression-evolution/v1/";
    if !url.starts_with(base)||url.contains('#')||url.contains("/../"){return false}
    if output {url[base.len()..].starts_with("attempts/")}else{url[base.len()..].starts_with("objects/inputs/sha256/")}
}
fn child(data_path:&Path,candidate_path:&Path,output:&Path)->Result<(),Error>{
    let bytes=fs::read(data_path)?;let d=data(&bytes)?;let c:Candidate=serde_json::from_slice(&fs::read(candidate_path)?)?;let g=genome(&c)?;
    let mut out=OpenOptions::new().create_new(true).write(true).open(output)?;
    for fold in 0..3 {
        let started=now();let tick=Instant::now();
        let train:Vec<usize>=d.folds.iter().enumerate().filter_map(|(i,f)|if if d.ordered{*f<fold}else{*f!=fold}{Some(i)}else{None}).collect();
        let valid:Vec<usize>=d.folds.iter().enumerate().filter_map(|(i,f)|(*f==fold).then_some(i)).collect();
        let x:Rows=train.iter().map(|i|d.x[*i].clone()).collect();let y:Vec<f64>=train.iter().map(|i|d.y[*i]).collect();
        let vx:Rows=valid.iter().map(|i|d.x[*i].clone()).collect();let vy:Vec<f64>=valid.iter().map(|i|d.y[*i]).collect();
        let fit_start=now();let model=feat_math::fit(&g,&x,&y)?;let fit_end=now();
        let pred=model.predict(&vx);let metric=metrics(&vy,&pred)?;let train_metric=metrics(&y,&model.predict(&x))?;
        let model_bytes=serde_json::to_vec(&model)?;if model_bytes.len()>32*1024*1024{return Err("model exceeds32MiB archive admission".into())}
        let restored:Model=serde_json::from_slice(&model_bytes)?;let replay=restored.predict(&vx);
        if pred.iter().map(|v|v.to_bits()).collect::<Vec<_>>()!=replay.iter().map(|v|v.to_bits()).collect::<Vec<_>>(){return Err("model replay differs".into())}
        let baseline=y.iter().sum::<f64>()/y.len() as f64;let baseline_metrics=metrics(&vy,&vec![baseline;vy.len()])?;
        let row=json!({"schema":"jope.regression-fit/v1","program_sha256":c.program_sha256,"fold":fold,"train_row_positions":train,"validation_row_positions":valid,"started_utc_us":started,"fit_started_utc_us":fit_start,"fit_ended_utc_us":fit_end,"ended_utc_us":now(),"elapsed_ns_text":tick.elapsed().as_nanos().to_string(),"memory":memory(),"metrics":metric,"train_metrics":train_metric,"baseline_metrics":baseline_metrics,"baseline":"mean_of_fold_training_targets","predictions":pred,"validation_targets":vy,"model":serde_json::from_slice::<Value>(&model_bytes)?,"model_sha256":hash(&model_bytes),"model_codec":"feat_math::Model/serde_json-float-roundtrip","serialized_prediction_replay_exact":true});
        serde_json::to_writer(&mut out,&row)?;out.write_all(b"\n")?;out.flush()?;out.sync_all()?;
    }
    Ok(())
}
fn run_candidate(c:&Candidate,data_path:&Path,dir:&Path,timeout:u64)->Result<Value,Error>{
    let cp=dir.join(format!("{}.candidate.json",c.program_sha256));let op=dir.join(format!("{}.fits.jsonl",c.program_sha256));
    write_json(&cp,&serde_json::to_value(c)?)?;let started=now();let tick=Instant::now();
    let stderr_path=dir.join(format!("{}.stderr",c.program_sha256));
    let mut process=Command::new(std::env::current_exe()?).arg("--fit").arg(data_path).arg(&cp).arg(&op).stdout(Stdio::null()).stderr(Stdio::from(File::create(&stderr_path)?)).spawn()?;
    let mut timed_out=false;
    let status=loop{if let Some(s)=process.try_wait()?{break s}if tick.elapsed().as_millis()>=timeout as u128{timed_out=true;process.kill()?;break process.wait()?}std::thread::sleep(Duration::from_millis(3));};
    let mut fits=Vec::new();if op.exists(){for line in fs::read_to_string(&op)?.lines(){if let Ok(v)=serde_json::from_str::<Value>(line){fits.push(v)}}}
    let complete=status.success()&&fits.len()==3;
    let stderr=fs::read_to_string(stderr_path).unwrap_or_default();
    Ok(json!({"schema":"jope.gp-evaluation/v1","program":c,"status":if complete{"completed"}else if timed_out{"timeout"}else{"failed"},"child_exit":status.code(),"child_stderr":stderr.chars().take(8192).collect::<String>(),"started_utc_us":started,"ended_utc_us":now(),"elapsed_ns_text":tick.elapsed().as_nanos().to_string(),"folds":fits,"comparison_role":"inner_cv_search","canonical_learning_admission":false}))
}
async fn invoke(event:LambdaEvent<Request>)->Result<Value,Error>{
    let r=event.payload;let started=now();let tick=Instant::now();
    if r.schema!="jope.regression-request/v1"||r.jobs.is_empty()||r.jobs.len()>5||!(100..=30000).contains(&r.candidate_timeout_ms)||r.attempt_id.is_empty()||!r.attempt_id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-')||!allowed_url(&r.output_url,true){return Err("invalid request".into())}
    // Presigned write is bound to one exact immutable attempt path; no ambient S3 permissions.
    let expected=format!("https://veox-jopedime-learning-use2.s3.us-east-2.amazonaws.com/experiments/regression-evolution/v1/attempts/{}.json.gz?",r.attempt_id);
    if !r.output_url.starts_with(&expected){return Err("output capability differs from attempt".into())}
    let dir=PathBuf::from(format!("/tmp/jope-{}",r.attempt_id));fs::create_dir(&dir)?;
    let agent:ureq::Agent=ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).build().into();
    let mut records=Vec::new();
    println!("{}",json!({"event":"handler_start","attempt_id":r.attempt_id,"request_id":event.context.request_id,"utc_us":started}));
    if r.jobs.iter().map(|j|j.input_bytes).sum::<usize>()>256*1024*1024{return Err("batch input bytes exceed256MiB".into())}
    // Validate and download the entire batch before the first fit; a later bad
    // input must not strand successful fits from an earlier dataset.
    for (ji,j) in r.jobs.iter().enumerate(){
        if !hex(&j.context_sha256)||!hex(&j.input_sha256)||!allowed_url(&j.input_url,false)||j.input_bytes>256*1024*1024||j.candidates.len()>16 {return Err("job admission failed".into())}
        let mut seen=BTreeSet::new();for c in &j.candidates {genome(c)?;if !seen.insert(&c.program_sha256){return Err("duplicate requested program".into())}}
        let mut resp=agent.get(&j.input_url).call()?;let compressed=resp.body_mut().as_reader().take(128*1024*1024+1).read_to_vec()?;
        let mut raw=Vec::new();GzDecoder::new(compressed.as_slice()).take(256*1024*1024+1).read_to_end(&mut raw)?;
        if raw.len()!=j.input_bytes||hash(&raw)!=j.input_sha256{return Err("input payload identity mismatch".into())}
        let dp=dir.join(format!("data-{ji}.bin"));fs::write(&dp,&raw)?;drop(raw);
    }
    for (ji,j) in r.jobs.iter().enumerate(){
        let dp=dir.join(format!("data-{ji}.bin"));
        let subdir=dir.join(format!("job-{ji}"));fs::create_dir(&subdir)?;
        for c in &j.candidates{
            if now()/1000+r.candidate_timeout_ms+25000>=event.context.deadline {records.push(json!({"schema":"jope.gp-evaluation/v1","context_sha256":j.context_sha256,"program":c,"status":"not_started_deadline","folds":[],"comparison_role":"inner_cv_search","canonical_learning_admission":false}));continue}
            let mut row=run_candidate(c,&dp,&subdir,r.candidate_timeout_ms)?;row["context_sha256"]=json!(j.context_sha256);
            println!("{}",json!({"event":"candidate_end","attempt_id":r.attempt_id,"context_sha256":j.context_sha256,"program_sha256":c.program_sha256,"status":row["status"],"utc_us":now()}));records.push(row);
        }
    }
    let pack=json!({"schema":"jope.regression-invocation/v1","attempt_id":r.attempt_id,"aws_request_id":event.context.request_id,"started_utc_us":started,"ended_utc_us":now(),"elapsed_ns_text":tick.elapsed().as_nanos().to_string(),"memory":memory(),"architecture":std::env::consts::ARCH,"records":records});
    let bytes=serde_json::to_vec(&pack)?;let mut gz=GzEncoder::new(Vec::new(),Compression::fast());gz.write_all(&bytes)?;let compressed=gz.finish()?;
    if compressed.len()>32*1024*1024{return Err("archive exceeds32MiB reservation".into())}
    agent.put(&r.output_url).header("If-None-Match","*").header("Content-Type","application/gzip").send(&compressed)?;
    let result=json!({"schema":"jope.regression-invocation-receipt/v1","attempt_id":r.attempt_id,"archive_sha256":hash(&compressed),"archive_bytes":compressed.len(),"uncompressed_sha256":hash(&bytes),"records":pack["records"].as_array().unwrap().len(),"started_utc_us":started,"archive_completed_utc_us":now(),"handler_elapsed_ns_text":tick.elapsed().as_nanos().to_string()});
    fs::remove_dir_all(&dir)?;Ok(result)
}
trait ReadAll:Read {fn read_to_vec(mut self)->std::io::Result<Vec<u8>> where Self:Sized {let mut v=Vec::new();self.read_to_end(&mut v)?;Ok(v)}}
impl<T:Read> ReadAll for T {}

fn propose(v:Value)->Result<Value,Error>{
    let parents=v["parents"].as_array().ok_or("parents")?;let count=v["count"].as_u64().ok_or("count")?.min(256);let offset=v["offset"].as_u64().unwrap_or(0);let seed=v["seed"].as_u64().ok_or("seed")?;let generation=v["generation"].as_u64().unwrap_or(0) as usize;
    let mut out=Vec::new();let mut seen=BTreeSet::new();
    for draw in offset..offset+count*16{
        let s=seed.wrapping_add(draw.wrapping_mul(0x9e3779b97f4a7c15));let source=if parents.is_empty()||draw%5==0{feat_math::evolve::random_canonical_pipeline(s,4)}else{
            let a=parents[draw as usize%parents.len()].as_str().ok_or("parent DSL")?;
            let p=V3Candidate{syntax:a.into(),semantic_hash:hash(a.as_bytes()),lineage:vec![],island:0,operator:"parent".into(),step_micros:250000,generation:generation.saturating_sub(1)};
            if draw%5==1 && parents.len()>1{let mut b=p.clone();b.syntax=parents[(draw as usize+1)%parents.len()].as_str().ok_or("parent DSL")?.into();b.semantic_hash=hash(b.syntax.as_bytes());feat_math::crossover_candidates(&p,&b,s,generation,0)?.syntax}else{feat_math::mutate_candidate(&p,s,generation,0)?.syntax}
        };
        let g=GenomeIr::parse(&source)?;
        if let GenomeIr::Pipeline(p)=&g {if p.stages.len()>9{continue}}
        let dsl=g.canonical();let sha=hash(dsl.as_bytes());
        if !seen.insert(sha.clone()){continue}
        out.push(json!({"dsl":dsl,"program_sha256":sha,"draw":draw,"seed":s,"operator":if parents.is_empty()||draw%5==0{"grammar_random"}else if draw%5==1 && parents.len()>1{"typed_crossover"}else{"typed_mutation"}}));if out.len()>=count as usize{break}
    }
    Ok(json!({"schema":"jope.regression-proposals/v1","proposals":out}))
}
#[tokio::main(flavor="current_thread")]
async fn main()->Result<(),Error>{
    let a:Vec<String>=std::env::args().collect();
    if a.get(1).map(|x|x.as_str())==Some("--fit"){return child(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]))}
    if let Some(mode)=a.get(1){let v:Value=serde_json::from_reader(std::io::stdin())?;let out=match mode.as_str(){"--propose"=>propose(v)?,"--canonical-json"=>v,"--canonical-programs"=>json!(v.as_array().ok_or("array")?.iter().map(|s|match s.as_str().and_then(|s|GenomeIr::parse(s).ok()){Some(g)=>{let d=g.canonical();json!({"program_sha256":hash(d.as_bytes()),"dsl":d})},None=>json!({"error":"not_parseable"})}).collect::<Vec<_>>()),"--canonical-program"=>{let d=GenomeIr::parse(v["dsl"].as_str().ok_or("dsl")?)?.canonical();json!({"program_sha256":hash(d.as_bytes()),"dsl":d})},_=>return Err("unknown mode".into())};println!("{}",serde_json::to_string(&out)?);return Ok(())}
    lambda_runtime::run(service_fn(invoke)).await
}

#[cfg(test)]mod tests{
 use super::*;
 fn fixture()->Vec<u8>{let mut b=b"JRG1".to_vec();b.extend(18u64.to_le_bytes());b.extend(2u32.to_le_bytes());b.push(0);for i in 0i32..18{b.extend((i%3).to_le_bytes())}for i in 0..18{for v in [i as f64,(i*i) as f64,3.*i as f64+1.]{b.extend(v.to_le_bytes())}}b}
 #[test]fn format_rejects_truncation_and_nonfinite(){let a=fixture();assert_eq!(data(&a).unwrap().x.len(),18);assert!(data(&a[..a.len()-1]).is_err());let mut b=a.clone();b[89..97].copy_from_slice(&f64::NAN.to_le_bytes());assert!(data(&b).is_err());}
 #[test]fn metrics_handle_constant_and_domains(){let m=metrics(&[1.,1.],&[1.,1.]).unwrap();assert_eq!(m["rmse"],0.);assert!(m["r2"].is_null());let m=metrics(&[0.,-1.],&[1.,-2.]).unwrap();assert!(m["mape_fraction_nonzero_targets"].is_null());assert!(m["rmsle_nonnegative_only"].is_null());assert!(metrics(&[1.],&[f64::NAN]).is_err());}
 #[test]fn roundtrip_and_train_fold_isolation(){let mut d=data(&fixture()).unwrap();let run=|d:&Data|{let tr:Vec<_>=(0..18).filter(|i|d.folds[*i]!=0).collect();let x=tr.iter().map(|i|d.x[*i].clone()).collect();let y:Vec<f64>=tr.iter().map(|i|d.y[*i]).collect();let m=feat_math::fit(&GenomeIr::parse("(pipe X (ridge lambda=0.1))").unwrap(),&x,&y).unwrap();let b=serde_json::to_vec(&m).unwrap();let restored:Model=serde_json::from_slice(&b).unwrap();assert_eq!(m.predict(&d.x),restored.predict(&d.x));b};let before=run(&d);for i in 0..18{if d.folds[i]==0{d.y[i]=999999.}}assert_eq!(before,run(&d));}
 #[test]fn proposal_replay_is_exact(){let v=json!({"parents":["(pipe X (ridge lambda=0.1))"],"count":16,"seed":42,"generation":1});assert_eq!(propose(v.clone()).unwrap(),propose(v).unwrap());}
 #[test]fn url_scope(){assert!(!allowed_url("https://example.com/foo",true));assert!(!allowed_url("https://veox-jopedime-learning-use2.s3.us-east-2.amazonaws.com/hwm/x",false));}
}
