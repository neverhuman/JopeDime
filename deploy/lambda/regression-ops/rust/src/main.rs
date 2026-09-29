use feat_math::{Model, Rows};
use flate2::read::GzDecoder;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io::Read, time::{Instant,SystemTime,UNIX_EPOCH}};
type E=Box<dyn Error>;
fn hash(b:&[u8])->String{format!("{:x}",Sha256::digest(b))}
fn now()->u64{SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64}
fn rmse(y:&[f64],p:&[f64])->Result<f64,E>{if y.is_empty()||y.len()!=p.len()||p.iter().any(|x|!x.is_finite()){return Err("nonfinite/invalid predictions".into())}let v=(y.iter().zip(p).map(|(a,b)|(a-b).powi(2)).sum::<f64>()/y.len() as f64).sqrt();if !v.is_finite(){return Err("metric overflow".into())}Ok(v)}
fn step(s:&mut u64)->u64{*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;*s}
fn permutation(n:usize,seed:u64)->Vec<usize>{let mut p:Vec<usize>=(0..n).collect();let mut s=seed|1;for i in (1..n).rev(){let j=step(&mut s) as usize%(i+1);p.swap(i,j)}p}
fn assess(req:Value)->Result<Value,E>{
 let started=now();let tick=Instant::now();let fold:Value=serde_json::from_slice(&fs::read(req["fold_path"].as_str().ok_or("fold path")?)?)?;
 let model:Model=serde_json::from_value(fold["model"].clone())?;
 if hash(&serde_json::to_vec(&model)?)!=fold["model_sha256"].as_str().ok_or("model hash")?{return Err("typed model hash mismatch".into())}
 let input=fs::read(req["input_path"].as_str().ok_or("input path")?)?;let mut raw=Vec::new();GzDecoder::new(input.as_slice()).take(256*1024*1024+1).read_to_end(&mut raw)?;
 if hash(&raw)!=req["input_sha256"].as_str().ok_or("input hash")?||raw.len()<17||&raw[..4]!=b"JRG1"{return Err("input binding".into())}
 let n=u64::from_le_bytes(raw[4..12].try_into()?) as usize;let width=u32::from_le_bytes(raw[12..16].try_into()?) as usize;
 if width==0||width>512||n>1_100_000||raw.len()!=17+n*4+n*(width+1)*8{return Err("input bounds".into())}
 let rows=fold["validation_row_positions"].as_array().ok_or("validation rows")?;let predictions=fold["predictions"].as_array().ok_or("predictions")?;let targets=fold["validation_targets"].as_array().ok_or("targets")?;
 if rows.len()!=predictions.len()||rows.len()!=targets.len(){return Err("fold vector length".into())}
 let seed=u64::from_str_radix(&hash(format!("{}:{}",req["context_sha256"],fold["program_sha256"]).as_bytes())[..16],16)?;
 let mut positions:Vec<usize>=(0..rows.len()).collect();positions.sort_by_key(|i|hash(format!("{seed}:row:{i}").as_bytes()));positions.truncate(req["row_limit"].as_u64().unwrap_or(256).min(4096) as usize);positions.sort();
 let mut x=Rows::new();let mut y=Vec::new();let mut archived=Vec::new();let mut row_ids=Vec::new();
 for k in positions {let i=rows[k].as_u64().ok_or("row index")? as usize;if i>=n{return Err("row out of bounds".into())}let offset=17+n*4+i*(width+1)*8;let v:Vec<f64>=(0..=width).map(|j|f64::from_le_bytes(raw[offset+j*8..offset+j*8+8].try_into().unwrap())).collect();if v.iter().any(|x|!x.is_finite())||v[width].to_bits()!=targets[k].as_f64().ok_or("target")?.to_bits(){return Err("fold target/input mismatch".into())}x.push(v[..width].to_vec());y.push(v[width]);archived.push(predictions[k].as_f64().ok_or("prediction")?);row_ids.push(i);}
 let baseline=model.predict(&x);let max_abs=baseline.iter().zip(&archived).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
 if baseline.iter().zip(&archived).any(|(a,b)|(a-b).abs()>1e-8+1e-8*b.abs()){return Err("cross-architecture model replay differs beyond declared tolerance".into())}
 let loss=rmse(&y,&baseline)?;let mut features:Vec<usize>=(0..width).collect();features.sort_by_key(|i|hash(format!("{seed}:feature:{i}").as_bytes()));features.truncate(req["feature_limit"].as_u64().unwrap_or(32).min(512) as usize);features.sort();
 let repeats=req["permutation_repeats"].as_u64().unwrap_or(3).min(16);let mut output=Vec::new();
 for j in &features {for repeat in 0..repeats {let pseed=seed.wrapping_add((*j as u64).wrapping_mul(0x9e3779b97f4a7c15)).wrapping_add(repeat);let order=permutation(x.len(),pseed);let mut altered=x.clone();for i in 0..x.len(){altered[i][*j]=x[order[i]][*j]}let permuted=rmse(&y,&model.predict(&altered))?;output.push(json!({"input_feature_index":j,"permutation_repeat":repeat,"seed_u64_text":pseed.to_string(),"rmse_original_sample":loss,"rmse_permuted":permuted,"delta_rmse":permuted-loss}));}}
 Ok(json!({"schema":"jope.gp-permutation-importance/v1","context_sha256":req["context_sha256"],"program_sha256":fold["program_sha256"],"inner_fold":fold["fold"],"model_sha256":fold["model_sha256"],"input_sha256":req["input_sha256"],"validation_row_positions":row_ids,"original_validation_rows":rows.len(),"input_feature_indices":features,"original_features":width,"rows_sampled":x.len(),"permutation_repeats":repeats,"importance":output,"baseline_sample_rmse":loss,"model_hash_replay_verified":true,"prediction_max_abs_difference":max_abs,"prediction_tolerance":"abs<=1e-8+1e-8*abs(archived prediction)","assessment_architecture":std::env::consts::ARCH,"started_utc_us":started,"ended_utc_us":now(),"elapsed_ns_text":tick.elapsed().as_nanos().to_string(),"new_training_fits":0,"label_role":"post-freeze-inner-validation-diagnostic","interpretation":"marginal permutation sensitivity, not causal importance; row/feature sampling is label-blind; unavailable features are not zero importance"}))
}
fn main()->Result<(),E>{let request:Value=serde_json::from_reader(std::io::stdin())?;println!("{}",serde_json::to_string(&assess(request)?)?);Ok(())}
#[cfg(test)]mod tests{use super::*;#[test]fn permutation_is_replayable_bijection(){let p=permutation(101,42);assert_eq!(p,permutation(101,42));let mut s=p;s.sort();assert_eq!(s,(0..101).collect::<Vec<_>>());}#[test]fn metrics_reject_bad_values(){assert!(rmse(&[1.],&[f64::NAN]).is_err());assert_eq!(rmse(&[1.,2.],&[1.,2.]).unwrap(),0.);}}
