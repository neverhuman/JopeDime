//! Test-only, actual host qualification of a completed frozen model.
use anyhow::{ensure, Context, Result};
use jope_runtime::model_object::{Manifest, ModelObject};
use jope_service::{
    events::EventLog,
    export::bundle::{record_model_parity, MeasuredModelParity},
    store::{RecordKey, Store},
};
use std::{collections::BTreeMap, fs, path::Path, process::Command, sync::Arc};
pub(super) fn from_env(home: &Path) -> Result<()> {
    let run: jope_api::ids::RunId = std::env::var("JOPE_INT01_QUALIFY_RUN")?.parse()?;
    let csv =
        fs::read(std::env::var_os("JOPE_INT01_QUALIFY_CSV").context("qualification CSV missing")?)?;
    ensure!(
        jope_runtime::sha256_hex(&csv) == std::env::var("JOPE_INT01_QUALIFY_SHA256")?,
        "qualification input changed"
    );
    let out = std::path::PathBuf::from(
        std::env::var_os("JOPE_INT01_QUALIFY_OUT").context("qualification evidence missing")?,
    );
    let store = Arc::new(Store::open(home)?);
    let snapshot = EventLog::new(Arc::clone(&store))
        .snapshot(&run)?
        .context("completed run missing")?;
    ensure!(
        snapshot.state == jope_api::RunState::Completed,
        "qualify only completed models"
    );
    let artifact = snapshot
        .best
        .as_ref()
        .and_then(|b| b.model_artifact.as_ref())
        .context("completed model missing")?;
    let record = store
        .get(&RecordKey::Artifact(artifact.clone()))?
        .context("model record missing")?;
    let manifest_bytes = store.read_blob(&record.blobs[0])?;
    let manifest: Manifest = jope_runtime::strict_json::decode(&manifest_bytes)?;
    let mut parts = BTreeMap::new();
    for part in &manifest.files {
        let reference = record
            .blobs
            .iter()
            .find(|b| {
                b.bytes == part.bytes
                    && b.sha256
                        .iter()
                        .map(|v| format!("{v:02x}"))
                        .collect::<String>()
                        == part.sha256
            })
            .context("frozen model part missing")?;
        parts.insert(part.path.clone(), store.read_blob(reference)?);
    }
    let object = ModelObject::decode(manifest_bytes, parts, 16 * 1024 * 1024)?;
    ensure!(
        record.data["model_digest"] == object.id(),
        "wrong model record"
    );
    let pipeline = object.math_pipeline()?;
    let table = jope_runtime::table::parse_csv(&csv, &Default::default())?;
    ensure!(
        table.rows.len() >= 64,
        "qualification needs 64 original TRAIN feature rows"
    );
    let indices = pipeline
        .input
        .columns
        .iter()
        .map(|c| {
            ensure!(
                c.name != pipeline.input.target.name,
                "target entered check features"
            );
            table
                .headers
                .iter()
                .position(|h| h == &c.name)
                .context("feature missing")
        })
        .collect::<Result<Vec<_>>>()?;
    let rows: Vec<Vec<Option<String>>> = table
        .rows
        .iter()
        .take(64)
        .map(|r| indices.iter().map(|&i| r[i].clone()).collect())
        .collect();
    let borrowed = rows
        .iter()
        .map(|r| r.iter().map(Option::as_deref).collect())
        .collect::<Vec<_>>();
    let expected = pipeline.predict_cells(&borrowed)?;
    fs::create_dir_all(&out)?;
    fs::create_dir_all(out.join("scratch"))?;
    fs::write(out.join("rows.json"), serde_json::to_vec(&rows)?)?;
    fs::write(out.join("engine.json"), serde_json::to_vec(&expected)?)?;
    fs::write(out.join("predictor.py"), object.export_source("python")?)?;
    fs::write(out.join("predictor.rs"), object.export_source("rust")?)?;
    fs::write(out.join("measure.py"), MEASURE)?;
    let output = Command::new("python3")
        .arg("-B")
        .arg(out.join("measure.py"))
        .arg(&out)
        .env("TMPDIR", out.join("scratch"))
        .output()?;
    fs::write(out.join("measure.stdout"), &output.stdout)?;
    fs::write(out.join("measure.stderr"), &output.stderr)?;
    ensure!(
        output.status.success(),
        "actual toolchain qualification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let measured: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("observations.json"))?)?;
    let receipt = MeasuredModelParity {
        schema: "jope.measured-model-parity/v1".into(),
        model_digest: object.id().into(),
        emitter_version: env!("CARGO_PKG_VERSION").into(),
        refit: serde_json::from_value(
            record
                .data
                .get("refit")
                .cloned()
                .unwrap_or(serde_json::json!("fit")),
        )?,
        source_sha256: serde_json::from_value(measured["source_sha256"].clone())?,
        toolchains: serde_json::from_value(measured["toolchains"].clone())?,
        rows,
        engine: expected,
        observed: serde_json::from_value(measured["observed"].clone())?,
    };
    let parity = record_model_parity(&store, artifact, &object, &receipt)?;
    fs::write(
        out.join("validated-parity.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    fs::write(
        out.join("registration.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_artifact":artifact,"model_digest":object.id(),"parity_artifact":parity,
            "run_id":run,"source":std::env::var("JOPE_INT01_SOURCE_HEAD").ok(),
            "actual_check_rows":64,"new_fits":0,"external_holdout_test_sealed_reads":0
        }))?,
    )?;
    Ok(())
}
const MEASURE: &str = r####"import sys,os,json,subprocess,hashlib,math,time,datetime,shutil
from pathlib import Path
root=Path(sys.argv[1]);rows=json.loads((root/'rows.json').read_text());engine=json.loads((root/'engine.json').read_text());commands=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(label,args):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
 r=subprocess.run([str(a) for a in args],capture_output=True,timeout=60)
 (root/(label+'.stdout')).write_bytes(r.stdout);(root/(label+'.stderr')).write_bytes(r.stderr)
 commands.append(dict(label=label,argv=[str(a) for a in args],start=start,seconds=time.monotonic()-t,exit=r.returncode))
 (root/'commands.json').write_text(json.dumps(commands,indent=2))
 assert r.returncode==0,(label,r.returncode,r.stderr.decode(errors='replace'))
 return r.stdout
run('python-self-check',[sys.executable,'-B',root/'predictor.py','--self-check'])
pycode='import importlib.util,json,sys;from pathlib import Path;s=importlib.util.spec_from_file_location("qualified_predictor",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(json.dumps(m.predict_rows(json.loads(Path(sys.argv[2]).read_text())),allow_nan=False))'
py=json.loads(run('python-predictions',[sys.executable,'-B','-c',pycode,root/'predictor.py',root/'rows.json']))
rustc=shutil.which('rustc');assert rustc,'rustc required for real qualification'
run('rust-self-check-compile',[rustc,'--edition=2021','--test',root/'predictor.rs','-o',root/'rust-self-check'])
selfcheck=run('rust-self-check',[root/'rust-self-check','--test-threads=1'])
assert b'1 passed' in selfcheck,'zero Rust self-check tests is not a pass'
rsrows=',\n'.join('vec!['+','.join('None' if v is None else 'Some('+json.dumps(v,ensure_ascii=False)+')' for v in row)+']' for row in rows)
wrapper='#[path="predictor.rs"] mod predictor;\nfn main(){let rows:Vec<Vec<Option<&str>>>=vec!['+rsrows+'];println!("{:?}",predictor::predict_cells(&rows).expect("prediction"));}\n'
(root/'rust-batch.rs').write_text(wrapper)
run('rust-batch-compile',[rustc,'--edition=2021',root/'rust-batch.rs','-o',root/'rust-batch'])
rs=json.loads(run('rust-predictions',[root/'rust-batch']))
assert len(rows)==len(engine)==len(py)==len(rs)==64
for observed in [py,rs]:
 assert all(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12+1e-10*abs(b) for a,b in zip(observed,engine))
rust_version=run('rust-version',[rustc,'--version','--verbose']).decode()
rust_root=run('rust-sysroot',[rustc,'--print','sysroot']).decode().strip()
toolchains={'python':sys.version+' executable_sha256='+sha(sys.executable),'rust':rust_version+' executable_sha256='+sha(Path(rust_root)/'bin/rustc')}
result=dict(source_sha256={'python':sha(root/'predictor.py'),'rust':sha(root/'predictor.rs')},toolchains=toolchains,observed={'python':py,'rust':rs},max_abs_error={'python':max(abs(a-b) for a,b in zip(py,engine)),'rust':max(abs(a-b) for a,b in zip(rs,engine))})
(root/'observations.json').write_text(json.dumps(result,allow_nan=False))
"####;
