//! M3 CPU measurements through the product's public numerical APIs.
use anyhow::{ensure, Context, Result};
use jope_service::{
    dataset::jmat,
    ranker::{featurizer::Featurizer, landmarks, proj16},
    worker,
};
use safetensors::{Dtype, SafeTensors};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn sha(raw: &[u8]) -> String {
    Sha256::digest(raw)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn read(path: &Path, pin: &str) -> Result<Vec<u8>> {
    ensure!(
        pin.len() == 64
            && pin
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "invalid pin"
    );
    ensure!(
        fs::symlink_metadata(path)?.is_file() && fs::metadata(path)?.len() <= 64 * 1024 * 1024,
        "not a bounded regular file"
    );
    let raw = fs::read(path)?;
    ensure!(sha(&raw) == pin, "file hash differs: {}", path.display());
    Ok(raw)
}
fn text<'a>(row: &'a Value, key: &str) -> Result<&'a str> {
    row[key].as_str().with_context(|| format!("missing {key}"))
}
fn floats(raw: &[u8], name: &str) -> Result<Vec<f32>> {
    let file = SafeTensors::deserialize(raw)?;
    let t = file.tensor(name)?;
    ensure!(t.dtype() == Dtype::F32, "F32 required");
    Ok(t.data()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
struct Custody {
    root: PathBuf,
    index: Value,
}
impl Custody {
    fn open(job: &Value) -> Result<Self> {
        let root = PathBuf::from(text(job, "store")?);
        ensure!(
            root.is_absolute() && root.canonicalize()? == root,
            "canonical store required"
        );
        let pin = text(job, "catalog_sha256")?;
        let index = serde_json::from_slice(&read(&root.join(pin), pin)?)?;
        Ok(Self { root, index })
    }
    fn load(&self, name: &str) -> Result<Vec<u8>> {
        let spec = &self.index["files"][name];
        let pin = text(spec, "sha256")?;
        let raw = read(&self.root.join(pin), pin)?;
        ensure!(
            raw.len() as u64 == spec["bytes"].as_u64().context("byte count")?,
            "file length differs"
        );
        Ok(raw)
    }
}
fn features(job: &Value) -> Result<Value> {
    let store = Custody::open(job)?;
    let rows = store.load("dsl/programs.jsonl")?;
    let f = Featurizer::default();
    let mut result = vec![];
    for line in std::str::from_utf8(&rows)?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let dsl = text(&row, "dsl")?;
        let key = text(&row, "program_sha256")?;
        ensure!(sha(dsl.as_bytes()) == key, "DSL pin differs");
        let feature = f.featurize(dsl)?;
        let projected = proj16::proj16(&feature.map(f64::from));
        result.push(json!({"program_sha256":key,"feat412":feature.map(f64::from).as_slice(),"proj16":projected}));
    }
    Ok(json!({"names":f.feature_names().as_slice(),"programs":result}))
}
fn landmark(job: &Value) -> Result<Value> {
    let store = Custody::open(job)?;
    let name = text(job, "fixture")?;
    let raw = store.load(name)?;
    let file = SafeTensors::deserialize(&raw)?;
    let input = file.tensor("input.X")?;
    ensure!(input.shape().len() == 2, "matrix required");
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let x = floats(&raw, "input.X")?;
    let y = floats(&raw, "input.y")?;
    let actual = landmarks::fit_transform_and_extract_bundle_high(&x, rows, cols, &y)?;
    Ok(
        json!({"context_sha256":text(job,"context_sha256")?,"rows":rows,"columns":cols,
        "v256":actual.vector().as_slice(),"col_full":actual.raw_columns().iter().flatten().copied().collect::<Vec<_>>(),
        "col":actual.columns().iter().flatten().copied().collect::<Vec<_>>()}),
    )
}
fn u32_at(raw: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap()) as usize
}
fn evaluator(job: &Value) -> Result<Value> {
    let raw = read(Path::new(text(job, "table")?), text(job, "table_sha256")?)?;
    ensure!(raw.len() >= 17 && &raw[..4] == b"JTT1", "JTT1 required");
    let rows = u32_at(&raw, 4);
    let query = u32_at(&raw, 8);
    let cols = u32_at(&raw, 12);
    ensure!(
        query == 0
            && raw[16] == 2
            && (50..=2048).contains(&rows)
            && (1..=512).contains(&cols)
            && raw.len() == 17 + rows + 8 * rows * (cols + 1),
        "full TRAIN two-fold table required"
    );
    let fold_ids = &raw[17..17 + rows];
    ensure!(fold_ids.iter().all(|f| *f < 2), "fold identity");
    let mut names = (0..cols).map(|i| format!("x{i}")).collect::<Vec<_>>();
    names.push("target".into());
    let mut records = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut cells = Vec::with_capacity(cols + 1);
        for col in 0..=cols {
            let at = 17 + rows + 8 * (row * (cols + 1) + col);
            let value = f64::from_le_bytes(raw[at..at + 8].try_into().unwrap());
            ensure!(col < cols || value.is_finite(), "invalid target");
            cells.push(value.is_finite().then(|| value.to_string()));
        }
        records.push(cells);
    }
    let table = jope_runtime::table::CsvTable {
        headers: names.clone(),
        rows: records,
        sha256: sha(&raw),
        bytes: raw.len() as u64,
    };
    let bytes = jmat::encode(&table)?;
    let matrix = jmat::Matrix::open(bytes)?;
    let folds = (0..2)
        .map(|f| {
            json!({"repeat":0,
        "train":(0..rows).filter(|&i|fold_ids[i]!=f).collect::<Vec<_>>(),
        "validation":(0..rows).filter(|&i|fold_ids[i]==f).collect::<Vec<_>>()})
        })
        .collect::<Vec<_>>();
    let request = worker::Request {
        attempt: worker::Attempt {
            run: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse()?,
            round: 0,
            slot: 0,
            attempt: 1,
        },
        matrix_path: PathBuf::from("unused-direct-callback"),
        fold_spec: json!({"version":2,"matrix_sha256":sha(matrix.bytes()),"target":"target","features":names[..cols],"folds":folds}),
        dsl: text(job, "dsl")?.into(),
        objective: json!("rmse"),
        seed: 73,
        limits: worker::Limits {
            wall_ms: 120000,
            rss_bytes: 1024 * 1024 * 1024,
        },
    };
    let actual = worker::evaluate::native(&request, &matrix);
    Ok(match actual {
        Ok(score) => {
            json!({"context_sha256":job["context_sha256"],"program_sha256":job["program_sha256"],"fitness":score.values[0],"fold_rmse":score.fold_scores,"predictions_sha256":score.predictions_digest,"failure":null})
        }
        Err(failure) => {
            json!({"context_sha256":job["context_sha256"],"program_sha256":job["program_sha256"],"fitness":null,"failure":format!("{failure:?}")})
        }
    })
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().context("mode required")?;
    let job: Value = serde_json::from_slice(&fs::read(args.next().context("job file required")?)?)?;
    let result = match mode.as_str() {
        "features" => features(&job),
        "landmarks" => landmark(&job),
        "evaluator" => evaluator(&job),
        _ => anyhow::bail!("unknown M3 mode"),
    }?;
    println!(
        "{}",
        json!({"schema":"jope.rel07.m3-native-measurement/v1","source_sha":option_env!("JOPE_BUILD_SOURCE_SHA").unwrap_or(""),"mode":mode,"measurement":result})
    );
    Ok(())
}
