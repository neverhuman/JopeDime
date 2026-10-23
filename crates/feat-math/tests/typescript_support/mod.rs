// Opt-in executable TypeScript qualification shared with existing golden tests.
use std::fs;
use std::path::Path;
use std::process::Command;

pub fn enabled() -> bool {
    std::env::var("JOPE_TYPESCRIPT_PARITY").as_deref() == Ok("1")
}

fn command(root: &Path, program: &str, args: &[&str]) {
    let output = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .unwrap_or_else(|e| panic!("required TypeScript tool {program}: {e}"));
    assert!(
        output.status.success(),
        "{program} {args:?}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

pub fn run(
    root: &Path,
    source: &str,
    kernel: &str,
    input: &serde_json::Value,
    expected: &[f64],
    fitted: bool,
) {
    if !enabled() {
        return;
    }
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("model.kernel.rs"), kernel).unwrap();
    command(
        root,
        "rustc",
        &[
            "--edition=2021",
            "--crate-name",
            "model_kernel",
            "--target",
            "wasm32-unknown-unknown",
            "--crate-type",
            "cdylib",
            "-C",
            "opt-level=3",
            "-C",
            "panic=abort",
            "model.kernel.rs",
            "-o",
            "model.wasm",
        ],
    );
    fs::write(root.join("input.json"), serde_json::to_vec(input).unwrap()).unwrap();
    fs::write(
        root.join("expected.json"),
        serde_json::to_vec(expected).unwrap(),
    )
    .unwrap();
    for (folder, module, extension, loader) in [
        (
            "cjs",
            "commonjs",
            "cjs",
            "const {createModel}=require('./model.js');",
        ),
        (
            "esm",
            "ES2020",
            "mjs",
            "import {createModel} from './model.js';",
        ),
    ] {
        let dir = root.join(folder);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.ts"), source).unwrap();
        fs::write(dir.join("consumer.ts"), CONSUMER).unwrap();
        let package = if folder == "esm" {
            r#"{"type":"module"}"#
        } else {
            r#"{"type":"commonjs"}"#
        };
        fs::write(dir.join("package.json"), package).unwrap();
        let compiler = std::env::var("JOPE_TSC").unwrap_or_else(|_| "tsc".into());
        command(
            &dir,
            &compiler,
            &[
                "--strict",
                "--target",
                "ES2020",
                "--module",
                module,
                "--lib",
                "ES2020,DOM",
                "--declaration",
                "model.ts",
                "consumer.ts",
            ],
        );
        let driver = format!("{loader}\nconst fitted={fitted};\n{DRIVER}");
        let driver_name = format!("driver.{extension}");
        fs::write(dir.join(&driver_name), driver).unwrap();
        command(&dir, "node", &[&driver_name]);
        eprintln!(
            "TypeScript {folder}: {} predictions qualified",
            expected.len()
        );
    }
}

const CONSUMER: &str = r#"import { Model, createModel } from "./model.js";
function typeContract(model: Model): Float64Array {
  model.fit([[1, 2]], [3]);
  model.fit([new Float64Array([1, 2])], new Float64Array([3]));
  // @ts-expect-error numeric prediction refuses string features
  model.predict([["wrong"]]);
  // @ts-expect-error numeric fitting refuses string targets
  model.fit([[1]], ["wrong"]);
  // @ts-expect-error fitted string prediction refuses numeric fields
  model.predictStrings([[1]]);
  const pending: Promise<Model> = createModel(new Uint8Array());
  void pending;
  return model.predict([[1, 2]]);
}
void typeContract;
"#;
const DRIVER: &str = r#"
(async()=>{
  const fs=await import('node:fs');
  const assert=await import('node:assert/strict');
  const bytes=fs.readFileSync('../model.wasm');
  const module=await WebAssembly.compile(bytes);
  assert.equal(WebAssembly.Module.imports(module).length,0);
  const input=JSON.parse(fs.readFileSync('../input.json','utf8'));
  const expected=JSON.parse(fs.readFileSync('../expected.json','utf8'));
  const raw=(await WebAssembly.instantiate(module,{})).exports;
  assert.equal(raw.model_abi_version(),1);
  for(const allocator of ['model_values','model_labels','model_text']) {
    assert.equal(raw[allocator](16777217),0);
    assert.equal(raw[allocator](0xffffffff),0);
  }
  assert.notEqual(raw.model_fit(0,1),0);
  assert.notEqual(raw.model_fit(1,0),0);
  assert.notEqual(raw.model_fit(0xffffffff,0xffffffff),0);
  assert.notEqual(raw.model_predict(1,1),0);
  assert.equal(raw.model_output_len(),0);
  if(fitted) {
    const p=raw.model_text(1);new Uint8Array(raw.memory.buffer,p,1)[0]=255;
    assert.notEqual(raw.model_predict_text(),0); // Invalid UTF-8, not lossy replacement.
  } else {
    assert.notEqual(raw.model_predict_text(),0);
    const p=raw.model_values(1);new Float64Array(raw.memory.buffer,p,1)[0]=1;
    assert.notEqual(raw.model_fit(2,1),0); // Declared dimensions differ from storage.
    const q=raw.model_labels(1);new Float64Array(raw.memory.buffer,q,1)[0]=NaN;
    assert.notEqual(raw.model_fit(1,1),0);
  }
  raw.model_dispose();raw.model_dispose();assert.equal(raw.model_output_len(),0);
  const model=await createModel(module);
  const other=await createModel(bytes);
  // Respect the byteOffset/byteLength of caller views instead of compiling
  // unrelated prefix/suffix bytes from the backing ArrayBuffer.
  const padded=new Uint8Array(bytes.length+16);padded.set(bytes,8);
  const sliced=await createModel(padded.subarray(8,8+bytes.length));sliced.dispose();
  const arrayBuffer=await createModel(new Uint8Array(bytes).buffer);arrayBuffer.dispose();
  let got;
  if(fitted) {
    got=model.predictStrings(input.rows);
    assert.throws(()=>model.fit([[1]],[1]));
    assert.throws(()=>model.predictCsv('wrong\n1\n'));
    assert.throws(()=>model.predictStrings([[1]]));
    assert.throws(()=>model.predictStrings([new Array(input.rows[0].length)]));
    assert.equal(model.predictStrings([]).length,0);
    assert.throws(()=>model.predictCsv('"unterminated'));
    assert.deepEqual(model.predictStrings(input.rows),got);
    const otherResult=other.predictStrings(input.rows);
    assert.deepEqual(otherResult,got);
  } else {
    assert.throws(()=>model.predict(input.test));
    assert.throws(()=>model.fit([],[]));
    assert.throws(()=>model.fit([[1],[1,2]],[1,2]));
    assert.throws(()=>model.fit([[1]],[NaN]));
    assert.throws(()=>model.fit([[1]],new Array(1)));
    assert.throws(()=>model.fit([[1]],['1']));
    assert.throws(()=>model.fit([new Array(input.train[0].length)],[1]));
    assert.equal(model.fit(input.train,input.targets),model);
    got=model.predict(input.test);
    assert.equal(model.predict([]).length,0);
    assert.equal(model.score([],[]),0);
    assert.equal(model.score(input.test,new Array(input.test.length).fill(2)),0);
    assert.throws(()=>model.score(input.test,new Array(input.test.length)));
    assert.throws(()=>model.score(input.test,input.test.map(()=>Infinity)));
    const mean=expected.reduce((a,b)=>a+b,0)/expected.length;
    const total=expected.reduce((a,b)=>a+(b-mean)**2,0);
    const residual=expected.reduce((a,b,i)=>a+(b-got[i])**2,0);
    assert.ok(Math.abs(model.score(input.test,expected)-(total<=0?0:1-residual/total))<1e-12);
    assert.throws(()=>model.predict([new Array(input.train[0].length+1).fill(0)]));
    const retained=Array.from(got);
    const shifted=input.targets.map(v=>v+0.25);
    other.fit(input.train,shifted);
    other.predict(input.test);
    assert.deepEqual(model.predict(input.test),got);
    assert.deepEqual(Array.from(got),retained);
    if(input.stress) {
      const many=Array.from({length:65536},()=>input.test[0]);
      const grown=model.predict(many);
      assert.equal(grown.length,many.length);
      assert.deepEqual(Array.from(got),retained);
    }
    model.fit(input.train,input.targets);
    assert.deepEqual(model.predict(input.test),got);
  }
  assert.equal(got.length,expected.length);
  for(let i=0;i<got.length;i++){
    const tolerance=1e-8*(1+Math.abs(expected[i]));
    assert.ok(Number.isFinite(got[i])&&Math.abs(got[i]-expected[i])<=tolerance,
      'row '+i+': '+got[i]+' versus '+expected[i]+' tolerance '+tolerance);
  }
  const saved=Array.from(got);
  model.dispose(); model.dispose();
  assert.throws(()=>fitted?model.predictStrings(input.rows):model.predict(input.test));
  assert.deepEqual(Array.from(got),saved);
  other.dispose();
  console.log(JSON.stringify({predictions:got.length,fitted,modules:'strict',tolerance:'1e-8*(1+abs(reference))'}));
})().catch(e=>{console.error(e);process.exitCode=1});
"#;
