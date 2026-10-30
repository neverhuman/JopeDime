import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,readFileSync,writeFileSync,rmSync,symlinkSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {spawnSync,execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';

test('matrix evidence rejects stale identities, missing engines, omitted cases, altered inputs and fabricated success',()=>{
  const dir=mkdtempSync(join(tmpdir(),'typescript-evidence-'));
  const verifier=resolve('deploy/ci/typescript/evidence.mjs');
  const inputNames=['deploy/ci/typescript/package-lock.json','deploy/ci/typescript/browser-deps.lock.json','rust-toolchain.toml'];
  const hash=b=>createHash('sha256').update(b).digest('hex');
  const write=(name,value)=>{mkdirSync(resolve(dir,name,'..'),{recursive:true});writeFileSync(join(dir,name),value);};
  const json=(name,value)=>write(name,JSON.stringify(value)+'\n');
  const git=(...args)=>execFileSync('git',args,{cwd:dir,encoding:'utf8'}).trim();
  try {
    for(const name of inputNames) write(name,readFileSync(name));
    write('.gitignore','target/\n');
    git('init','-q');git('add','.');git('-c','user.name=Qualification fixture','-c','user.email=fixture@example.invalid','commit','-qm','qualification fixture');
    const sha=git('rev-parse','HEAD'), root='target/typescript-evidence';
    const version=process.versions.node;
    const env={...process.env,JOPE_SOURCE_SHA:sha,GITHUB_SHA:sha,GITHUB_EVENT_NAME:'workflow_dispatch',GITHUB_RUN_ID:'123',GITHUB_RUN_ATTEMPT:'1',JOPE_TS_EXPECTED_NODE:version};
    const ok='test result: ok. 1 passed; 0 failed\n', execution='TypeScript cjs: 1 predictions qualified\nTypeScript esm: 1 predictions qualified\n';
    const census=['recipe','bound','legacy-bound'].map(kind=>({kind,total:1,executed:['fixture'],refused:[],unaccounted:0}));
    census.push({kind:'v2',total:1,executed:1,unaccounted:0});
    for(const name of ['recipes','fitted','bindings','pipeline','cli']) write(`${root}/${name}.log`,ok+execution);
    write(`${root}/census.log`,ok+execution.repeat(10)+census.map(c=>'TypeScript census: '+JSON.stringify(c)+'\n').join('')+'TypeScript metadata: strict heading qualified\n');
    json(`${root}/census.json`,census);
    write(`${root}/contracts.log`,'pass 7\nfail 0\nmodel.js | 100.00 | 100.00 | 100.00\n');
    write(`${root}/receipt-controls.log`,'pass 1\nfail 0\n');
    write(`${root}/identity.txt`,`source=${sha}\nrun=123\nattempt=1\n`);
    write(`${root}/toolchains.txt`,`rustc 1.97.1 (fixture)\nv${version}\nVersion 7.0.2\n`);
    const declaration='export interface PredictorRow { readonly "age": number | null; readonly "region": string | null; readonly "quote\\"field": number | null; }\nexport declare function predict(row: PredictorRow): number;\nexport declare function predictBatch(rows: readonly PredictorRow[]): number[];\n';
    const module='export function predict(row) { return 1; }\nexport function predictBatch(rows) { return rows.map(() => 1); }\n';
    const fitted=`${root}/fitted-bundle`;
    write(`${root}/fitted-bundle-consumer.log`,'running 1 test\ntest export::bundle::tests::typescript_bundle_consumer_compiles_and_calls_named_predictors ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n');
    write(`${fitted}/predictor.mjs`,module);
    write(`${fitted}/predictor.d.ts`,declaration);
    write(`${fitted}/predictor.d.mts`,declaration);
    write(`${fitted}/fitted-consumer.mts`,'import { predict, predictBatch, type PredictorRow } from "./predictor.mjs";\nconst row: PredictorRow = { age: null, region: "north", \'quote"field\': 2 };\nconst one: number = predict(row); const many: number[] = predictBatch([row]);\nvoid one; void many;\n');
    write(`${fitted}/tsc.log`,'command: tsc --strict --noEmit --target ES2020 --module NodeNext --moduleResolution NodeNext fitted-consumer.mts\nstatus: exit status: 0\nstdout:\nstderr:\n');
    write(`${fitted}/node-runtime.log`,'node exercised exported predict and predictBatch with the same fitted row; result finite and equal\n');
    json(`${fitted}/manifest.json`,{entries:{'src/typescript/predictor.mjs':hash(module),'src/typescript/predictor.d.ts':hash(declaration),'src/typescript/predictor.d.mts':hash(declaration)}});
    const cases=Array.from({length:9},(_,i)=>({kind:'recipe',name:'fixture-'+i}));
    const browsers={source:sha,cases,results:['chromium','firefox','webkit'].flatMap(browser=>cases.map(c=>({browser,version:'fixture',...c,predictions:1,worker:true})))};
    json(`${root}/browsers.json`,browsers);
    json(`${root}/browser-dependencies.json`,{lock_sha256:hash(readFileSync(inputNames[1]))});
    const run=(mode,extra={})=>spawnSync(process.execPath,[verifier,mode,root,...(mode==='verify'?[version]:[])],{cwd:dir,env:{...env,...extra},encoding:'utf8'});
    let result=run('create');assert.equal(result.status,0,result.stderr);
    result=run('verify');assert.equal(result.status,0,result.stderr);
    result=run('verify',{GITHUB_RUN_ATTEMPT:'2'});assert.equal(result.status,0,result.stderr);
    const original=JSON.parse(readFileSync(join(dir,root,'receipt.json'),'utf8'));
    for(const mutation of [r=>r.source_sha='a'.repeat(40),r=>r.event_sha='b'.repeat(40),r=>r.attempt=0,r=>r.attempt=2,r=>r.node='0.0.0',r=>r.dirty_source=['edited'],r=>delete r.files['census.log']]) {
      const r=structuredClone(original);mutation(r);json(`${root}/receipt.json`,r);assert.notEqual(run('verify').status,0);
    }
    const rejectContent=(name,value)=>{
      const file=join(dir,root,name), saved=readFileSync(file);write(`${root}/${name}`,value);
      const r=structuredClone(original);r.files[name]=hash(readFileSync(file));json(`${root}/receipt.json`,r);
      assert.notEqual(run('verify').status,0,`must reject ${name} even with a matching file hash`);write(`${root}/${name}`,saved);
    };
    rejectContent('cli.log','test result: ok. 0 passed; 0 failed\n');
    rejectContent('contracts.log','pass 7\nfail 0\nmodel.js | 99.99 | 100.00 | 100.00\n');
    rejectContent('receipt-controls.log','pass 0\nfail 0\n');
    rejectContent('fitted-bundle/tsc.log','command: tsc --strict\nstatus: exit status: 2\n');
    rejectContent('fitted-bundle/predictor.d.ts',declaration+'// altered after archive creation\n');
    if(version==='26.1.0') {
      rejectContent('browsers.json',JSON.stringify({...browsers,results:browsers.results.filter(r=>r.browser!=='webkit')}));
      rejectContent('browsers.json',JSON.stringify({...browsers,results:browsers.results.slice(1)}));
      rejectContent('browsers.json',JSON.stringify({...browsers,results:browsers.results.map((r,i)=>i===0?{...r,worker:false}:r)}));
    }
    json(`${root}/receipt.json`,original);
    const source=readFileSync(join(dir,inputNames[0]));write(inputNames[0],Buffer.concat([source,Buffer.from(' ')]));assert.notEqual(run('verify').status,0);write(inputNames[0],source);
    const file=join(dir,root,'cli.log'), saved=readFileSync(file);rmSync(file);symlinkSync(join(dir,root,'pipeline.log'),file);assert.notEqual(run('verify').status,0);rmSync(file);write(`${root}/cli.log`,saved);
    result=run('verify');assert.equal(result.status,0,result.stderr);
  } finally {rmSync(dir,{recursive:true,force:true});}
});
