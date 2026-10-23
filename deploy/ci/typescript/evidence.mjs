// Bind complete model qualification to source, native event, Node and run attempt.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,readdirSync,lstatSync,realpathSync} from 'node:fs';
import {resolve,relative,join} from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
const [mode,directory,nodeVersion]=process.argv.slice(2);
assert.ok(['create','verify'].includes(mode));
const root=resolve(directory), read=name=>readFileSync(join(root,name),'utf8');
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const git=(...args)=>execFileSync('git',args,{encoding:'utf8'}).trim();
const inputs=['deploy/ci/typescript/package-lock.json','deploy/ci/typescript/browser-deps.lock.json','rust-toolchain.toml'];
function inventory(dir=root) {
  const files=[];
  for(const name of readdirSync(dir)) {
    const file=join(dir,name), stat=lstatSync(file);
    assert.ok(!stat.isSymbolicLink(),'evidence symlinks forbidden');
    if(stat.isDirectory()) files.push(...inventory(file));
    else if(stat.isFile() && relative(root,file)!=='receipt.json') files.push(relative(root,file));
    else assert.ok(stat.isFile(),'unexpected evidence type');
  }
  return files.sort();
}
function qualification(version) {
  const counts={};
  for(const name of ['recipes','fitted','bindings','pipeline','cli','census']) {
    const text=read(name+'.log'), matches=[...text.matchAll(/test result: ok\. (\d+) passed; (\d+) failed/g)];
    assert.ok(matches.length && matches.every(m=>Number(m[2])===0),name);
    counts[name]=matches.reduce((n,m)=>n+Number(m[1]),0); assert.ok(counts[name]>0,name);
    if(['recipes','fitted','census'].includes(name)) {
      const cjs=text.split('TypeScript cjs:').length-1;
      assert.ok(cjs>0 && cjs===text.split('TypeScript esm:').length-1,name);
    }
  }
  const census=JSON.parse(read('census.json'));
  assert.deepEqual(census.map(c=>c.kind).sort(),['bound','legacy-bound','recipe','v2']);
  for(const c of census) {
    assert.ok(c.total>0 && c.unaccounted===0);
    if(c.kind==='v2') assert.equal(c.executed,c.total);
    else {
      assert.equal(c.executed.length+c.refused.length,c.total);
      assert.equal(new Set(c.executed).size,c.executed.length);
      assert.equal(new Set(c.refused).size,c.refused.length);
      assert.ok(!c.executed.some(x=>new Set(c.refused).has(x)));
    }
  }
  const text=read('census.log'), metadata=text.split('TypeScript metadata: strict heading qualified').length-1;
  assert.equal(metadata,1);
  const cases=census.reduce((n,c)=>n+(c.kind==='v2'?c.executed:c.executed.length),5);
  assert.equal(text.split('TypeScript cjs:').length-1,cases+metadata);
  assert.match(read('contracts.log'),/pass [1-9]\d*/); assert.match(read('contracts.log'),/fail 0\b/);
  assert.match(read('contracts.log'),/model\.js\s+\|\s+100\.00\s+\|\s+100\.00\s+\|\s+100\.00/);
  assert.match(read('receipt-controls.log'),/pass [1-9]\d*/); assert.match(read('receipt-controls.log'),/fail 0\b/);
  let browsers=[];
  if(version==='26.1.0') {
    const report=JSON.parse(read('browsers.json'));
    assert.equal(report.source,process.env.JOPE_SOURCE_SHA);
    assert.equal(report.cases.length,cases);
    const caseIds=report.cases.map(c=>c.kind+'/'+c.name); assert.equal(new Set(caseIds).size,cases);
    browsers=[...new Set(report.results.map(r=>r.browser))].sort();
    assert.deepEqual(browsers,['chromium','firefox','webkit']);
    assert.equal(report.results.length,cases*3);
    for(const name of browsers) {
      const rows=report.results.filter(r=>r.browser===name);
      assert.deepEqual(rows.map(r=>r.kind+'/'+r.name).sort(),[...caseIds].sort());
      assert.ok(rows.every(r=>r.worker===true && Number.isInteger(r.predictions) && r.predictions>0 && typeof r.version==='string' && r.version.length>0));
    }
    const native=JSON.parse(read('browser-dependencies.json'));
    assert.equal(native.lock_sha256,hash(readFileSync(inputs[1])));
  }
  return {counts,cases,browsers,facade:{lines:100,branches:100,functions:100}};
}
const version=mode==='create'?process.env.JOPE_TS_EXPECTED_NODE:nodeVersion;
assert.ok(['24.19.0','26.1.0'].includes(version),'exact supported Node version required');
const source=process.env.JOPE_SOURCE_SHA, run=process.env.GITHUB_RUN_ID, attempt=Number(process.env.GITHUB_RUN_ATTEMPT);
assert.match(source,/^[a-f0-9]{40}$/); assert.ok(run && attempt>0);
assert.match(process.env.GITHUB_SHA,/^[a-f0-9]{40}$/);
assert.ok(['pull_request','workflow_dispatch'].includes(process.env.GITHUB_EVENT_NAME));
if(mode==='create') {
  assert.equal(process.version,'v'+version);
  const checkout=git('rev-parse','HEAD'); assert.equal(checkout,source);
  assert.equal(git('status','--porcelain=v1','--untracked-files=normal'),'','clean source required');
  const identity=Object.fromEntries(read('identity.txt').trim().split('\n').map(s=>s.split('=')));
  assert.equal(identity.source,source); assert.equal(identity.run,run); assert.equal(Number(identity.attempt),attempt);
  const tools=read('toolchains.txt').split('\n'); assert.match(tools[0],/^rustc 1\.97\.1 /); assert.equal(tools[1],'v'+version); assert.match(tools[2],/7\.0\.2/);
  const proof=qualification(version), files=Object.fromEntries(inventory().map(f=>[f,hash(readFileSync(join(root,f)))]));
  const receipt={schema:'jope.typescript-qualification/v1',source_sha:source,checkout_sha:checkout,event_sha:process.env.GITHUB_SHA,source_tree:git('rev-parse','HEAD^{tree}'),run_id:run,attempt,event:process.env.GITHUB_EVENT_NAME,node:version,dirty_source:[],proof,inputs:Object.fromEntries(inputs.map(p=>[p,hash(readFileSync(p))])),files};
  writeFileSync(join(root,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');
} else {
  const r=JSON.parse(read('receipt.json'));
  assert.equal(r.schema,'jope.typescript-qualification/v1');
  assert.equal(r.source_sha,source); assert.equal(r.checkout_sha,source); assert.equal(r.event_sha,process.env.GITHUB_SHA);
  assert.equal(r.run_id,run); assert.equal(r.attempt,attempt); assert.equal(r.event,process.env.GITHUB_EVENT_NAME); assert.equal(r.node,version);
  assert.deepEqual(r.dirty_source,[]);
  assert.equal(r.source_tree,git('rev-parse','HEAD^{tree}'));
  assert.deepEqual(Object.keys(r.inputs).sort(),[...inputs].sort());
  for(const name of inputs) assert.equal(r.inputs[name],hash(readFileSync(name)));
  assert.deepEqual(Object.keys(r.files).sort(),inventory());
  for(const [name,digest] of Object.entries(r.files)) {
    const file=resolve(root,name); assert.ok(file.startsWith(root+'/') && realpathSync(file)===file);
    assert.equal(hash(readFileSync(file)),digest);
  }
  assert.deepEqual(r.proof,qualification(version));
}
console.log(`Complete source/run/attempt-bound TypeScript qualification verified: Node ${version}`);
