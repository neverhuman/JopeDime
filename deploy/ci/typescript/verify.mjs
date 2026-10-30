// Fail closed on absent tests, incomplete censuses, or absent coverage files.
// Predicate-for-predicate port of the original PR412 Python verifier (ADR-0006).
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const root='target/typescript-evidence';
const read=name=>readFileSync(`${root}/${name}.log`,'utf8');
for(const name of ['recipes','fitted','bindings','pipeline','cli','census']) {
  const text=read(name);
  const counts=[...text.matchAll(/test result: ok\. (\d+) passed; (\d+) failed/g)];
  assert.ok(counts.length && counts.reduce((n,m)=>n+Number(m[1]),0)>0, name+': zero tests');
  assert.ok(counts.every(m=>Number(m[2])===0),name+': failed tests');
  if(['recipes','fitted','census'].includes(name)) {
    const cjs=text.split('TypeScript cjs:').length-1, esm=text.split('TypeScript esm:').length-1;
    assert.ok(cjs===esm && cjs>0,name);
  }
}
const text=read('census');
const censuses=[...text.matchAll(/TypeScript census: (\{[^\n]+\})/g)].map(m=>JSON.parse(m[1]));
assert.deepEqual(censuses.map(c=>c.kind).sort(),['bound','legacy-bound','recipe','v2']);
for(const c of censuses) {
  assert.ok(c.total>0 && c.unaccounted===0);
  if(c.kind==='v2') assert.equal(c.executed,c.total);
  else {
    assert.equal(c.executed.length+c.refused.length,c.total);
    assert.ok(!c.executed.some(x=>new Set(c.refused).has(x)));
  }
}
const metadata=text.split('TypeScript metadata: strict heading qualified').length-1;
assert.equal(metadata,1,'metadata boundary qualification missing');
const expected=censuses.reduce((n,c)=>n+(c.kind==='v2'?c.executed:c.executed.length),5+metadata);
assert.equal(text.split('TypeScript cjs:').length-1,expected);
const contracts=read('contracts');
assert.match(contracts,/pass [1-9]\d*/); assert.match(contracts,/fail 0\b/);
assert.match(contracts,/model\.js\s+\|\s+100\.00\s+\|\s+100\.00\s+\|\s+100\.00/,'facade coverage absent or below 100%');
if (process.argv.includes('--require-fitted-bundle')) {
const fittedRoot=`${root}/fitted-bundle`, fittedRead=name=>readFileSync(`${fittedRoot}/${name}`,'utf8');
assert.match(read('fitted-bundle-consumer'),/test result: ok\. 1 passed; 0 failed/,'fitted bundle consumer test absent or failed');
const declaration=fittedRead('predictor.d.ts');
assert.equal(fittedRead('predictor.d.mts'),declaration,'NodeNext declaration sidecar differs');
assert.ok(declaration.includes('readonly "age": number | null;'));
assert.ok(declaration.includes('readonly "region": string | null;'));
assert.ok(declaration.includes(`readonly ${JSON.stringify('quote"field')}: number | null;`));
const module=fittedRead('predictor.mjs');
assert.match(module,/export function predict\(/); assert.match(module,/export function predictBatch\(/);
assert.match(fittedRead('fitted-consumer.mts'),/from "\.\/predictor\.mjs"/);
assert.match(fittedRead('fitted-consumer.mts'),/predictBatch\(\[example\]\)/);
assert.match(fittedRead('tsc.log'),/command: tsc --strict[\s\S]*status: exit status: 0/);
assert.match(fittedRead('node-runtime.log'),/predict and predictBatch/);
const manifest=JSON.parse(fittedRead('manifest.json')), sha=bytes=>createHash('sha256').update(bytes).digest('hex');
assert.equal(manifest.entries['src/typescript/predictor.d.ts'],sha(declaration));
assert.equal(manifest.entries['src/typescript/predictor.d.mts'],manifest.entries['src/typescript/predictor.d.ts']);
assert.equal(manifest.entries['src/typescript/predictor.mjs'],sha(module));
}
writeFileSync(`${root}/census.json`,JSON.stringify(censuses,null,2)+'\n');
console.log('Complete TypeScript and fitted-bundle consumer qualification verified');
