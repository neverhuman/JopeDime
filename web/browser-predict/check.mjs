import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const [wasmPath, fixtureRoot] = process.argv.slice(2);
const module = await WebAssembly.compile(fs.readFileSync(wasmPath));
assert.deepEqual(WebAssembly.Module.imports(module), []);
assert(!WebAssembly.Module.exports(module).some(e => /fit|train/i.test(e.name)));
const k = (await WebAssembly.instantiate(module, {})).exports;
assert.equal(k.browser_abi_version(), 1);
function call(name, value) {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const ptr = k.browser_input(bytes.length);
  assert(ptr); new Uint8Array(k.memory.buffer, ptr, bytes.length).set(bytes);
  k[name]();
  return JSON.parse(new TextDecoder().decode(new Uint8Array(k.memory.buffer, k.browser_output(), k.browser_output_len())));
}
let rows = 0;
for (const name of ['ridge', 'gpr']) {
  const fixture = JSON.parse(fs.readFileSync(path.join(fixtureRoot, name + '.json')));
  const loaded = call('browser_load', fixture.envelope);
  assert(loaded.ok, JSON.stringify(loaded));
  for (const probe of fixture.probes) {
    const result = call('browser_predict', { values: probe.values });
    assert(result.ok, JSON.stringify(result));
    assert(Math.abs(result.result.value - probe.expected) <= 1e-9 * Math.max(1, Math.abs(probe.expected)), `${name}: native/WASM prediction differs`);
    rows++;
  }
  const changed = structuredClone(fixture.envelope);
  changed.parts['pipeline.json'] = Buffer.from('{}').toString('base64');
  assert.equal(call('browser_load', changed).ok, false);
  assert.equal(call('browser_predict', { values: ['0', '4'] }).ok, false);
}
assert.equal(k.browser_input(49 * 1024 * 1024), 0);
k.browser_dispose();
console.log(JSON.stringify({ abi: 1, noHostImports: true, inferenceOnly: true, nativeParityRows: rows, tamperingRejected: true, staleModelCleared: true }));
