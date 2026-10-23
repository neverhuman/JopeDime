// Acceptance: node examples/model-objects/check.mjs /absolute/path/to/jopedime
import assert from 'node:assert/strict';
import { mkdtempSync, copyFileSync, readFileSync, rmSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { JopeModel, saveState } from './client.mjs';
const executable = resolve(process.argv[2] ?? 'target/debug/jopedime');
const fixtures = fileURLToPath(new URL('.', import.meta.url));
const original = process.cwd();
const scratch = mkdtempSync(join(tmpdir(), 'jope-client-'));
function call(...args) {
  const result = spawnSync(executable, args, {encoding: 'utf8'});
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return JSON.parse(result.stdout);
}
try {
  copyFileSync(executable, join(scratch, 'jopedime'));
  copyFileSync(join(fixtures, 'train.csv'), join(scratch, 'train.csv'));
  copyFileSync(join(fixtures, 'rows.csv'), join(scratch, 'rows.csv'));
  process.chdir(scratch);
  const {model} = JopeModel.fit({train: 'train.csv', target: 'y', method: 'ridge', executable: './jopedime'});
  const initial = model.verify();
  model.predict('rows.csv', 'first.csv');
  let history = call('model', 'session', 'record', '--name', 'example', '--id', initial.id, '--expected-head', 'empty');
  const retained = saveState({format: 'org.example/checkpoint/v1', primary: 'state.bin',
    files: {'state.bin': {bytes: new Uint8Array([0, 255, 1]), mediaType: 'application/octet-stream'}}, executable});
  assert.equal(retained.inspection.executable, false);
  history = call('model', 'session', 'record', '--name', 'example', '--id', retained.inspection.id, '--expected-head', history.head);
  history = call('model', 'session', 'rollback', '--name', 'example', '--version', '1', '--expected-head', history.head);
  assert.equal(history.versions.length, 3);
  assert.equal(history.versions[2].model_id, initial.id);
  const selected = new JopeModel(history.model_path, './jopedime');
  mkdirSync('another-directory');
  process.chdir('another-directory');
  assert.equal(model.verify().id, initial.id); // relative executable was pinned before cwd changed
  selected.predict('../rows.csv', '../rollback.csv');
  assert.deepEqual(readFileSync('../first.csv'), readFileSync('../rollback.csv'));
  assert.throws(() => saveState({format: 'org.example/state/v1', primary: 'NUL', files: {NUL: {bytes: new Uint8Array([1]), mediaType: 'application/octet-stream'}}, executable}));
  console.log('PASS: disk inference, opaque retention, session rollback and changed working directory');
} finally {
  process.chdir(original);
  rmSync(scratch, {recursive: true, force: true});
}
