import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const target = process.env.CARGO_TARGET_DIR || path.join(root, 'target');
fs.mkdirSync(target, { recursive: true });
const scratch = fs.mkdtempSync(path.join(target, 'browser-security-controls-'));
try {
  const bin = path.join(scratch, 'bin');
  fs.mkdirSync(bin);
  const trace = path.join(scratch, 'calls.log');
  const cargoScript = [
    '#!/usr/bin/env bash',
    'set -euo pipefail',
    'printf "%s\\n" "$*" >> "$BROWSER_SCAN_TRACE"',
    'case "$*" in',
    '  "deny --version") echo "cargo-deny $BROWSER_SCAN_VERSION" ;;',
    '  "deny --manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features list --layout crate")',
    '    if [ "$BROWSER_SCAN_GRAPH_EXIT" != 0 ]; then exit "$BROWSER_SCAN_GRAPH_EXIT"; fi',
    '    for package in jope-browser-predict feat-math jope-runtime; do',
    '      if [ "$package" != "$BROWSER_SCAN_OMIT" ]; then printf "%s@1: Apache-2.0\\n" "$package"; fi',
    '    done ;;',
    '  "deny --manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features check advisories bans licenses sources")',
    '    exit "$BROWSER_SCAN_CHECK_EXIT" ;;',
    '  *) exit 94 ;;',
    'esac',
  ].join('\n');
  fs.writeFileSync(path.join(bin, 'cargo'), cargoScript, { mode: 0o755 });
  function run(extra = {}) {
    return spawnSync('bash', ['web/browser-predict/security.sh'], {
      cwd: root, encoding: 'utf8',
      env: { ...process.env, PATH: bin + path.delimiter + process.env.PATH,
        CARGO_TARGET_DIR: path.join(scratch, 'target'), BROWSER_SCAN_TRACE: trace,
        BROWSER_SCAN_VERSION: '0.20.2', BROWSER_SCAN_GRAPH_EXIT: '0',
        BROWSER_SCAN_CHECK_EXIT: '0', BROWSER_SCAN_OMIT: '', ...extra },
    });
  }
  const positive = run();
  assert.equal(positive.status, 0, positive.stdout + positive.stderr);
  assert(fs.readFileSync(trace, 'utf8').includes('--manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features check advisories bans licenses sources'));
  const receipt = JSON.parse(fs.readFileSync(path.join(scratch, 'target/browser-predict/security/receipt.json')));
  assert.deepEqual(receipt.checks, ['advisories', 'bans', 'licenses', 'sources']);
  assert.equal(receipt.inputs['web/browser-predict/Cargo.lock'].length, 64);
  assert.equal(run({ BROWSER_SCAN_GRAPH_EXIT: '17' }).status, 17);
  assert.equal(run({ BROWSER_SCAN_CHECK_EXIT: '19' }).status, 19);
  assert.notEqual(run({ BROWSER_SCAN_VERSION: '0.19.8' }).status, 0);
  assert.notEqual(run({ BROWSER_SCAN_VERSION: '0.20.20' }).status, 0);
  for (const name of ['jope-browser-predict', 'feat-math', 'jope-runtime']) {
    const result = run({ BROWSER_SCAN_OMIT: name });
    assert.notEqual(result.status, 0);
    assert(result.stderr.includes('missing: ' + name));
  }
  console.log(JSON.stringify({ browserSecurityCommandControls: 8, scope: 'actual entrypoint with scanner stubs; real dependency scan follows separately' }));
} finally {
  fs.rmSync(scratch, { recursive: true });
}
