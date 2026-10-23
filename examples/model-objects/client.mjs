// A dependency-free Node client. TypeScript consumers use client.d.mts.
// Fitting/inference and all store validation are delegated to the shared native implementation.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';

function call(executable, args) {
  const result = spawnSync(executable, args, { encoding: 'utf8', maxBuffer: 16 << 20, shell: false });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`jopedime exit ${result.status}: ${result.stderr || result.stdout}`);
  return JSON.parse(result.stdout);
}

export class JopeModel {
  constructor(path, executable = 'jopedime') {
    this.path = resolve(path);
    this.executable = /[/\\]/.test(executable) ? resolve(executable) : executable;
  }
  static fit({ train, target, task = 'auto', method = 'auto', root = 'models', executable = 'jopedime' }) {
    const receipt = call(executable, ['fit', '--train', train, '--target', target, '--task', task,
      '--method', method, '--model-root', root, '--threads', '1', '--json']);
    return { model: new JopeModel(receipt.outputs[0].path, executable), receipt };
  }
  inspect() { return call(this.executable, ['model', 'inspect', '--model', this.path]); }
  verify() { return call(this.executable, ['model', 'verify', '--model', this.path, '--require-executable']); }
  predict(input, output) {
    return call(this.executable, ['predict', '--model', this.path, '--input', input,
      '--output', output, '--threads', '1', '--json']);
  }
  export(format, output) {
    return call(this.executable, ['model', 'export', '--model', this.path, '--format', format, '--output', output]);
  }
}

// Independent estimators can persist their own versioned codec. No JavaScript object dump is
// assumed to be portable: `format` and its decoder define the bytes and execution semantics.
export function saveState({ format, primary, files, metadata = {}, requiredCapabilities = [],
  root = 'models', executable = 'jopedime' }) {
  const absoluteRoot = resolve(root);
  mkdirSync(absoluteRoot, { recursive: true });
  const staging = mkdtempSync(join(absoluteRoot, '.import-'));
  try {
    const names = new Set();
    const descriptors = Object.entries(files).map(([path, part]) => {
      const lower = path.toLowerCase();
      if (!/^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$/.test(path) || path.endsWith('.')
          || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(path)
          || lower === 'manifest.json' || names.has(lower)) {
        throw new Error('Parts require unique flat filenames');
      }
      names.add(lower);
      const bytes = Buffer.from(part.bytes);
      writeFileSync(join(staging, path), bytes, { flag: 'wx', mode: 0o600 });
      return { path, media_type: part.mediaType, bytes: bytes.length,
        sha256: createHash('sha256').update(bytes).digest('hex') };
    });
    const manifest = { schema: 'jope.model-object/v1', minimum_reader_version: 1, format,
      producer: { name: 'jopedime-node-example', version: '1', language: 'javascript' },
      primary, required_capabilities: requiredCapabilities, files: descriptors, metadata };
    const manifestPath = join(staging, 'manifest.json');
    writeFileSync(manifestPath, JSON.stringify(manifest), { flag: 'wx', mode: 0o600 });
    return call(executable, ['model', 'import', '--manifest', manifestPath, '--root', absoluteRoot]);
  } finally {
    // Only this call's unique staging directory is removed; committed objects remain intact.
    rmSync(staging, { recursive: true, force: true });
  }
}
