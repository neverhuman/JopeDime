import { JopeModel, saveState } from './client.mjs';
import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const executable = resolve(process.argv[2] || 'target/debug/jopedime');
const output = resolve(process.argv[3] || 'target/model-objects-example');
const fixtures = fileURLToPath(new URL('.', import.meta.url));
mkdirSync(output, { recursive: true });
const { model } = JopeModel.fit({ train: `${fixtures}train.csv`, target: 'y', task: 'regression',
  method: 'ridge', root: `${output}/models`, executable });
model.verify();
model.predict(`${fixtures}rows.csv`, `${output}/predictions.csv`);
model.export('bound-json', `${output}/state.json`);
// A different producer imports the complete bound state, with the same envelope contract.
const imported = saveState({ format: 'org.jopedime/battle-artifact/v2', primary: 'state.json',
  files: { 'state.json': { bytes: readFileSync(`${output}/state.json`), mediaType: 'application/json' } },
  root: `${output}/models`, executable });
if (!imported.inspection.executable) throw new Error('Bound-state import needs a supported decoder');
console.log(JSON.stringify({ native: model.inspect(), imported, predictions: `${output}/predictions.csv` }, null, 2));
