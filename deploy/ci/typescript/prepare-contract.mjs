import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {execFileSync} from 'node:child_process';
const root=resolve('target/typescript-contract');mkdirSync(root,{recursive:true});
const template=readFileSync('crates/feat-math/src/typescript/model.ts','utf8');
// Substitute only the emitter's header marker. Each VM has its own binding,
// allowing coverage of recipe and fitted branches in the same template body.
writeFileSync(root+'/model.ts',template.replace('__BOUND_HEADERS__','(globalThis as unknown as {headers: ReadonlyArray<string> | null}).headers'));
execFileSync(resolve('deploy/ci/typescript/node_modules/.bin/tsc'),['--strict','--target','ES2020','--module','commonjs','--lib','ES2020,DOM',root+'/model.ts'],{stdio:'inherit'});
