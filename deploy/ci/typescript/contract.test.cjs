// Exercise the emitted facade, with fault-injecting ABI doubles for paths a
// healthy Rust module cannot produce. Real Wasm parity is a separate required lane.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const filename=path.resolve(process.env.JOPE_TS_CONTRACT_SOURCE);
const source=fs.readFileSync(filename,'utf8');
function load({headers=null,bigEndian=false,mutate=()=>{},imports=[],compileError=false}={}) {
  const instances=[];
  class Module { static imports(){return imports;} }
  const wasm={Module,Memory:WebAssembly.Memory,
    compile:async()=>{if(compileError)throw new WebAssembly.CompileError('invalid');return new Module();},
    instantiate:async()=>{
      const memory=new WebAssembly.Memory({initial:2});
      let length=0;
      const state={memory,values:[],targets:[],text:'',status:0,trap:false,disposed:0,refuse:''};
      const kernel={memory,
        model_abi_version:()=>1,
        model_values:n=>state.refuse==='values'?0:64,
        model_labels:n=>state.refuse==='labels'?0:32768,
        model_text:n=>state.refuse==='text'?0:64,
        model_output:()=>65536,
        model_output_len:()=>length,
        model_fit:(rows,cols)=>{state.values=Array.from(new Float64Array(memory.buffer,64,rows*cols));state.targets=Array.from(new Float64Array(memory.buffer,32768,rows));if(state.trap)throw new WebAssembly.RuntimeError('fit trap');return state.status;},
        model_predict:(rows,cols)=>{if(state.trap)throw new WebAssembly.RuntimeError('predict trap');length=rows;new Float64Array(memory.buffer,65536,rows).fill(1);return state.status;},
        model_predict_text:()=>{state.text=new TextDecoder().decode(new Uint8Array(memory.buffer,64,32)).replace(/\0.*$/s,'');length=1;new Float64Array(memory.buffer,65536,1)[0]=2;return state.status;},
        model_dispose:()=>{state.disposed++;if(state.trap)throw new WebAssembly.RuntimeError('dispose trap');},
      };
      mutate(kernel,state); instances.push(state);return {exports:kernel};
    }};
  const exports={};
  const context=vm.createContext({exports,headers,WebAssembly:wasm,ArrayBuffer,Uint8Array,Float64Array,DataView,TextEncoder,
    Uint16Array:bigEndian?class {constructor(){this.buffer=new Uint8Array([0,1]).buffer;}}:Uint16Array});
  new vm.Script(source,{filename}).runInContext(context);
  return {create:exports.createModel,module:new Module(),instances};
}
const bytes=new Uint8Array([0]);
test('module sources, isolated instances and ABI validation',async()=>{
  const a=load();
  for(const input of [bytes,bytes.buffer,a.module]){const m=await a.create(input);m.dispose();}
  assert.equal(a.instances.length,3);
  await assert.rejects(load({compileError:true}).create(bytes),/invalid/);
  await assert.rejects(load({imports:[{}]}).create(bytes),/imports/);
  await assert.rejects(load({mutate:k=>delete k.memory}).create(bytes),/memory/);
  for(const name of ['model_abi_version','model_values','model_labels','model_text','model_output','model_output_len','model_fit','model_predict','model_predict_text','model_dispose'])
    await assert.rejects(load({mutate:k=>delete k[name]}).create(bytes),/incomplete/);
  await assert.rejects(load({mutate:k=>k.model_abi_version=()=>2}).create(bytes),/version/);
});
for(const bigEndian of [false,true])test('numeric lifecycle and transfers, endian='+bigEndian,async()=>{
  const a=load({bigEndian});const m=await a.create(bytes);const state=a.instances[0];
  assert.throws(()=>m.predict([[1]]),/fit required/);
  assert.throws(()=>m.predictStrings([['1']]),/fitted/);
  assert.throws(()=>m.predictCsv('x\n1'),/fitted/);
  for(const [x,y] of [[[],[]],[[[]],[1]],[[[1],[1,2]],[1,2]],[[['x']],[1]],[[[1]],[]],[[[1]],[NaN]],[[[1]],[Infinity]],[[[1]],['1']],[[[1]],new Array(1)]])assert.throws(()=>m.fit(x,y));
  for(const n of [NaN,-1,16777217,0.5])assert.throws(()=>m.fit({length:n,0:{length:1}},[]),/limit/);
  assert.equal(m.fit([new Float64Array([3,4]),[5,6]],new Float64Array([0,2])),m);
  assert.deepEqual(state.values,[3,4,5,6]);assert.deepEqual(state.targets,[0,2]);
  const out=m.predict([[7,8]]);assert.deepEqual(Array.from(out),[1]);
  assert.equal(m.score([[0,0],[1,1]],[0,2]),0);
  assert.equal(m.score([[0,0]],[2]),0);assert.equal(m.score([],[]),0);
  for(const y of [[],new Array(2),[0,NaN],[0,'2']])assert.throws(()=>m.score([[0,0],[1,1]],y));
  assert.throws(()=>m.predict([[1]]),/width/);
  state.status=1;assert.throws(()=>m.predict([[0,0]]),/status 1/);state.status=0;
  state.memory.grow(1);m.predict([[2,3]]);assert.deepEqual(Array.from(out),[1]);
  m.dispose();m.dispose();assert.equal(state.disposed,1);assert.throws(()=>m.predict([]),/disposed/);assert.throws(()=>m.score([],[]),/disposed/);
});
for(const bigEndian of [false,true])test('string/CSV contract, endian='+bigEndian,async()=>{
  const a=load({headers:['h'],bigEndian});const m=await a.create(bytes);const state=a.instances[0];
  assert.throws(()=>m.fit([[1]],[2]),/refitted/);assert.throws(()=>m.predict([[1]]),/predictStrings/);
  for(const rows of [[[]],[['a','b']],[[1]],[new Array(1)]])assert.throws(()=>m.predictStrings(rows));
  assert.deepEqual(Array.from(m.predictStrings([['a,"b\n雪']])),[2]);
  assert.match(state.text,/"a,""b\n雪"/);
  assert.deepEqual(Array.from(m.predictCsv('h\n2\n')),[2]);
  assert.throws(()=>m.predictCsv('h\n2\n',2),/count mismatch/);
  assert.throws(()=>m.predictCsv('x'.repeat(16777217)),/limit/);
  state.status=1;assert.throws(()=>m.predictCsv('bad'),/status/);state.status=0;m.predictCsv('h\n1');m.dispose();
});
test('allocation refusal and traps invalidate the affected instance',async()=>{
  for(const refuse of ['values','labels']){
    const a=load();const m=await a.create(bytes);a.instances[0].refuse=refuse;
    assert.throws(()=>m.fit([[1]],[2]),/allocation/);assert.throws(()=>m.predict([[1]]),/disposed/);
  }
  const a=load({headers:['h']});const m=await a.create(bytes);a.instances[0].refuse='text';
  assert.throws(()=>m.predictCsv('h\n1'),/allocation/);assert.throws(()=>m.predictCsv('h\n1'),/disposed/);
  for(const operation of ['fit','predict','dispose']){
    const a=load();const m=await a.create(bytes);if(operation==='predict')m.fit([[1]],[2]);a.instances[0].trap=true;
    assert.throws(()=>operation==='fit'?m.fit([[1]],[2]):operation==='predict'?m.predict([[1]]):m.dispose(),/trap/);
    assert.throws(()=>m.fit([[1]],[2]),/disposed/);m.dispose();
  }
});
test('qualification verifier rejects zero tests, missing census cases and missing coverage',()=>{
  const {spawnSync}=require('node:child_process');
  const os=require('node:os');
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'typescript-receipts-'));
  const evidence=path.join(dir,'target/typescript-evidence');fs.mkdirSync(evidence,{recursive:true});
  const verifier=path.resolve('deploy/ci/typescript/verify.py');
  const ok='test result: ok. 1 passed; 0 failed\n';
  const execution='TypeScript cjs: 1 predictions qualified\nTypeScript esm: 1 predictions qualified\n';
  const records=['recipe','bound','legacy-bound'].map(kind=>({kind,total:1,executed:['x'],refused:[],unaccounted:0}));records.push({kind:'v2',total:1,executed:1,unaccounted:0});
  const census=ok+execution.repeat(10)+records.map(c=>'TypeScript census: '+JSON.stringify(c)+'\n').join('')+'TypeScript metadata: strict heading qualified\n';
  const coverage='pass 7\nfail 0\nmodel.js | 100.00 | 100.00 | 100.00\n';
  for(const name of ['recipes','fitted','bindings','pipeline','cli'])fs.writeFileSync(path.join(evidence,name+'.log'),ok+execution);
  const run=()=>spawnSync('python3',[verifier],{cwd:dir,encoding:'utf8'});
  const restore=()=>{fs.writeFileSync(path.join(evidence,'census.log'),census);fs.writeFileSync(path.join(evidence,'contracts.log'),coverage);fs.writeFileSync(path.join(evidence,'cli.log'),ok);};
  try{
    restore();let result=run();assert.equal(result.status,0,result.stderr);
    for(const changed of [census.replace('"unaccounted":0','"unaccounted":1'),census.replace('TypeScript cjs:','absent'),census.replace('"kind":"bound"','"kind":"recipe"')]){
      restore();fs.writeFileSync(path.join(evidence,'census.log'),changed);assert.notEqual(run().status,0);
    }
    for(const changed of ['',coverage.replace('100.00','99.99'),coverage.replace('model.js','unrelated.js')]){
      restore();fs.writeFileSync(path.join(evidence,'contracts.log'),changed);assert.notEqual(run().status,0);
    }
    restore();fs.writeFileSync(path.join(evidence,'cli.log'),'test result: ok. 0 passed; 0 failed\n');assert.notEqual(run().status,0);
  }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
