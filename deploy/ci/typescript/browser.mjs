import {nativeWebkitOptions} from './native.mjs';
import {chromium,firefox,webkit} from 'playwright';
import {createServer} from 'node:http';
import {readFile,readdir,writeFile,mkdir} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
import assert from 'node:assert/strict';
const root=resolve('target/typescript-census');
const cases=[];
for(const kind of ['recipe','bound','v2'])
  for(const name of await readdir(resolve(root,kind)))cases.push({kind,name});
assert.ok(cases.length>0,'no browser fixtures');
const census=JSON.parse(await readFile('target/typescript-evidence/census.json','utf8'));
const expectedCases=census.reduce((n,c)=>n+(c.kind==='v2'?c.executed:c.executed.length),5);
assert.equal(cases.length,expectedCases,'browser corpus must match the completed census');
const server=createServer(async(req,res)=>{
  try{
    const pathname=new URL(req.url,'http://localhost').pathname;
    // A real same-origin document is required before browser module workers run.
    // Firefox can wait indefinitely on its internal error page after a root404.
    if(pathname==='/') {
      res.writeHead(200,{'content-type':'text/html; charset=utf-8','cache-control':'no-store'});
      res.end('<!doctype html><meta charset="utf-8"><title>TypeScript model qualification</title>');
      return;
    }
    const file=resolve(root,'.'+decodeURIComponent(pathname));
    if(!file.startsWith(root+sep))throw new Error('path');
    const body=await readFile(file);
    res.writeHead(200,{'content-type':({'.wasm':'application/wasm','.js':'text/javascript','.json':'application/json'})[extname(file)]||'text/plain','cache-control':'no-store'});res.end(body);
  }catch{res.writeHead(404);res.end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const origin='http://127.0.0.1:'+server.address().port;
const results=[];
const engines={chromium,firefox,webkit};
const selected=(process.env.JOPE_TS_BROWSERS||'chromium,firefox,webkit').split(',');
assert.ok(selected.length>0&&selected.every(name=>engines[name]),'unknown browser');
assert.equal(new Set(selected).size,selected.length,'duplicate browser');
if(process.env.GITHUB_ACTIONS==='true') assert.deepEqual([...selected].sort(),['chromium','firefox','webkit'],'CI must execute every engine');
try{
  for(const name of selected){
    const engine=engines[name];
    const browser=await engine.launch({headless:true,...(name==='webkit'?nativeWebkitOptions(engine):{}),...(name==='chromium'&&process.env.JOPE_TS_CHROMIUM_EXECUTABLE?{executablePath:process.env.JOPE_TS_CHROMIUM_EXECUTABLE,args:['--no-sandbox','--disable-dev-shm-usage']}: {})});
    const version=browser.version();
    try{
      console.log(name+' '+version+': starting complete corpus');
      const page=await browser.newPage();await page.goto(origin+'/');
      for(const fixture of cases){
        const url=origin+'/'+fixture.kind+'/'+fixture.name;
        const result=await page.evaluate(async({url,fitted})=>{
          const {createModel}=await import(url+'/esm/model.js');
          const bytes=await (await fetch(url+'/model.wasm')).arrayBuffer();
          const compiled=await WebAssembly.compile(bytes);
          if(WebAssembly.Module.imports(compiled).length)throw new Error('imports');
          const input=await (await fetch(url+'/input.json')).json();
          const expected=await (await fetch(url+'/expected.json')).json();
          const m=await createModel(compiled);if(!fitted)m.fit(input.train,input.targets);
          const out=fitted?m.predictStrings(input.rows):m.predict(input.test);
          if(out.length!==expected.length)throw new Error('length');
          out.forEach((v,i)=>{if(!Number.isFinite(v)||Math.abs(v-expected[i])>1e-8*(1+Math.abs(expected[i])))throw new Error('row '+i+': '+v+' != '+expected[i]);});
          const saved=Array.from(out);m.dispose();m.dispose();
          if(saved.some((v,i)=>v!==out[i]))throw new Error('output invalidated');
          // Compile/instantiate and run the identical model in an actual module worker.
          const blob=new Blob([`self.onmessage=async({data})=>{try{const {createModel}=await import(data.url+'/esm/model.js');const m=await createModel(data.compiled);if(!data.fitted)m.fit(data.input.train,data.input.targets);const p=data.fitted?m.predictStrings(data.input.rows):m.predict(data.input.test);m.dispose();self.postMessage({values:Array.from(p)});}catch(e){self.postMessage({error:String(e)});}};`],{type:'text/javascript'});
          const workerUrl=URL.createObjectURL(blob);const worker=new Worker(workerUrl,{type:'module'});
          try{
            const reply=await new Promise((resolve,reject)=>{
              const timer=setTimeout(()=>reject(new Error('module worker timed out: '+url)),30000);
              worker.onmessage=e=>{clearTimeout(timer);resolve(e.data);};
              worker.onerror=e=>{clearTimeout(timer);reject(new Error(e.message));};
              worker.postMessage({url,compiled,input,fitted});
            });
            if(reply.error)throw new Error(reply.error);
            if(JSON.stringify(reply.values)!==JSON.stringify(saved))throw new Error('worker mismatch');
          }finally{worker.terminate();URL.revokeObjectURL(workerUrl);}
          return out.length;
        },{url,fitted:fixture.kind==='bound'});
        results.push({browser:name,version,...fixture,predictions:result,worker:true});
        if(results.length%20===0)console.log(name+': '+fixture.kind+'/'+fixture.name+' ('+results.length+' completed)');
      }
      console.log(name+' '+version+': '+cases.length+' ESM and module worker fixtures passed');
    }finally{await browser.close();}
  }
  assert.equal(results.length,cases.length*selected.length);
  await mkdir('target/typescript-evidence',{recursive:true});
  await writeFile('target/typescript-evidence/browsers.json',JSON.stringify({source:process.env.JOPE_SOURCE_SHA||null,cases,results},null,2));
}finally{await new Promise(r=>server.close(r));}
