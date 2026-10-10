// Campaign configuration and retained progress do not establish a live optimizer.
// This snapshot schema has no verified physical optimizer-change receipt.
function optimizerDisplay(s,h,nowSeconds=Date.now()/1000){
 const service=h.neural_service;
 const fresh=t=>typeof t==='number'&&Number.isFinite(t)&&nowSeconds-t>=0&&nowSeconds-t<90;
 const failedRead=(h.errors||[]).some(x=>typeof x==='string'&&x.startsWith('Neural training:'));
 if(failedRead)return {state:'unknown',text:'Optimizer status unavailable: training source verification failed. Retained observations do not establish current progress.'};
 if(!fresh(s.at)||!fresh(h.at))return {state:'unknown',text:'Optimizer status unknown: service observations are stale or unavailable. Recorded loss and batch history are retained.'};
 if(!service||typeof service!=='object')return {state:'unknown',text:'Optimizer status unknown: no current trainer service observation.'};
 const active=service.ActiveState,sub=service.SubState;
 if(active==='failed')return {state:'stopped',text:'Optimizer is stopped: trainer service failed'+(service.Result?' ('+String(service.Result)+')':'')+(service.ExecMainStatus!=null?' · recorded exit status '+String(service.ExecMainStatus):'')+'.'};
 if(active==='inactive')return {state:'stopped',text:'Optimizer is stopped: trainer service is inactive. Completed batches and losses are historical observations.'};
 if(active==='deactivating')return {state:'stopping',text:'Trainer service is stopping; current optimizer execution is unverified.'};
 if(active==='activating')return {state:'starting',text:'Trainer service is starting; no current committed optimizer progress has been verified.'};
 if(active==='active'&&sub==='running')return {state:'running_unverified',text:'Trainer service reports active/running; current committed optimizer progress is unverified.'};
 return {state:'unknown',text:'Optimizer status unknown: trainer service state '+String(active||'unobserved')+'/'+String(sub||'unobserved')+'.'};
}
