const assert=require('node:assert/strict');
const {flywheelView,review}=require('./flywheel.js');
const hosts=[{host:'xbabe1',at:995},{host:'xbabe2',at:995},{host:'xbabe3',at:995}];
const cases=[];
function check(name,fn){if(process.argv[2]&&name!==process.argv[2])return;fn();cases.push({name,state:'passed'});}
check('fresh xb2 cannot certify stale xb3',()=>assert.equal(flywheelView({at:995,hosts:hosts.map(x=>x.host==='xbabe3'?{...x,at:800}:x)},1000).fresh,false));
check('missing host does not certify whole source',()=>assert.equal(flywheelView({at:995,hosts:hosts.slice(0,2)},1000).fresh,false));
check('all three current hosts can label observations fresh',()=>assert.equal(flywheelView({at:995,hosts},1000).fresh,true));
check('partial encoding explicitly reports subset',()=>{
 const v=flywheelView({at:995,hosts:[{host:'xbabe2',at:995,rolling_kumo:{verified_new_vectors:5}}]},1000);
 const stage=v.stages.find(x=>x.id==='encoding');assert.match(stage.metric,/5 reported vectors/);assert.match(stage.evidence,/partial reporting/);assert.match(stage.evidence,/missing cohorts are unknown/);
});
check('reset run with new config invalidates old review',()=>{
 const v=flywheelView({hosts:[{host:'xbabe2',neural_training:{schema:'jope.flywheel-neural-progress/v1',config_sha256:'b'.repeat(64),execution_config_sha256:review.executionConfig,completed_batches:1}}]},1000);
 assert(v.superseded);assert.match(v.stages.find(x=>x.id==='learning').metric,/identity unverified/);
});
check('exact captured identity permits historical label without live claim',()=>{
 const v=flywheelView({hosts:[{host:'xbabe2',neural_training:{run_id:review.run,schema:'jope.flywheel-neural-progress/v1',config_sha256:review.config,execution_config_sha256:review.executionConfig,completed_batches:17773}}]},1000);
 assert.equal(v.superseded,false);assert.match(v.stages.find(x=>x.id==='learning').metric,/historical updates/);assert.match(v.learning,/not yet qualified/);
});
console.log(JSON.stringify({schema:'codex.new-dashboard-review-boundaries/v1',executed:cases.length,failed:0,cases,production_service_runs:0,optimizer_updates:0}));
