const assert=require('node:assert/strict');
const {flywheelView,observedSum}=require('./flywheel.js');
const result=[];
function check(name,fn){if(process.argv[2]&&name!==process.argv[2])return;fn();result.push({name,state:'passed'});}
const stage=(view,id)=>view.stages.find(x=>x.id===id);
check('missing evidence never becomes zero',()=>{
 const v=flywheelView({},1000);
 for(const id of ['data','encoding','proposal','execution','learning'])assert.equal(stage(v,id).metric,'Not reported');
 assert.equal(v.fresh,false);assert.equal(v.age,null);
});
check('fresh campaign and GPU do not prove learning',()=>{
 const v=flywheelView({at:995,hosts:[{host:'xbabe1',at:995},{host:'xbabe3',at:995},{host:'xbabe2',at:996,neural_continuous:true,gpus:[{utilization_percent:100}],neural_service:{ActiveState:'active',SubState:'running'},neural_training:{completed_batches:17773}}]},1000);
 assert(v.fresh);assert.equal(stage(v,'learning').state,'held');assert.equal(stage(v,'admission').state,'build');assert.match(v.learning,/not yet qualified/);
});
check('future source observation refused',()=>{
 assert.equal(flywheelView({at:1001,hosts:[{host:'xbabe2',at:995}]},1000).age,null);
 assert.equal(flywheelView({at:995,hosts:[{host:'xbabe2',at:1001}]},1000).fresh,false);
});
check('boundary age ninety is stale',()=>assert.equal(flywheelView({at:910,hosts:[{host:'xbabe2',at:999}]},1000).fresh,false));
check('partial host source cannot appear fresh',()=>assert.equal(flywheelView({at:995,hosts:[]},1000).fresh,false));
check('invalid counts never admitted',()=>{
 for(const value of [-1,NaN,Infinity,'49',1.5,Number.MAX_SAFE_INTEGER+1]){
  const v=flywheelView({hosts:[{host:'xbabe2',evergreen:{published:value},neural_training:{published_results:value}}]},1000);
  assert.equal(stage(v,'data').metric,'Not reported');assert.equal(stage(v,'execution').metric,'Not reported');
 }
});
check('published zero stays a recorded zero',()=>assert.match(stage(flywheelView({hosts:[{host:'xbabe2',evergreen:{published:0}}]},1000),'data').metric,/^0 variants/));
check('encoding sum scope and absence distinct',()=>{
 assert.equal(observedSum([]),null);assert.equal(observedSum([null,undefined]),null);
 assert.equal(observedSum([4,0,undefined,9]),13);assert.equal(observedSum([Number.MAX_SAFE_INTEGER,1]),null);
 const v=flywheelView({hosts:[{host:'xbabe2',rolling_kumo:{verified_new_vectors:5}},{host:'xbabe3',kumos_publication:{completed_variants:7}}]},1000);
 assert.match(stage(v,'encoding').metric,/^12 reported vectors/);
});
check('new counter requires new review rather than live claim',()=>{
 const v=flywheelView({hosts:[{host:'xbabe2',neural_training:{completed_batches:17774}}]},1000);
 assert(v.superseded);assert.equal(stage(v,'learning').state,'held');assert.match(stage(v,'learning').evidence,/not live optimizer proof/);
});
check('measured proposals and results remain separate',()=>{
 const v=flywheelView({hosts:[{host:'xbabe2',neural_training:{model_generated_programs:75,published_results:12,winrate_vs_random:1}}]},1000);
 assert.match(stage(v,'proposal').metric,/^75 /);assert.match(stage(v,'execution').metric,/^12 /);
 assert.equal(stage(v,'evaluation').state,'unqualified');assert.match(stage(v,'evaluation').metric,/unmeasured/);
});
check('renderer is pure and preserves source history',()=>{
 const s={at:995,hosts:[{host:'xbabe2',at:995,neural_training:{completed_batches:17773,loss_history:[{batch:1,train_loss:3,validation_loss:4}]}}]};
 const before=JSON.stringify(s);flywheelView(s,1000);assert.equal(JSON.stringify(s),before);
});
check('malformed host records do not invent counts',()=>assert.equal(stage(flywheelView({hosts:[null,0,'x']},1000),'data').metric,'Not reported'));
console.log(JSON.stringify({schema:'codex.new-dashboard-flywheel-fixtures/v1',executed:result.length,failed:0,cases:result,production_service_runs:0,optimizer_updates:0}));
