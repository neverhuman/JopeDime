'use strict';
// Counts come from the existing collector. Qualification comes from a dated review,
// never from campaign flags, utilization, a checkpoint filename or elapsed time.
const review = Object.freeze({
  at: '2026-10-10T14:26:47Z', generation: 'v2 build',
  learner: 'Build only · launch not authorized',
  historicalStep: 17773,
  run: 'native-continuous-20261009-v1',
  config: 'fec184217e7c50a3fc3bc1cc7e33edafdbd218d1fafa1bd0d68debf626dcd4ed',
  executionConfig: 'f358735358d2f56ad12106bcf327454d6af87ca440c46fc35e86558414b1de93',
  build: 'Independent review joins the new CUDA next-update parity, ancestry and dry-run receipts. Final-write deadline and restart outbox safeguards still need correction; new-target learning remains unqualified.',
  monitoring: 'Last owned publisher closed normally. A successor is held on complete-history input custody and its fixed memory budget.',
  provenance: 'R19 build-only ruling; R20/R21 takeover proof; independent v2-r2 receipt 06d08998; V14 normal terminal review.'
});

function validCount(value) {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null;
}
function observedSum(values) {
  const present = values.filter(value => value !== undefined && value !== null);
  if (!present.length || !present.every(value => validCount(value) !== null)) return null;
  const sum=present.reduce((a, b) => a + b, 0);
  return validCount(sum);
}
function snapshotAge(at, now) {
  return typeof at === 'number' && Number.isFinite(at) && at <= now ? now - at : null;
}
function flywheelView(snapshot, now = Date.now() / 1000, sourceUnavailable=false) {
  const hosts = Array.isArray(snapshot?.hosts) ? snapshot.hosts.filter(value=>value&&typeof value==='object') : [];
  const host = hosts.find(value => value.host === 'xbabe2') || {};
  const neural = host.neural_training || {};
  const growth = host.evergreen || {};
  const encoded = observedSum([host.rolling_kumo?.verified_new_vectors,
    ...hosts.flatMap(value => [value.evergreen_encoding_publication?.completed_variants,
      value.kumos_publication?.completed_variants, value.growth10_kumos_publication?.completed_variants])]);
  const age = snapshotAge(snapshot?.at, now);
  const fresh = !sourceUnavailable && age !== null && age < 90 && ['xbabe1','xbabe2','xbabe3'].every(id=>{
    const observed=hosts.find(value=>value.host===id);
    const hostAge=snapshotAge(observed?.at,now);
    return hostAge!==null&&hostAge<90;
  });
  const count = (value, unit) => validCount(value) === null ? 'Not reported' : validCount(value).toLocaleString() + ' ' + unit;
  const suffix = fresh ? 'Observed source record' : 'Retained source record';
  const step = validCount(neural.completed_batches);
  const identityMatches = neural.run_id===review.run && neural.schema==='jope.flywheel-neural-progress/v1' &&
    neural.config_sha256===review.config && neural.execution_config_sha256===review.executionConfig;
  const superseded = !identityMatches || (step !== null && step > review.historicalStep);
  const stages = [
    {id:'data', title:'Data & lineage', state:'recorded', metric:count(growth.published,'variants in S3'),
      description:'Regression contexts carry their parent, exact split, transform, seed, target and input identity. Descendants keep their parent’s train or validation role.',
      evidence:suffix + '. Raw publication is separate from encoding, scoring and optimizer exposure.',
      next:'Join each new context to verified inputs, source metadata and its inherited role before downstream use.'},
    {id:'encoding', title:'Kumo encoding', state:'recorded', metric:count(encoded,'reported vectors'),
      description:'Licensed Kumo Small supplies the frozen 2,051-value input. A raw dataset is usable only after exact encoding and publication readback.',
      evidence:suffix + '. Sum of the publication cohorts present in this snapshot; missing cohorts are unknown. This is partial reporting, not the complete original catalog or a full encoder census.',
      next:'Qualify fresh inputs without replaying completed encodings or admitting other encoder blocks.'},
    {id:'proposal', title:'Model → GP', state:'recorded', metric:count(neural.model_generated_programs,'model proposals'),
      description:'The policy decodes a typed program structure. Valid emitted proposals enter a durable execution queue; generation itself provides no reward.',
      evidence:suffix + '. Emitted proposals and measured results have different identities and counts.',
      next:'Resolve the exact proposed program, context, evaluator and global attempt history before scoring.'},
    {id:'execution', title:'Native & Rust', state:'recorded', metric:count(neural.published_results,'model results in S3'),
      description:'Run a GP on its exact context, emit working Rust, and require native/Rust agreement at the fixed 1e-9 tolerance before accepting a numeric result.',
      evidence:suffix + '. This is the neural campaign’s publication count, not a new evaluation or a proof of learning.',
      next:'Keep program parity failures nonnumeric with no reward and no retry; keep infrastructure and unknown outcomes distinct.'},
    {id:'evolution', title:'Typed GP evolution', state:'unqualified', metric:'Integration proof pending',
      description:'Mutate and recombine valid model-produced structures under the same context and equal budgets. Keep parent custody and compare measured outcomes.',
      evidence:'An integrated model → evolution → better measured target loop is not yet qualified by this dashboard.',
      next:'Produce a same-context, equal-budget comparison with exact ancestry; do not mix evaluator or objective versions.'},
    {id:'library', title:'Versioned game library', state:'recorded', metric:'Exact identities required',
      description:'Keep costs and scores tied to program, data, split, evaluator, objective and normalizer. Improving a model does not automatically invalidate an old measured play.',
      evidence:'Completed keys are cache-only. Missing artifacts and unknown attempts never authorize automatic reexecution.',
      next:'Qualify consumed membership, archive-aware resolution and tombstones before any deletion. Retention prices are forecasts until billed.'},
    {id:'targets', title:'Numerical targets', state:'build', metric:'Qualification pending',
      description:'Convert measured train-role outcomes into feedback with one explicit objective and normalizer. Dense supervised targets, teacher promotion and actor learning are separate.',
      evidence:review.build,
      next:'Close references and numerical preparation; independently verify target values and full input lineage.'},
    {id:'admission', title:'Closed-epoch admission', state:'build', metric:'Repeatable boundary pending',
      description:'Admit an immutable target generation at a closed epoch boundary while retaining ordered references, roles, fingerprints, planner and outbox custody.',
      evidence:'New source is build-only. Repeatable live adoption and actual new-target batch exposure remain unqualified.',
      next:'Verify complete membership, durable ACK and cumulative joins, then prove first actual exposure in the next admitted batch.'},
    {id:'learning', title:'Committed learning', state:'held', metric:count(step,identityMatches?'historical updates':'reported updates · identity unverified'),
      description:'An optimizer update counts only when planner/control commits join a physically changed model and preserved AdamW, CPU/CUDA RNG and checkpoint ancestry.',
      evidence:superseded ? 'Run/config identity is absent or differs, or a newer counter is present; refresh the dated build review. This is not live optimizer proof.' : 'The accepted v1 run is stopped. V2 is build-only; generated-result learning has not been proved.',
      next:'Finish deadline and restart custody safeguards, qualify owned resources, and obtain the required live approval. The pre-R14 CUDA next-update receipt has been independently joined.'},
    {id:'evaluation', title:'Same-context evaluation', state:'unqualified', metric:'Flywheel uplift unmeasured',
      description:'Compare learned proposals and evolved programs against frozen alternatives with matched contexts and budgets. Fixed loss, expanded validation and real games answer different questions.',
      evidence:'This view contains no qualified feedback-caused improvement, generalization, win-rate or Elo claim.',
      next:'Measure an independently qualified evaluation after new-target learning. Return eligible train feedback to the next generation.'}
  ];
  return {stages, age, fresh, superseded, status: sourceUnavailable ? 'Connection unavailable · retained observations' : fresh ? 'Fresh source observations' : 'Retained observations · refresh paused',
    reviewAt:review.at, learning:'Feedback → learning is not yet qualified', monitoring:review.monitoring};
}

function mountFlywheel(snapshot, sourceUnavailable=false) {
  const view = flywheelView(snapshot,Date.now()/1000,sourceUnavailable);
  const nodes = document.querySelector('#flywheelNodes');
  if (!nodes) return;
  const old = document.querySelector('.fly-step[aria-pressed="true"]')?.dataset.stage || 'targets';
  const focused = document.activeElement?.classList.contains('fly-step') ? document.activeElement.dataset.stage : null;
  nodes.replaceChildren();
  const stateLabel = {recorded:'Recorded evidence', build:'Build in progress', held:'Live gate held', unqualified:'Proof pending'};
  const detail = document.querySelector('#flywheelDetail');
  const select = id => {
    const stage = view.stages.find(value => value.id === id);
    for (const button of nodes.querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.stage === id));
    detail.replaceChildren();
    for (const [tag, cls, content] of [
      ['div','eyebrow',stateLabel[stage.state]],['h3','',stage.title],['p','fly-metric',stage.metric],
      ['p','',stage.description],['h4','','Evidence'],['p','',stage.evidence],['h4','','Next completion gate'],['p','',stage.next]
    ]) { const item = document.createElement(tag); item.className=cls; item.textContent=content; detail.append(item); }
  };
  view.stages.forEach((stage, index) => {
    const button = document.createElement('button');
    button.type='button';button.className='fly-step '+stage.state;button.dataset.stage=stage.id;
    button.setAttribute('aria-label', stage.title);button.setAttribute('aria-controls','flywheelDetail');
    for (const [cls, content] of [['fly-number',String(index+1).padStart(2,'0')],['fly-title',stage.title],
      ['fly-count',stage.metric],['fly-state',stateLabel[stage.state]]]) {
      const item = document.createElement('span');item.className=cls;item.textContent=content;button.append(item);
    }
    button.onclick=()=>select(stage.id);nodes.append(button);
  });
  select(old);
  if(focused)nodes.querySelector('[data-stage="'+focused+'"]')?.focus({preventScroll:true});
  document.querySelector('#flywheelReadiness').textContent=view.learning;
  document.querySelector('#flywheelFreshness').textContent=view.status;
  document.querySelector('#flywheelAge').textContent=view.age === null ? 'Observation time unavailable' :
    'Snapshot age '+Math.floor(view.age/60).toLocaleString()+' min · counts retain their original timestamp';
  document.querySelector('#flywheelReviewAt').textContent='Build review '+new Date(view.reviewAt).toISOString().replace('T',' ').replace('.000Z',' UTC');
}

if (typeof module !== 'undefined') module.exports={flywheelView,validCount,observedSum,snapshotAge,review};
