"""Original-context native execution + emitted Rust parity, distinct cache schema."""
from dataclasses import asdict
import json,math,struct,time
from pathlib import Path
import numpy as np
from native_inputs import OriginalScoreContext,need,sha,canonical,digest
from jope_flywheel.program_io import bound,create,read,tick
from jope_flywheel.timing import native_run as run
from jope_flywheel.program_worker import split_scale,csv_bytes,compare_predictions
from native_failure import ParityRejected, parity_diagnostic
from jope_flywheel.score_join import build_records,validate_records
from jope_flywheel.program_gate import ARTIFACTS,ARTIFACT_LIMITS,object_bytes
from jope_flywheel.s3_library import S3Library,PREFIX
SCHEMA='jope.native-original-program-result/v1'

def request_identity(context,program):
 need(type(context)is OriginalScoreContext,'original typed context')
 return digest({'schema':'jope.native-original-evaluation-key/v1','context':asdict(context),'program_sha256':sha(program.encode()),'protocol':'jope.s1-two-fold/v1'})

def local_entry(context,program,bodies):
 return {'schema':SCHEMA,'context':asdict(context),'evaluation_key':request_identity(context,program),'program_sha256':sha(program.encode()),'status':'completed','measurement':'executed',
 'artifacts':{role:{'sha256':sha(body),'bytes':len(body)}for role,body in bodies.items()}}

def verify_bodies(context,program,entry,bodies,nativecanonical_json):
 need(entry['schema']==SCHEMA and entry['context']==asdict(context),'exact original context')
 key=request_identity(context,program);ph=sha(program.encode());need(entry['evaluation_key']==key and entry['program_sha256']==ph and entry['status']=='completed'and entry['measurement']=='executed','measured program identity')
 need(set(bodies)==set(ARTIFACTS)==set(entry['artifacts']),'full artifact closure')
 for role,b in bodies.items():need(0<len(b)<=ARTIFACT_LIMITS[role]and sha(b)==entry['artifacts'][role]['sha256']and len(b)==entry['artifacts'][role]['bytes'],'artifact '+role)
 need(bodies['program']==program.encode(),'canonical GP string');raw=bodies['input'];need(raw[:4]==b'JTT1'and sha(raw)==context.input_sha256,'registered original input')
 n,nt,p,k=struct.unpack('<IIIB',raw[4:17]);end=17+n+8*n*(p+1)
 need(n>=50 and nt>=10 and 1<=p<=512 and k==2 and len(raw)==end+8*nt*(p+1),'original S1 shape')
 # Decode no outer feature/target value. Only inspect mask bytes.
 need(all(raw[end+i*(p+1)*8+p*8:end+(i+1)*(p+1)*8]==b'\0'*8 for i in range(nt)),'outer targets must be masked')
 folds=np.frombuffer(raw[17:17+n],dtype=np.uint8);data=np.frombuffer(raw[17+n:end],dtype='<f8').reshape(n,p+1)
 need(np.isfinite(data).all()and set(folds.tolist())=={0,1}and min(np.bincount(folds))>=25,'finite two-fold training rows')
 need(sha(data[:,-1].astype('<f8').tobytes())==context.target_sha256 and sha(folds.tobytes())==context.split_sha256,'original target/split')
 e=json.loads(bodies['evaluation']);parity=json.loads(bodies['rust_parity'])
 need(e['schema']=='jope.s1-evaluation/v1'and e['status']=='completed'and e['fold_protocol']=='jope.s1-two-fold/v1'and e['inner_cv_folds']==2 and e['input_semantics']=='unscaled_fit_rows_v1'and 'official_test'not in e,'native two fold protocol')
 need(e['program']['dsl']==program and e['program']['program_sha256']==ph and len(e['inner_cv'])==2,'native canonical candidate')
 scores=[]
 for i,f in enumerate(e['inner_cv']):
  expected=data[folds==i,-1];pred=f['predictions'];target=f['targets'];rmse=f['metrics']['rmse'];split=split_scale(data[:,:p],data[:,-1],folds,i)
  need(f['index']==i and f['kind']=='inner_cv'and f['score_rows']==len(expected)and f['train_rows']==n-len(expected)and f['scaler_sha256']==split['scaler_sha256']and f['serialized_prediction_replay_exact']is True,'fit-only memberships and scaler')
  need(len(pred)==len(target)==len(expected)and np.array_equal(np.array(target,dtype='<f8'),expected)and all(math.isfinite(x)for x in pred+target),'actual target/prediction closure')
  observed=math.sqrt(math.fsum((a-b)**2 for a,b in zip(pred,target))/len(pred));need(type(rmse)in(int,float)and math.isfinite(rmse)and rmse>=0 and math.isclose(rmse,observed,rel_tol=1e-10,abs_tol=1e-12),'RMSE reconstruction');scores.append(rmse)
 need(parity['schema']=='jope.flywheel-rust-parity/v1'and parity['passed']is True and parity['executed']is True and parity['folds_verified']==2 and parity['evaluation_key']==key and parity['program_sha256']==ph and parity['evaluator_sha256']==context.evaluator_sha256,'executed Rust parity')
 for role in ('input','evaluation','rust_source','rust_executable'):need(parity[role+'_sha256']==sha(bodies[role]),'parity identity')
 provenance=parity['native_provenance'];need(provenance['input_sha256']==context.input_sha256 and provenance['evaluator_sha256']==context.evaluator_sha256 and provenance['program_sha256']==ph and provenance['evaluation_sha256']==sha(bodies['evaluation']),'native producer')
 need(provenance['input_descriptor']['context']['context_id']==context.identity,'original descriptor')
 validate_records(bodies['score_records'],context,ph,key,scores,sha(bodies['rust_parity']),provenance['source_commit'],provenance['execution_timestamp'],nativecanonical_json)
 return {'evaluation_key':key,'program_sha256':ph,'context_id':context.identity,'inner_fold_rmse':scores,'measurement':'executed','schema':SCHEMA,'teaching_eligible':False}

def verified_measurement(store,context,program,entry):
 bodies={r:object_bytes(store,entry['artifacts'][r])for r in ARTIFACTS}
 return verify_bodies(context,program,entry,bodies,store.nativecanonical_json)

class NativeLibrary(S3Library):
 def commit(self,context,program,entry):
  measured=verified_measurement(self,context,program,entry);ref=self.put(PREFIX+measured['evaluation_key']+'/record.json',canonical(entry));return {'record':ref,'measurement':measured}

class NativeEvaluator:
 def __init__(self,inputs,store,native,config,deadline,plan):self.inputs,self.store,self.native,self.config,self.deadline,self.plan=inputs,store,native,config,deadline,plan
 def execute(self,descriptor,phase,program,dest):
        context,raw,x,y,folds,descriptor=self.inputs.describe(descriptor,phase)
        key=request_identity(context,program);dest=Path(dest);dest.mkdir(exist_ok=False)
        need(self.native.canonical_program(program)==program,'canonical input')
        create(dest/'input.jtt1',raw); create(dest/'input-descriptor.json',canonical(descriptor))
        candidate = {'dsl':program,'program_sha256':sha(program.encode()),'birth_seq':0,'parents':[],
                     'operator':'observed_external_program'}
        create(dest/'candidate.json',canonical(candidate))
        for role in ('worker','emitter','rustc'): bound(self.config[role],64*1024**2)
        commands = {}
        commands['lambda'] = run([self.config['worker']['path'],'--fit-s1',str(dest/'input.jtt1'),
            str(dest/'candidate.json'),str(dest/'evaluation.json')],dest,'lambda',self.deadline,90)
        evaluation_bytes = read(dest/'evaluation.json',4*1024**2); evaluation = json.loads(evaluation_bytes)
        need(evaluation['schema'] == 'jope.s1-evaluation/v1' and evaluation['program'] == candidate and
             evaluation['status'] == 'completed' and evaluation['fold_protocol'] == 'jope.s1-two-fold/v1' and
             evaluation['inner_cv_folds'] == 2 and evaluation['input_semantics'] == 'unscaled_fit_rows_v1' and
             'official_test' not in evaluation and len(evaluation['inner_cv']) == 2, 'complete native two-fold evaluation')
        x,y,folds = (x,y,folds); scaled = [split_scale(x,y,folds,i) for i in range(2)]
        for i,split in enumerate(scaled):
            fold = evaluation['inner_cv'][i]
            need(fold['index'] == i and fold['kind'] == 'inner_cv' and fold['scaler_sha256'] == split['scaler_sha256'] and
                 fold['train_rows'] == split['train_rows'] and fold['score_rows'] == split['score_rows'] and
                 fold['targets'] == split['targets'] and fold['serialized_prediction_replay_exact'] is True,
                 'native fit-only scaler/membership/model replay')
            create(dest/('fold'+str(i)+'-train.csv'),csv_bytes(split['train']))
            create(dest/('fold'+str(i)+'-validation.csv'),csv_bytes(split['validation']))
        commands['emit'] = run([self.config['emitter']['path'],str(dest/'candidate.json'),str(dest/'emitted')],
                               dest,'emit',self.deadline,30)
        manifest_bytes = read(dest/'emitted/manifest.json'); manifest = json.loads(manifest_bytes)
        code = read(dest/'emitted/model.rs',4*1024**2)
        need(manifest['schema'] == 'battle-artifact-v1' and manifest['program_sha256'] == candidate['program_sha256'] and
             manifest['rs_sha256'] == sha(code) and manifest['program'] == program and
             read(dest/'emitted/program.dsl') == program.encode(), 'native emitted source/program identity')
        binary = dest/'emitted/model'
        commands['compile'] = run([self.config['rustc']['path'],'--edition=2021','-O',str(dest/'emitted/model.rs'),'-o',str(binary)],
                                   dest,'compile',self.deadline,120)
        checks,predictions = [],[]
        for i,split in enumerate(scaled):
            predicted = dest/('fold'+str(i)+'-predictions.csv')
            commands['execute'+str(i)] = run([str(binary),str(dest/('fold'+str(i)+'-train.csv')),
                str(dest/('fold'+str(i)+'-validation.csv')),str(predicted)],dest,'execute'+str(i),self.deadline,90)
            body = read(predicted,2*1024**2); predictions.append(body.decode())
            values = [float(line) for line in body.decode().splitlines()]
            actual = [v*split['yspan']+split['ylo'] for v in values]
            try:
                result = compare_predictions(actual,evaluation['inner_cv'][i]['predictions'])
            except ValueError as error:
                # Only this unchanged fixed-parity rejection is recoverable per candidate.
                if str(error) != 'emitted/native prediction parity failed; tolerance is fixed': raise
                diagnostic = parity_diagnostic(actual,evaluation['inner_cv'][i]['predictions'])
                diagnostic.update({'fold':i,'scaler_sha256':split['scaler_sha256'],
                    'scaled_predictions_sha256':sha(body),'evaluation_sha256':sha(evaluation_bytes),
                    'rust_source_sha256':sha(code),'rust_executable_sha256':sha(read(binary,16*1024**2)),
                    'input_sha256':sha(raw),'program_sha256':candidate['program_sha256'],
                    'evaluation_key':key,'context':asdict(context),'input_descriptor':descriptor,
                    'worker':self.config['worker'],'emitter':self.config['emitter'],'rustc':self.config['rustc'],
                    'source_commit':self.config['source_commit'],'commands':commands,
                    'emitter_manifest_json_utf8':manifest_bytes.decode(),
                    'observed_utc_time_ns':time.time_ns(),'observed_monotonic_ns':time.monotonic_ns()})
                create(dest/'parity-rejection.json',canonical(diagnostic))
                raise ParityRejected(diagnostic) from error
            result.update({'fold':i,'scaler_sha256':split['scaler_sha256'],'scaled_predictions_sha256':sha(body)})
            checks.append(result)
        for role in ('worker','emitter','rustc'): bound(self.config[role],64*1024**2)
        again=self.inputs.describe(descriptor['context'],phase)
        need(again[0]==context and again[1]==raw,'fresh original source unchanged')
        bodies = {'input':raw,'program':program.encode(),'evaluation':evaluation_bytes,
                  'rust_source':code,'rust_executable':read(binary,16*1024**2)}
        provenance = {'schema':'jope.flywheel-native-execution-provenance/v1','evaluator_sha256':context.evaluator_sha256,
            'input_sha256':sha(raw),'program_sha256':candidate['program_sha256'],'evaluation_sha256':sha(evaluation_bytes),
            'source_commit':self.config['source_commit'],'source_build_plan_sha256':self.config['source_build_plan_sha256'],
            'execution_timestamp':int(time.time()),'worker':self.config['worker'],'emitter':self.config['emitter'],
            'rustc':self.config['rustc'],'canonicalizer':self.config['native_canonicalizer'],'commands':commands,
            'input_descriptor':descriptor,'emitter_manifest_json_utf8':manifest_bytes.decode(),
            'emitted_predictions_csv_utf8':predictions,'fold_checks':checks,'plan_sha256':self.plan,
            'scope':'Original S1 context and input identity; only training rows evaluated; outer targets zeroed without decoding. Research only.'}
        parity = {'schema':'jope.flywheel-rust-parity/v1','passed':True,'executed':True,'folds_verified':2,
                  'evaluation_key':key,'program_sha256':candidate['program_sha256'],'evaluator_sha256':context.evaluator_sha256,
                  'native_provenance':provenance}
        parity.update({role+'_sha256':sha(bodies[role]) for role in ('input','evaluation','rust_source','rust_executable')})
        bodies['rust_parity'] = canonical(parity)
        bodies['score_records'] = build_records(context,candidate['program_sha256'],key,
            [f['metrics']['rmse'] for f in evaluation['inner_cv']],sha(bodies['rust_parity']),self.config['source_commit'],
            provenance['execution_timestamp'],self.native.canonical_json)
        entry = local_entry(context,program,bodies)
        verify_bodies(context,program,entry,bodies,self.native.canonical_json)
        artifacts = {}
        for role,body in bodies.items(): artifacts[role] = create(dest/('artifact-'+role),body)
        tick(self.deadline)
        return create(dest/'result.json',canonical({'schema':'jope.flywheel-worker-result/v1','state':'executed',
            'context':asdict(context),'program':program,'artifacts':artifacts,'teaching_eligible':False}))
