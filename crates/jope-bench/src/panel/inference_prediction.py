"""Benchmark retained public APIs; initialization/process time remains separately visible."""
import json, subprocess, sys, time, hashlib, struct, math
from pathlib import Path

def run(plan_path, journal, cutoff_ms):
    plan=json.loads(Path(plan_path).read_text()); journal=Path(journal)
    for dataset in plan['inputs']:
        raw=json.loads(Path(dataset['raw']['path']).read_text())
        rows=[r['values'] for r in raw['rows']]
        for backend in ('python','rust','cpp17'):
            source=journal/f"warm-{dataset['id']}-0"/backend
            if not (source/'bundle.json').exists():continue
            direct=json.loads((source.parent/'receipt.json').read_text())
            entry=next((b for b in direct.get('export',{}).get('backends',[]) if b.get('backend')==backend and b.get('status')=='completed'),None)
            if entry is None:continue
            bundle=json.loads((source/'bundle.json').read_text())
            if hashlib.sha256((source/'bundle.json').read_bytes()).hexdigest()!=entry['bundle']['sha256']:raise ValueError('prediction bundle changed')
            qpath=source/'qualification-plan.json'
            expected=json.loads(qpath.read_text())['expected']['data'][0]
            for size in (1,32,1024):
                if time.time()*1000>=cutoff_ms:return
                out=journal/f"prediction-{dataset['id']}-{backend}-{size}"
                if out.exists():continue # interrupted attempt is not silently replaced
                out.mkdir()
                selected=[rows[0] for _ in range(size)]
                for name,text in bundle['files'].items():(out/name).write_text(text)
                receipt={'schema':plan['schema'],'dataset':dataset['id'],'backend':backend,'batch_rows':size,'status':'failed','plan_identity':plan['experiment']['methods_sha256'],'source_sha':plan['experiment']['source_sha'],'binary_sha256':plan['binary']['sha256'],'input_sha256':dataset['raw']['sha256'],'source_qualification':entry['qualification_plan'],'bundle_sha256':hashlib.sha256((source/'bundle.json').read_bytes()).hexdigest()}
                try:
                    commands=[]
                    if backend=='rust':
                        literal='vec!['+','.join('vec!['+','.join('None' if x is None else 'Some(String::from_utf8(vec!['+','.join(str(b) for b in x.encode())+']).unwrap())' for x in row)+']' for row in selected)+']'
                        (out/'latency.rs').write_text('mod model;fn main(){let rows:Vec<Vec<Option<String>>>='+literal+';for _ in 0..10 {let t=std::time::Instant::now();let y=model::predict_raw(&rows).unwrap();let s=t.elapsed().as_secs_f64();println!("{}",s);std::fs::write("predictions.json",y).unwrap();}}')
                        commands=[['rustc','--edition=2021','-O','latency.rs','-o','latency']]; launch=['./latency']
                    else:
                        if backend=='cpp17':commands=[['g++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-shared','-fPIC','model.cpp','-o','model.so']]
                        (out/'rows.json').write_text(json.dumps(selected))
                        module='model' if backend=='python' else 'boundary'
                        (out/'latency.py').write_text(f'import {module},json,time\nrows=json.load(open("rows.json"))\nfor _ in range(10):\n t=time.perf_counter();y={module}.predict_raw(rows);s=time.perf_counter()-t;print(s)\njson.dump(y,open("predictions.json","w"),allow_nan=False)\n')
                        launch=['python3','latency.py']
                    for i,args in enumerate(commands):
                        r=subprocess.run(args,cwd=out,capture_output=True,timeout=min(30,max(.001,(cutoff_ms-time.time()*1000)/1000)))
                        (out/f'compile-{i}.stderr').write_bytes(r.stderr)
                        if r.returncode:raise RuntimeError(f'compiler exit {r.returncode}')
                    started=time.perf_counter();r=subprocess.run(launch,cwd=out,capture_output=True,timeout=min(30,max(.001,(cutoff_ms-time.time()*1000)/1000)));wall=time.perf_counter()-started
                    (out/'stdout').write_bytes(r.stdout);(out/'stderr').write_bytes(r.stderr)
                    if r.returncode:raise RuntimeError(f'predict exit {r.returncode}')
                    samples=[float(x) for x in r.stdout.splitlines()]
                    if len(samples)!=10 or any(not math.isfinite(x) or x<0 for x in samples):raise ValueError('missing timing samples')
                    actual=json.loads((out/'predictions.json').read_text())
                    if actual.get('kind')!='values' or len(actual['data'])!=size or any(struct.pack('>d',x)!=struct.pack('>d',expected) for x in actual['data']):raise ValueError('prediction parity failed')
                    receipt.update(status='completed',parity='exact_bits',prediction={'path':str(out/'predictions.json'),'sha256':hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()},samples_seconds=samples,process_wall_seconds=wall,initialization_and_io_residual_seconds=wall-sum(samples),commands=commands+[launch],note='Public API timing includes return serialization; process residual also includes initialization, parsing, printing and teardown. Repeated first fitting-feature row; no scoring labels.')
                except Exception as exc:receipt['failure']=str(exc)
                (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':run(sys.argv[1],sys.argv[2],int(sys.argv[3]))
