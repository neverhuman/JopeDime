#!/usr/bin/env python3
"""Build the monograph from maintained chapters and retained numeric evidence.
Reuses the manual renderer's GitHub slug policy and WeasyPrint stack.
No network, model, fitting, training or benchmark execution occurs here.
"""
import argparse,csv,hashlib,importlib.util,json,math,re,shutil,subprocess,sys
from pathlib import Path
from urllib.parse import urlparse,unquote
from datetime import datetime,timezone
import markdown
from bs4 import BeautifulSoup
from weasyprint import HTML,CSS,default_url_fetcher
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];MANUAL=ROOT.parent/'asm-manual'
spec=importlib.util.spec_from_file_location('manual_renderer',MANUAL/'tools/build_manual.py');manual=importlib.util.module_from_spec(spec);spec.loader.exec_module(manual)
INPUTS={}
def pin(p):
    p=Path(p).resolve();INPUTS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p

def figure(out,name,labels):
    fig,ax=plt.subplots(figsize=(9,2.3));ax.axis('off')
    for i,label in enumerate(labels):
        x=(i+.5)/len(labels);ax.text(x,.5,label,ha='center',va='center',fontsize=9,bbox={'boxstyle':'round,pad=.5','facecolor':'#e6f1f5','edgecolor':'#316878'},transform=ax.transAxes)
        if i:ax.annotate('',xy=(x-.07,.5),xytext=(x-1/len(labels)+.07,.5),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#316878'})
    fig.savefig(out/name,bbox_inches='tight');plt.close(fig)

def plots(out):
    figure(out,'architecture.svg',['Fitting\nrows only','GPU encoders\n2,048 context','Policy\nprograms','Bounded GP\nOOF selection','Fitted state\nsource + parity'])
    figure(out,'context.svg',['Hyperion\nrow 128 → pool 512','Foundation + labels\n2,048 → 1,024','Target\n512','Concat + clip ±32\n2,048'])
    figure(out,'lowering.svg',['Typed recipe\noriginal selected','Fit preprocessing\n+ model state','Deterministic\nbackend templates','Compile / import\npublic API','Compare ordered\nnative predictions'])
    trace=json.loads(pin(ROOT/'data/candidate-traces.json').read_text())
    winner=next(t for t in trace if t['winner']['candidate']['origin']=='crossover')['winner']
    assert '(row_diff_stats append=true)' in winner['candidate']['genome'] and '(histgbm rounds=180)' in winner['candidate']['genome']
    fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
    nodes=[(.2,.9,'Measured parent 1'),(.8,.9,'Measured parent 2'),(.5,.58,'Crossover offspring: pipe'),(.08,.12,'X'),(.28,.12,'row_diff_stats'),(.48,.12,'winsor_clip'),(.68,.12,'winsor_moments'),(.9,.12,'histgbm\n180 rounds')]
    for x,y,label in nodes:ax.text(x,y,label,ha='center',va='center',fontsize=8,bbox={'boxstyle':'round,pad=.4','facecolor':'#e6f1f5','edgecolor':'#316878'},transform=ax.transAxes)
    for x in (.2,.8):ax.annotate('',xy=(.5,.65),xytext=(x,.84),xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
    for x in (.08,.28,.48,.68,.9):ax.plot([.5,x],[.51,.23],color='#316878',lw=.8,transform=ax.transAxes)
    fig.savefig(out/'gp-tree.svg',bbox_inches='tight');plt.close(fig)
    rows=list(csv.DictReader(pin(MANUAL/'data/per-step-history/a100-step-bins.csv').open()))
    fig,ax=plt.subplots(figsize=(9,3.3))
    for cycle in sorted({r['cycle'] for r in rows},key=int):
        selected=[r for r in rows if r['cycle']==cycle and r['metric']=='loss'];ax.plot([int(r['last_step']) for r in selected],[float(r['mean']) for r in selected],lw=.9)
    ax.axvline(60000,color='black',ls='--',lw=.8,label='Accepted step 60,000');ax.set(xlabel='Historical optimizer step (100-step bins; separate blocks)',ylabel='Recorded composite training loss');ax.legend();fig.tight_layout();fig.savefig(out/'training.svg');plt.close(fig)
    gp=list(csv.DictReader(pin(ROOT/'data/gp-incumbents.csv').open()))
    fig,ax=plt.subplots(figsize=(9,3.3))
    for arm in sorted({r['arm'] for r in gp}):
        rows0=[r for r in gp if r['arm']==arm and r['best_loss']];ax.step([int(r['attempt']) for r in rows0],[float(r['best_loss']) for r in rows0],where='post',label=arm[:8])
    ax.set(xlabel='Attempt inventory (historical source68f56404)',ylabel='Best measured mean OOF RMSE');ax.legend(title='Arm identity');fig.tight_layout();fig.savefig(out/'gp-incumbents.svg');plt.close(fig)
    rows=list(csv.DictReader(pin(MANUAL/'data/deep-stack/a100-throughput-bins.csv').open()))
    fig,ax=plt.subplots(figsize=(9,3.3))
    for cycle in sorted({r['cycle'] for r in rows},key=int):
        selected=[r for r in rows if r['cycle']==cycle];ax.plot([int(r['last_step']) for r in selected],[float(r['examples_per_second']) for r in selected],lw=.8)
    ax.set(xlabel='Historical optimizer step (separate corpus blocks)',ylabel='Reported examples / second');fig.tight_layout();fig.savefig(out/'throughput.svg');plt.close(fig)

def fetch(url,*args,**kwargs):
    parsed=urlparse(url)
    if parsed.scheme not in ('file','data'):raise ValueError('remote rendering resource refused')
    return default_url_fetcher(url,*args,**kwargs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--study',type=Path);ap.add_argument('--cutoff',required=True);args=ap.parse_args()
    pin(Path(__file__))
    args.output.mkdir(parents=True,exist_ok=True);assets=args.output/'core-ip-assets';assets.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'core-ip-v1','axes.spines.top':False,'axes.spines.right':False})
    plots(assets)
    dataout=assets/'data';dataout.mkdir(exist_ok=True)
    for p in sorted((ROOT/'data').glob('*')):
        if p.is_file():shutil.copy2(pin(p),dataout/p.name)
    examples=assets/'examples';examples.mkdir(exist_ok=True)
    for p in sorted((ROOT/'examples').glob('*')):shutil.copy2(pin(p),examples/p.name)
    # Historical figures retain the original numeric inputs and generation receipts.
    for p in sorted((MANUAL/'data').rglob('*')):
        if p.is_file() and p.suffix in ('.csv','.json','.jsonl','.gz'):pin(p)
    for p in sorted((MANUAL/'tools').glob('*.py')):pin(p)
    shutil.copy2(pin(ROOT/'data/jailgun-audit.md'),assets/'jailgun-audit.md')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    tensors=json.loads(pin(ROOT/'data/tensors.json').read_text());catalog=json.loads(pin(ROOT/'data/catalog.json').read_text())
    table='| Tensor | Shape | Dtype | Parameters |\n|---|---|---|---:|\n'+'\n'.join('| `'+t['name']+'` | '+str(t['shape'])+' | '+t['dtype']+' | '+f"{t['parameters']:,}"+' |' for t in tensors['tensors'])
    study=json.loads(pin(args.study).read_text()) if args.study else None
    if study:
        q=study.get('qualification',{});observations=f"**Controller: {study.get('controller',{}).get('status','unknown')}. GPU qualification: {study.get('controller',{}).get('qualification_status','unknown')}. Completed measured warm paths: {study['completed_successfully']}/100.**\n\n"
        if q.get('failure'):observations+='Retained failure: `'+q['failure']+'`.\n\n'
        if q.get('cases'):
            observations+='| Case | Largest absolute error | Pass |\n|---|---:|---|\n'+'\n'.join('| '+c['case']+' | '+f"{max(c['errors'].values()):.8g}"+' | '+str(c['pass'])+' |' for c in q['cases'])+'\n\n'
        observations+='| Planned path | Recorded observations |\n|---|---:|\n| Warm requests | '+str(100-len([r for r in study['missing'] if r['mode']=='warm']))+' |\n| Fresh-process requests | '+str(len(study['cold_observations']))+' |\n| Full-GP arm receipts | '+str(sum(r.get('status')!='unexecuted' for r in study['full_gp']))+' |\n| Prediction batch receipts | '+str(sum(r.get('status') not in ('unexecuted','unsupported') for r in study['prediction']))+' |\n\n'
        observations+='No complete-path latency is inferred from successful fragments. See the retained summary and CSV for every planned attempt and failure.\n\n'
        assert study.get('window')==json.loads(pin(ROOT/'data/gpu-closeout.json').read_text())['window'], 'study window changed'
        if study['completed_successfully'] or any(r.get('accepted') for r in study['prediction']):raise ValueError('successful study requires fresh measured-result tables; failure-only table refused')
        evidence=json.loads(pin(ROOT/'data/gpu-study-manifest.json').read_text())
        observations+='Hardware preflight: '+evidence['hardware']['gpu'].strip()+'. '+evidence['hardware']['cpu_model'].split(':',1)[1].strip()+'. '+evidence['hardware']['rustc'].strip()+'. Build source `'+evidence['source_sha']+'`; binary SHA256 `'+evidence['binary']['sha256']+'`. Hardware presence is not CUDA qualification.\n\n'
        observations+='| Dataset | Fitting rows | Measured warm | Cold | GP arms | Prediction batches |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join('| '+d['dataset']+' | '+str(d['rows'])+' | 0 / 10 | 0 / 1 | '+('0 / 2' if d['gp'] else 'Not scheduled')+' | 0 / 9 |' for d in evidence['inputs'])+'\n\n'

        if study.get('window'):observations+='Original window: `'+json.dumps(study['window'],sort_keys=True)+'`.\n'
    else:observations='**Study not yet executed. This is a publication draft, not measurement evidence.**'
    traces=json.loads(pin(ROOT/'data/candidate-traces.json').read_text());tr=next(t for t in traces if t['winner']['candidate']['origin']=='crossover')
    w=tr['winner'];trace='Retained source `'+tr['source_sha']+'`, receipt SHA256 `'+tr['receipt_sha256']+'`. The neural-arm winner is generation1 **crossover**, with training-side OOF loss **'+str(w['feedback']['loss'])+'**. Its parents below are actual retained observations, not a schematic reconstruction.\n\n| Subject | Origin | Mean OOF loss | Program |\n|---|---|---:|---|\n'
    for label,o in [('Parent '+str(i+1),p) for i,p in enumerate(tr['parents'])]+[('Winner',w)]:trace+='| '+label+' | '+o['candidate']['origin']+' | '+str(o['feedback']['loss'])+' | `'+o['candidate']['genome']+'` |\n'
    trace+='\nThe winner records five physical fit starts and fitted OOF bundle SHA256 `'+w['feedback']['fitted_artifact_sha256']+'`. Each of five fold artifacts has its own retained hash in `candidate-traces.json`. This is fitted-state identity evidence; the monograph does not reproduce the private tree coefficients. The winning neural-arm lineage comes from exploration, so it is not evidence that neural proposals caused the improvement. No new held score was opened for this trace.\n'
    support=json.loads(pin(ROOT/'data/export-support.json').read_text())['fitted_inference']
    assert all(r['standalone_python']['supported']==r['standalone_rust']['supported'] for r in support)
    support_table='| Task | Method | Python/Rust | C++17/20 | JavaScript |\n|---|---|---|---|---|\n'+'\n'.join('| '+r['task']+' | '+r['method']+' | '+str(r['standalone_python']['supported'])+' | '+str(r['cpp17_cpp20']['supported'])+' | '+str(r['javascript']['supported'])+' |' for r in support)
    chunks=[];contents=[];headings=[];figures=[]
    for chapter,p in enumerate(sorted((ROOT/'sections').glob('*.md')),1):
        raw=pin(p).read_text().replace('TENSOR_TABLE',table).replace('STUDY_RESULTS',observations).replace('CANDIDATE_TRACE',trace).replace('EXPORT_TABLE',support_table)
        if chapter==2:
            raw+='\n## Refreshed catalog metadata\n\nRead at '+catalog['observed_utc']+'. Catalog SHA256 `'+catalog['sha256']+'`; '+str(catalog['rows'])+' entries, '+str(catalog['unique_ids'])+' unique dataset IDs.\n\n| Task label | Entries |\n|---|---:|\n'+'\n'.join('| '+k+' | '+str(v)+' |' for k,v in catalog['tasks'].items())+'\n\nThese are catalog task labels, not a statement of native trainer coverage. In particular `multi_label` is not automatically the product’s multiclass task.\n'
        # Freeze every included image and normalize export paths for both formats.
        def picture(m):
            caption,path=m.groups();path=path.split('#')[0]
            if path.startswith('../figures/'):
                target=assets/Path(path).name
            else:
                src=(p.parent/path).resolve();pin(src);target=assets/(hashlib.sha256(str(src).encode()).hexdigest()[:8]+'-'+src.name);shutil.copy2(src,target)
            if not target.exists():raise ValueError('missing figure '+str(target))
            number=len(figures)+1;figures.append({'number':number,'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
            return f'![Figure {number}. {caption}](core-ip-assets/{target.name})'
        raw=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',picture,raw)
        # Source references are pinned, with valid line anchors when provided.
        def source_link(m):
            path,line=m.groups();src=REPO/path
            if not src.is_file():return m.group(0)
            pin(src);return f'[{path}'+(':'+line if line else '')+f'](https://github.com/veox-ai/JopeDime/blob/{source}/{path}'+('#L'+line if line else '')+')'
        raw=re.sub(r'`((?:crates|docs|deploy|python)/[^`\s:]+)(?::(\d+)(?:[-–]\d+)?)?`',source_link,raw)
        counts=[chapter,0,0,0];seen={};in_fence=False;lines=[];last=1
        for line in raw.splitlines():
            if line.startswith(('```','~~~')):in_fence=not in_fence
            m=None if in_fence else re.match(r'^(#{1,4}) (.*)$',line)
            if m:
                level=len(m[1]);title=m[2]
                if level>last+1:raise ValueError(f'{p}: heading skips {last} to {level}: {title}')
                last=level
                if level>1:counts[level-1]+=1
                for k in range(level,4):counts[k]=0
                label='.'.join(str(c) for c in counts[:level]);anchor='c'+str(chapter)+'-'+manual.github_slug(title,seen)
                lines+=['<a id="'+anchor+'"></a>',m[1]+' '+label+' '+title]
                headings.append({'number':label,'title':title,'level':level,'anchor':anchor})
                if level<=2:contents.append('  '*(level-1)+f'- [{label} {title}](#{anchor})')
            else:lines.append(line)
        chunks.append('\n'.join(lines))
    contents.append('- [Appendix A. Exact tensor inventory](#appendix-a)')
    chunks.append('<a id="appendix-a"></a>\n# Appendix A. Exact tensor inventory\n\nThis appendix is additional to the twelve-chapter monograph. Refreshed accepted F32 safetensors inventory; shapes and counts are read from the pinned asset.\n\n'+table)
    front=f'''# Core IP Technical Reference

Technical monograph and GPU pipeline study · Evidence cutoff **{args.cutoff}**.

Publication source: `{source}`. Accepted weights SHA256: `{tensors['sha256']}`. Model inventory: **{tensors['tensor_count']} tensors; {tensors['parameters']:,} parameters**. Historical measurements retain their original source/date identities. Current benchmark source and execution identities are in the retained study plan; later publication edits do not relabel the measurement build.

## Contents

'''+ '\n'.join(contents)+'\n\n'
    md=front+'\n\n'.join(chunks)+'\n';mdpath=args.output/'CORE_IP_TECHNICAL_REFERENCE.md';mdpath.write_text(md)
    html=markdown.markdown(md,extensions=['tables','fenced_code','sane_lists']);soup=BeautifulSoup(html,'html.parser')
    ids={x['id'] for x in soup.find_all(id=True)}
    for a in soup.find_all('a',href=True):
        if a['href'].startswith('#') and a['href'][1:] not in ids:raise ValueError('broken internal link '+a['href'])
        target=urlparse(a['href'])
        if not target.scheme and target.path and not (args.output/unquote(target.path)).exists():raise ValueError('broken local link '+a['href'])
    for img in soup.find_all('img'):
        fig=soup.new_tag('figure');img.wrap(fig);caption=soup.new_tag('figcaption');caption.string=img.get('alt','');fig.append(caption)
    html='<!doctype html><html><head><meta charset="utf-8"><title>Core IP Technical Reference</title></head><body>'+str(soup)+'</body></html>'
    css='''@page {size:A4; margin:17mm 17mm 18mm; @bottom-center {content:counter(page);font:9pt sans-serif;} @top-right {content:"JopeDime · Core IP Technical Reference";font:8pt sans-serif;color:#566;}} body{font:9.8pt/1.28 "DejaVu Sans",sans-serif;color:#202b33;} h1{font-size:21pt;break-before:page;color:#17485c;} h1:first-child{break-before:auto;} h2{font-size:14pt;color:#245467;margin-top:18pt;} h3{font-size:11.5pt;} h4{font-size:10.5pt;} h1,h2,h3,h4{break-after:avoid;} p{text-align:left;margin:6pt 0;} a{color:#145b78;text-decoration:none;overflow-wrap:anywhere;} pre{font:7.1pt/1.25 "DejaVu Sans Mono";white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f5;padding:8pt;} code{font-family:"DejaVu Sans Mono";font-size:8pt;overflow-wrap:anywhere;} table{border-collapse:collapse;width:100%;font-size:8.2pt;margin:9pt 0;table-layout:auto;} th{background:#e7eef1;} th,td{border:0.4pt solid #b8c9d1;padding:4pt;vertical-align:top;overflow-wrap:anywhere;} th:first-child,td:first-child{min-width:48pt;} tr{break-inside:avoid;} thead{display:table-header-group;} figure{margin:12pt 0;break-inside:avoid;} img{max-width:100%;max-height:185mm;} figcaption{font-size:8pt;color:#475d68;} blockquote{margin:10pt 0;border-left:3pt solid #789;padding:6pt 10pt;background:#f3f5f7;} .equation{font-size:11pt;padding:9pt;background:#f0f4f6;break-inside:avoid}.equation span{float:right} li{margin:3pt 0;}'''
    (args.output/'CORE_IP_TECHNICAL_REFERENCE.html').write_text(html)
    doc=HTML(string=html,base_url=args.output.as_uri()+'/',url_fetcher=fetch).render(stylesheets=[CSS(string=css)])
    pdf=args.output/'CORE_IP_TECHNICAL_REFERENCE.pdf';doc.write_pdf(pdf)
    # Keep all referenced numeric evidence beside the exported monograph.
    dataout=assets/'data';dataout.mkdir(exist_ok=True)
    reproduction=assets/'reproduction';reproduction.mkdir(exist_ok=True)
    for path in INPUTS:
        p=Path(path)
        if p.suffix=='.py':shutil.copy2(p,reproduction/(hashlib.sha256(str(p).encode()).hexdigest()[:8]+'-'+p.name))
        if p.suffix in ('.csv','.json','.jsonl','.gz'):
            shutil.copy2(p,dataout/(hashlib.sha256(str(p).encode()).hexdigest()[:8]+'-'+p.name))
    manifest={'schema':'jope.core-ip-publication/v1','evidence_cutoff_utc':args.cutoff,'publication_source':source,'inputs':[{'path':p,'sha256':h} for p,h in sorted(INPUTS.items())],'outputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [mdpath,pdf]],'pages':len(doc.pages),'monograph_pages':next(i for i,page in enumerate(doc.pages) if 'appendix-a' in page.anchors),'headings':headings,'figures':figures,'markdown_pdf_same_content':True,'assets':[{'path':str(p.relative_to(args.output)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(assets.rglob('*')) if p.is_file()],'study':str(args.study) if args.study else None,'visual_inspection':'pending'}
    (args.output/'CORE_IP_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'pages':len(doc.pages),'monograph_pages':next(i for i,page in enumerate(doc.pages) if 'appendix-a' in page.anchors),'chapters':12,'figures':len(figures),'words':len(md.split()),'pdf':str(pdf)}))
if __name__=='__main__':main()
