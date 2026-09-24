#!/usr/bin/env python3
"""Re-extract original metrics using frozen CSV row-selection metadata; never overwrite sources."""
import argparse,csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'data/training-history'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 assert not args.output.exists(),'output must be a new custody directory'
 verification=json.loads((ROOT/'verification.json').read_text());cache={};sources={};outputs=[];numeric=lineage=0
 extracted=[]
 for artifact in verification['artifacts']:
  path=ROOT/artifact['file'];assert sha(path)==artifact['sha256']
  with path.open(newline='') as stream:
   reader=csv.DictReader(stream);fields=reader.fieldnames;rows=[]
   for row in reader:
    source=Path(row['source_path'])
    if source not in cache:
     raw=source.read_bytes();sources[source]=hashlib.sha256(raw).hexdigest()
     cache[source]=[json.loads(line) for line in raw.splitlines()] if source.suffix=='.jsonl' else json.loads(raw)
    assert sources[source]==row['source_sha256']
    obj=cache[source]
    if 'source_json_pointer' in row:
     value=obj[int(row['source_line'])-1] if source.suffix=='.jsonl' else obj
     for key in row['source_json_pointer'].strip('/').split('/'):
      key=key.replace('~1','/').replace('~0','~')
      value=value[int(key)] if isinstance(value,list) else value[key]
     assert math.isfinite(float(value)) and float(row['value'])==value
     row['value']=str(value);numeric+=1
    else:
     for key in ['step','model_sha256','parameter_sha256','optimizer_sha256','corpus_sha256','world','global_batch']:assert str(obj[key])==row[key]
     lineage+=1
    rows.append(row)
  extracted.append((path.name,fields,rows,sha(path)))
 assert (numeric,lineage)==(3218,78)
 for source,digest in sources.items():assert sha(source)==digest
 args.output.mkdir(parents=True)
 for name,fields,rows,template in extracted:
  path=args.output/name
  with path.open('x',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
  outputs.append(dict(file=name,sha256=sha(path),template_sha256=template,rows=len(rows)))
 result=dict(schema='jope.manual-training-reextraction/v1',numeric_values=numeric,lineages=lineage,outputs=outputs,sources={str(p):h for p,h in sources.items()},scope='Original metrics re-extracted using frozen row-selection metadata; no new observations, fitting or training.',fits=0)
 (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(numeric_values=numeric,lineages=lineage,outputs=len(outputs),fits=0)))
if __name__=='__main__':main()
