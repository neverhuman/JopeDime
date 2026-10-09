"""Create-only measured-program library, with exact-version content verification.

This is a new versioned wrapper around native S1 evaluation receipts. It does not
rewrite GP-HWM winners or pretend ordinary evaluated losers are HWM selections.
"""
import hashlib,io,json,time
from .protocol import canonical,sha256
from .program_gate import MAX_BYTES,ARTIFACTS,ARTIFACT_LIMITS,verified_measurement

BUCKET='veox-jopedime-learning-use2'
PREFIX='experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/'

def need(ok,msg):
    if not ok:raise ValueError(msg)
def code(exc):return getattr(exc,'response',{}).get('Error',{}).get('Code')

class S3Library:
    def __init__(self,client,*,deadline,nativecanonical_json=None):
        self.client=client;self.deadline=deadline;self.nativecanonical_json=nativecanonical_json
    def tick(self):
        if time.time()>=self.deadline:raise TimeoutError('program library operation deadline')
    def get_version(self,ref):
        self.tick();need(ref['bucket']==BUCKET and ref['key'].startswith(PREFIX),'library namespace escape')
        need(type(ref['bytes'])is int and 0<ref['bytes']<=MAX_BYTES,'object byte cap')
        sha256(ref['sha256']);version=ref['version_id'];need(type(version)is str and version not in ('','null'),'VersionId required')
        r=self.client.get_object(Bucket=BUCKET,Key=ref['key'],VersionId=version)
        need(r.get('VersionId')==version and r.get('ContentLength')==ref['bytes'],'exact version/size mismatch')
        out=bytearray()
        with r['Body']as stream:
            while True:
                self.tick();b=stream.read(min(1024**2,ref['bytes']-len(out)+1))
                if not b:break
                out.extend(b);need(len(out)<=ref['bytes'],'object exceeded expected bytes')
        self.tick();body=bytes(out)
        need(len(body)==ref['bytes'] and hashlib.sha256(body).hexdigest()==ref['sha256'],'exact version SHA mismatch')
        return body,version
    def lookup(self,evaluation_key):
        sha256(evaluation_key);self.tick();key=PREFIX+evaluation_key+'/record.json'
        try:r=self.client.head_object(Bucket=BUCKET,Key=key)
        except Exception as exc:
            self.tick()
            if code(exc)in('404','NoSuchKey','NotFound'):return None
            raise
        ref={'bucket':BUCKET,'key':key,'version_id':r.get('VersionId'),'sha256':r.get('Metadata',{}).get('sha256'),'bytes':r['ContentLength']}
        need(ref['bytes']<=2*1024**2,'result wrapper too large');body,_=self.get_version(ref);return json.loads(body)
    def put(self,key,body):
        self.tick();need(key.startswith(PREFIX) and type(body)is bytes and 0<len(body)<=MAX_BYTES,'bounded own object required')
        digest=hashlib.sha256(body).hexdigest()
        try:
            r=self.client.put_object(Bucket=BUCKET,Key=key,Body=body,ContentLength=len(body),IfNoneMatch='*',Metadata={'sha256':digest})
        except Exception as exc:
            if code(exc)not in('PreconditionFailed','ConditionalRequestConflict','412'):raise
            self.tick();r=self.client.head_object(Bucket=BUCKET,Key=key)
        ref={'bucket':BUCKET,'key':key,'version_id':r.get('VersionId'),'sha256':digest,'bytes':len(body)}
        self.get_version(ref);return ref
    def put_artifact(self,role,body):
        need(role in ARTIFACTS,'unknown program artifact role')
        need(type(body)is bytes and 0<len(body)<=ARTIFACT_LIMITS[role],'role-specific artifact bound')
        digest=hashlib.sha256(body).hexdigest();return self.put(PREFIX+'objects/'+digest+'/'+role,body)
    def commit(self,context,program,entry):
        # Publication of the lookup entry is last, after full artifact verification.
        measured=verified_measurement(self,context,program,entry);self.tick()
        ref=self.put(PREFIX+measured['evaluation_key']+'/record.json',canonical(entry))
        self.tick();return {'record':ref,'measurement':measured}

def client(profile='veox-jepson-prod'):
    import boto3
    from botocore.config import Config
    return boto3.Session(profile_name=profile).client('s3',config=Config(connect_timeout=5,read_timeout=20,retries={'max_attempts':0},max_pool_connections=4))
