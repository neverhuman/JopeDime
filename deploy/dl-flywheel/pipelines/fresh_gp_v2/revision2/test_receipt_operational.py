import io,sys,tempfile,time,unittest
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path[:0]=[str(H),str(H/'runtime')]
import numpy as np
from common import *
from jope_flywheel.program_io import sha
from receipt_gate import *
from staged_inputs import VersionCache

class Cases(unittest.TestCase):
    def fixture(self):
        ident='a'*64;part='validation';rawsha='b'*64;metasha='c'*64
        def ref(name,body):return {'bucket':BUCKET,'key':PREFIX+'datasets/flywheel/v1/'+ident+'/'+name,
            'version_id':'exact1','sha256':sha(body),'bytes':len(body)}
        files={n:ref(n,n.encode())for n in ('encoder.jtt1.gz','metadata.json','generation.json')}
        s3=dict(files);s3['receipt']=ref('prior.json',b'prior')
        meta={'content_sha256':rawsha,'target_sha256':'d'*64,'source_target_sha256':'e'*64,'parent_lineage_sha256':'f'*64,
              'partition':part,'rows':80,'features':3,'inner_counts':[40,40],'outer_test_rows':0}
        proof={'source_sha256':WORKER,'producer_phase_sealed':True,'original_coordinator_state':'complete',
               'finalization_custody':None,'manifest':{'sha256':'1'*64},'coordinator_ledger':{'event_count':9},
               'phase_ledgers':{'ofm':{'event_count':67}}}
        arr=io.BytesIO();np.save(arr,np.ones(2051,dtype='<f4'),allow_pickle=False);vector=arr.getvalue();vr=ref('encodings/kumo_tabular_s.npy',vector)
        qb={'passed':True,'tests':{str(i):True for i in range(6)}};qref={'path':'q.json','sha256':sha(canonical(qb)),'bytes':len(canonical(qb))}
        runtime={'sha256':'2'*64,'path':'/qualified/runtime.json'}
        record={'variant_identity':ident,'component':'kumo_tabular_s','dimension':2051,'phase':'ofm','manifest_sha256':'1'*64,
                'host':'xbabe3','input_raw_sha256':rawsha,'target_sha256':meta['target_sha256'],'partition':part,
                'parent_lineage_sha256':meta['parent_lineage_sha256'],'outer_test_rows':0,'teaching_eligible':False,
                'source_input':files['encoder.jtt1.gz'],'source_metadata':files['metadata.json'],
                'producer':{'weight':{'repo':'nvidia/Kumo-Tabular','revision':'3c3e10bbdb590ace29e7026847f92db3c603096d',
                                     'file':'small/regressor.pt','sha256':WEIGHT},'runtime_inventory':runtime},
                'qualification':qref,'output':vr}
        rr={'sha256':sha(canonical(record)),'bytes':len(canonical(record))}
        receipt={'schema':SCHEMA,**meta,'variant_identity':ident,'sequence':938,'metadata_sha256':metasha,'host':'xbabe3',
                 'required_encoders':['kumo_tabular_s'],'completed_encoders':['kumo_tabular_s'],'missing_encoders':[],
                 'roster_version':'kumo-s-only-v1','teaching_eligible':False,'lift_eligible':False,'new_gp_labels':0,
                 'native_context_admission':'unqualified','prepared':True,'raw_units':False,'rights':'research',
                 'publisher_sha256':'3'*64,'source_proof':proof,'prior_dataset_publication':{'receipt':s3['receipt'],'objects':files},
                 'objects':[vr],'encoder_records':{'kumo_tabular_s':{'record':record,'record_ref':rr,
                        'vector_ref':vr,'qualification':{'ref':qref,'body':qb}}}}
        row={'identity':ident,'sequence':938,'content_sha':rawsha,'metadata_sha':metasha,'parent':meta['parent_lineage_sha256'],
             'files':files,'s3':s3,'kumo':{'receipt':ref('kumo.json',canonical(receipt)+b'\n'),'publisher_sha256':'3'*64,
                'source_proof_sha256':digest(proof),'manifest_sha256':'1'*64,'runtime_inventory_sha256':digest(runtime),'vector':vr}}
        pub={'schema':'jope.flywheel-evergreen-dataset-publication/v1','variant_identity':ident,'content_sha256':rawsha,
             'metadata_sha256':metasha,'parent_lineage_sha256':meta['parent_lineage_sha256'],'teaching_eligible':False,'new_gp_labels':0,
             'publication_helper_sha256':GENERATOR_PUBLISHER,'pipeline_config_sha256':PIPELINE,'required_encoders':['kumo_tabular_s'],
             'qualification':{'partition':part,'deterministic_replay_exact':True,'target_sha256':meta['target_sha256'],'outer_test_rows':0}}
        config={'required_encoders':['kumo_tabular_s'],'generator_config':{'sha256':'4'*64}}
        class Cache:
            def get(self,ref,identity,name):return vector if name.endswith('.npy') else canonical(receipt)+b'\n'
        return config,row,meta,{'config_sha256':'4'*64},pub,Cache(),time.time()+60,receipt

    def test_real_file_exact_version_receipt_and_vector_reverified(self):
        f=self.fixture();config,row,meta,generation,pub,unused,deadline,receipt=f
        row['kumo']['receipt']['key']=PREFIX+'receipts/flywheel-datasets/v1/'+row['identity']+'/'+row['kumo']['receipt']['sha256']+'.json'
        with tempfile.TemporaryDirectory()as td:
            root=Path(td).resolve();vector=unused.get(row['kumo']['vector'],row['identity'],'x.npy');bodies={row['kumo']['receipt']['key']:canonical(receipt)+b'\n',row['kumo']['vector']['key']:vector}
            paths={}
            for n,(key,body)in enumerate(bodies.items()):p=root/str(n);p.write_bytes(body);paths[key]=p
            class Client:
                def get_object(self,**kw):
                    b=paths[kw['Key']].read_bytes();return {'VersionId':kw['VersionId'],'ContentLength':len(b),'Body':io.BytesIO(b)}
            cache=VersionCache(root/'cache',Client(),deadline)
            result=validate_current_receipts(config,row,meta,generation,pub,cache,deadline)
            self.assertEqual(result['partition'],'validation');self.assertFalse(result['training_admission'])
            next(cache.root.glob('*.bytes')).write_bytes(b'tampered')
            with self.assertRaises(ValueError):validate_current_receipts(config,row,meta,generation,pub,cache,deadline)
    def test_real_receipt_unqualified_role_and_target_are_refused(self):
        for key in ('target_sha256','partition'):
            f=self.fixture();f[-1][key]='wrong'
            with self.assertRaises(ValueError):validate_current_receipts(*f[:-1])
