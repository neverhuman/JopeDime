import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('report',Path(__file__).with_name('inference_report.py'))
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

class Reporting(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.plan={'schema':'jope.inference-benchmark/v2','experiment':{'methods_sha256':'m','source_sha':'s'},'binary':{'sha256':'b'},'inputs':[{'id':str(i),'raw':{'sha256':'r'}} for i in range(10)]}
    def tearDown(self):self.temp.cleanup()
    def test_missing_never_claims_speed(self):
        summary,_=report.summarize(self.plan,self.root)
        self.assertFalse(summary['publication_headline'])
        self.assertEqual(sum(r['mode']=='warm' for r in summary['missing']),100)
        self.assertNotIn('speedup',summary)
    def test_controller_failure_before_qualification_is_explicit(self):
        (self.root/'launch.json').write_text(json.dumps({'exit_code':1,'reason':'bounded service failed'}))
        summary,_=report.summarize(self.plan,self.root)
        self.assertEqual(summary['controller']['status'],'failed')
        self.assertEqual(summary['controller']['qualification_status'],'not_executed_or_interrupted')
        self.assertEqual(len(summary['prediction']),90)
        self.assertFalse(summary['publication_headline'])
    def test_failed_and_wrong_device_remain_failures(self):
        d=self.root/'warm-0-0';d.mkdir()
        (d/'receipt.json').write_text(json.dumps({'schema':self.plan['schema'],'dataset':'0','seed':0,'mode':'warm','success':True,'elapsed_seconds':1,'devices':['Cpu']*3}))
        summary,_=report.summarize(self.plan,self.root)
        self.assertEqual(summary['completed_successfully'],0)
        self.assertGreaterEqual(len(summary['failures']),1)
    def test_qualification_flag_without_evidence_refused(self):
        (self.root/'qualification.json').write_text(json.dumps({'passed':True,'cases':[{'pass':True}]*8}))
        summary,_=report.summarize(self.plan,self.root)
        self.assertFalse(summary['qualified'])

class ArtifactValidation(unittest.TestCase):
    def test_exact_parity_requires_matching_bundle(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            bundle={'manifest':{'backend':'python'},'files':{'model.py':'content'}}
            identity=hashlib.sha256(json.dumps(bundle,separators=(',',':')).encode()).hexdigest()
            values={'kind':'values','data':[1.0]}
            pins={}
            for name,value in [('bundle',bundle),('qualification_plan',{'expected':values,'backend':'python','bundle_sha256':identity}),('prediction',values)]:
                p=Path(root)/name;p.write_text(json.dumps(value));pins[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertTrue(report.parity_verified(pins))
            p=Path(pins['prediction']['path']);p.write_text(json.dumps({'kind':'values','data':[2.0]}));pins['prediction']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertFalse(report.parity_verified(pins))
    def test_corruption_is_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'asset';p.write_bytes(b'original')
            pin={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertTrue(report.verified(pin));p.write_bytes(b'corrupted')
            self.assertFalse(report.verified(pin))
    def test_parity_rejects_wrong_selected_prediction(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            pins={}
            for name,value in [('qualification_plan',{'expected':{'kind':'values','data':[1.0]}}),('prediction',{'kind':'values','data':[2.0]})]:
                p=Path(root)/name;p.write_text(json.dumps(value));pins[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertFalse(report.parity_verified(pins))
    def test_relative_tolerance_does_not_pass_exact_bit_policy(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            pins={}
            for name,value in [('qualification_plan',{'expected':{'kind':'values','data':[0.0]}}),('prediction',{'kind':'values','data':[-0.0]})]:
                p=Path(root)/name;p.write_text(json.dumps(value));pins[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertFalse(report.parity_verified(pins))

if __name__=='__main__':unittest.main()
