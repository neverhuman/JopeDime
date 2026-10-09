import hashlib,io,time,unittest
from jope_flywheel.s3_library import S3Library,BUCKET,PREFIX

class Error(Exception):
    def __init__(self,code):self.response={'Error':{'Code':code}}
class Fake:
    def __init__(self):self.objects={};self.denied=False
    def put_object(self,**kw):
        assert kw['IfNoneMatch']=='*'
        if kw['Key']in self.objects:raise Error('PreconditionFailed')
        self.objects[kw['Key']]=(kw['Body'],kw['Metadata'],'v1');return {'VersionId':'v1'}
    def head_object(self,**kw):
        if self.denied:raise Error('403')
        if kw['Key']not in self.objects:raise Error('404')
        b,m,v=self.objects[kw['Key']];return {'VersionId':v,'ContentLength':len(b),'Metadata':m}
    def get_object(self,**kw):
        b,m,v=self.objects[kw['Key']];assert kw['VersionId']==v
        return {'VersionId':v,'ContentLength':len(b),'Body':io.BytesIO(b)}
class LibraryTests(unittest.TestCase):
    def setUp(self):self.f=Fake();self.s=S3Library(self.f,deadline=time.time()+60)
    def test_absent_and_auth_failure_differ(self):
        self.assertIsNone(self.s.lookup('a'*64));self.f.denied=True
        with self.assertRaises(Error):self.s.lookup('a'*64)
    def test_create_once_verify_exact_version(self):
        a=self.s.put_artifact('program',b'(ridge)');b=self.s.put_artifact('program',b'(ridge)')
        self.assertEqual(a,b);self.assertEqual(len(self.f.objects),1)
    def test_collision_never_overwrites(self):
        self.s.put(PREFIX+'fixed',b'a')
        with self.assertRaises(ValueError):self.s.put(PREFIX+'fixed',b'b')
        self.assertEqual(self.f.objects[PREFIX+'fixed'][0],b'a')
    def test_no_unversioned_or_wrong_namespace(self):
        ref=self.s.put_artifact('program',b'abc')
        for k,v in [('version_id','null'),('bucket','foreign'),('key','foreign/key')]:
            with self.assertRaises(ValueError):self.s.get_version({**ref,k:v})
    def test_expired_and_last_read_deadline_refused(self):
        self.s.deadline=0
        with self.assertRaises(TimeoutError):self.s.put_artifact('program',b'abc')
        self.s.deadline=time.time()+60;ref=self.s.put_artifact('program',b'abc');original=self.f.get_object
        def slow(**kw):
            r=original(**kw);self.s.deadline=0;return r
        self.f.get_object=slow
        with self.assertRaises(TimeoutError):self.s.get_version(ref)
    def test_slow_absence_cannot_dispatch_after_deadline(self):
        def slow(**kw):self.s.deadline=0;raise Error('404')
        self.f.head_object=slow
        with self.assertRaises(TimeoutError):self.s.lookup('a'*64)
if __name__=='__main__':unittest.main()
