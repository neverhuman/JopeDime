"""At most four I/O tasks, exact request budgets, one frozen wall/monotonic clock.
Started tasks finish custody even when a peer fails; no further task is submitted.
Exceptions preserve every already-recorded object and do not imply rollback of S3.
"""
import concurrent.futures,threading,time

def need(ok,msg):
 if not ok:raise ValueError(msg)

class Gate:
 def __init__(self,deadline,gets,puts,heads=0):
  self.deadline=deadline;self.stop=time.monotonic()+deadline-time.time();self.limits={'get':gets,'put':puts,'head':heads};self.counts=dict.fromkeys(self.limits,0);self.lock=threading.Lock()
 def tick(self):need(min(self.deadline-time.time(),self.stop-time.monotonic())>0,'I/O shared deadline')
 def take(self,op):
  with self.lock:self.tick();need(self.counts[op]<self.limits[op],op+' request budget');self.counts[op]+=1
 def snapshot(self):
  with self.lock:return dict(self.counts)

class Body:
 def __init__(self,body,gate,cap):self.body=body;self.gate=gate;self.cap=cap;self.count=0
 def __enter__(self):return self
 def __exit__(self,*args):self.close();return False
 def read(self,n=-1):
  self.gate.tick();n=self.cap-self.count+1 if n<0 else min(n,self.cap-self.count+1);b=self.body.read(n);self.count+=len(b);self.gate.tick();need(self.count<=self.cap,'bounded GET body');return b
 def close(self):return self.body.close()

class Client:
 def __init__(self,client,gate):self.client,self.gate=client,gate
 @property
 def logger(self):return self.client.logger
 def get_object(self,**kw):
  need(isinstance(kw.get('VersionId'),str)and kw['VersionId']not in ('','null'),'exactVersion GET only');self.gate.take('get');r=self.client.get_object(**kw);self.gate.tick();need(type(r.get('ContentLength'))is int and 0<=r['ContentLength']<=8*1024**2,'response byte cap');r['Body']=Body(r['Body'],self.gate,r['ContentLength']);return r
 def put_object(self,**kw):
  need(kw.get('IfNoneMatch')=='*'and isinstance(kw.get('Body'),bytes)and len(kw['Body'])<=2*1024**2,'create-only bounded PUT');self.gate.take('put');r=self.client.put_object(**kw);self.gate.tick();return r
 def head_object(self,**kw):
  self.gate.take('head');r=self.client.head_object(**kw);self.gate.tick();return r

def parallel_each(items,fn,on_result,workers=4):
 need(workers==4,'fixed four I/O workers');it=iter(items);pending={};errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers,thread_name_prefix='kumo-io')as pool:
  def fill():
   while len(pending)<workers:
    try:item=next(it)
    except StopIteration:return
    pending[pool.submit(fn,item)]=item
  fill()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in done:
    item=pending.pop(future)
    try:result=future.result();on_result(result)
    except BaseException as e:errors.append({'identity':item.get('variant_identity',item.get('row',{}).get('identity'))if isinstance(item,dict)else str(item),'error':type(e).__name__+': '+str(e)[:400]})
   if not errors:fill()
 if errors:raise RuntimeError('bounded I/O task failures: '+repr(errors))
