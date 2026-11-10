import numpy as np,itertools,collections
def fit(rows,minimize=True,center=1500.,iters=250,reg=1e-4,tol=1e-8,clamp=40.):
    """rows: iterable (dataset,method,value). One match per (dataset, method pair); tie -> 0.5."""
    by=collections.defaultdict(dict)
    for d,m,v in rows: by[d][m]=v
    methods=sorted({m for d in by for m in by[d]}); ix={m:i for i,m in enumerate(methods)}
    A=[];B=[];O=[]
    for d in sorted(by):
        ms=sorted(by[d])
        for a,b in itertools.combinations(ms,2):
            va,vb=by[d][a],by[d][b]
            if abs(va-vb)<=1e-12*max(abs(va),abs(vb),1): o=.5
            else: o=1. if ((va<vb) if minimize else (va>vb)) else 0.
            A.append(ix[a]);B.append(ix[b]);O.append(o)
    A=np.array(A);B=np.array(B);O=np.array(O);n=len(methods);r=np.full(n,center);s=np.log(10)/400
    for _ in range(iters):
        p=1/(1+np.exp(-np.clip(s*(r[A]-r[B]),-700,700))); res=O-p; cur=p*(1-p)
        g=s*(np.bincount(A,res,n)-np.bincount(B,res,n)); I=s*s*(np.bincount(A,cur,n)+np.bincount(B,cur,n))
        u=np.clip((g-reg*(r-center))/(I+reg),-clamp,clamp); r+=u
        if np.abs(u).max()<tol: break
    r=r-r.mean()+center
    cov=collections.Counter(m for d in by for m in by[d])
    return sorted(((methods[i],r[i],cov[methods[i]]) for i in range(n)),key=lambda x:-x[1]), len(by)
