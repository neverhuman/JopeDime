pub(super) const CORE: &str = r#"
function clampRange(x,lo,hi){if(x<lo)x=lo;if(x>hi)x=hi;return x;}
function maxNum(a,b){if(Number.isNaN(a))return b;if(Number.isNaN(b))return a;return a<b?b:a;}
function totalCompare(a,b){
  if(a<b)return -1;if(a>b)return 1;
  if(Object.is(a,-0)&&Object.is(b,0))return -1;
  if(Object.is(a,0)&&Object.is(b,-0))return 1;
  return 0;
}
function sanitize(x){return x.map(row=>Array.from(row,v=>Number.isFinite(v)?v:0.0));}
function finishStage(x,out,append){
  for(let r=0;r<out.length;r++){
    let row=append?x[r].concat(out[r]):out[r];
    if(row.length>512)row.length=512;
    for(let j=0;j<row.length;j++)if(!Number.isFinite(row[j]))row[j]=0.0;
    if(row.length===0)row.push(0.0);
    out[r]=row;
  }
  return out;
}
function stdFit(x){
  const n=Math.max(x.length,1),p=x[0].length,means=new Array(p).fill(0.0),scales=new Array(p).fill(1.0);
  for(let j=0;j<p;j++){
    let s=0.0;for(const row of x)s+=row[j];means[j]=s/n;
    let v=0.0;for(const row of x){const d=row[j]-means[j];v+=d*d;}
    scales[j]=maxNum(Math.sqrt(v/n),1e-9);
  }
  return {means,scales};
}
function choleskySolve(a,b,p){
  for(let i=0;i<p;i++)for(let j=0;j<=i;j++){
    let s=a[i*p+j];for(let k=0;k<j;k++)s-=a[i*p+k]*a[j*p+k];
    a[i*p+j]=i===j?(s>1e-12?Math.sqrt(s):1e-6):s/a[j*p+j];
  }
  const z=new Array(p).fill(0.0);
  for(let i=0;i<p;i++){let s=b[i];for(let k=0;k<i;k++)s-=a[i*p+k]*z[k];z[i]=s/a[i*p+i];}
  const w=new Array(p).fill(0.0);
  for(let ii=0;ii<p;ii++){const i=p-1-ii;let s=z[i];for(let k=i+1;k<p;k++)s-=a[k*p+i]*w[k];w[i]=s/a[i*p+i];}
  return w;
}
function sortedColumn(x,j){const out=x.map(row=>row[j]);out.sort(totalCompare);return out;}
function quantile(s,q){if(s.length===0)return 0.0;const pos=q*(s.length-1),lo=Math.trunc(pos),hi=Math.min(lo+1,s.length-1),f=pos-lo;return s[lo]*(1.0-f)+s[hi]*f;}
function lowerBound(s,v){let lo=0,hi=s.length;while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(s[mid]<v)lo=mid+1;else hi=mid;}return lo;}
function upperBound(s,v){let lo=0,hi=s.length;while(lo<hi){const mid=Math.trunc((lo+hi)/2);if(s[mid]<=v)lo=mid+1;else hi=mid;}return lo;}
function midRank(s,v){return (lowerBound(s,v)+upperBound(s,v))/(2.0*s.length);}
function probit(p){
  const a=[-3.969683028665376e1,2.209460984245205e2,-2.759285104469687e2,1.383577518672690e2,-3.066479806614716e1,2.506628277459239],
    b=[-5.447609879822406e1,1.615858368580409e2,-1.556989798598866e2,6.680131188771972e1,-1.328068155288572e1],
    c=[-7.784894002430293e-3,-3.223964580411365e-1,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783],
    d=[7.784695709041462e-3,3.224671290700398e-1,2.445134137142996,3.754408661907416];
  p=clampRange(p,1e-12,1.0-1e-12);
  if(p<0.02425){const q=Math.sqrt(-2.0*mLog(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);}
  if(p<=1.0-0.02425){const q=p-0.5,r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);}
  const q=Math.sqrt(-2.0*mLog(1.0-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
}
function mSigmoid(v){return 1.0/(1.0+mExp(-clampRange(v,-30.0,30.0)));}
function dot(a,b){let s=0.0;for(let i=0;i<Math.min(a.length,b.length);i++)s+=a[i]*b[i];return s;}
function pearson(a,b){
  const n=Math.min(a.length,b.length);if(n<2)return 0.0;
  let ma=0.0,mb=0.0;for(let i=0;i<n;i++){ma+=a[i];mb+=b[i];}ma/=n;mb/=n;
  let sab=0.0,saa=0.0,sbb=0.0;for(let i=0;i<n;i++){const da=a[i]-ma,db=b[i]-mb;sab+=da*db;saa+=da*da;sbb+=db*db;}
  return saa<=0.0||sbb<=0.0?0.0:sab/Math.sqrt(saa*sbb);
}
function quantileEdges(x,bins){const out=[];for(let j=0;j<x[0].length;j++){const s=sortedColumn(x,j),e=[];for(let b=1;b<bins;b++)e.push(quantile(s,b/bins));out.push(e);}return out;}

const PCG_MASK64=0xffffffffffffffffn;
function rotr32(x,r){return ((x>>>r)|(x<<((32-r)&31)))>>>0;}
class Pcg32{
  constructor(seed,stream){this.state=0n;this.inc=((stream<<1n)|1n)&PCG_MASK64;this.nextU32();this.state=(this.state+seed)&PCG_MASK64;this.nextU32();}
  nextU32(){const old=this.state;this.state=(old*6364136223846793005n+this.inc)&PCG_MASK64;const x=Number((((old>>18n)^old)>>27n)&0xffffffffn)>>>0,r=Number((old>>59n)&31n);return rotr32(x,r);}
  nextF64(){const bits=(BigInt(this.nextU32())<<32n)|BigInt(this.nextU32());return Number(bits>>11n)*(1.0/9007199254740992.0);}
  rangeF64(a,b){return a+(b-a)*this.nextF64();}
  gaussian(){const u1=maxNum(this.nextF64(),1e-12),u2=this.nextF64();return Math.sqrt(-2.0*mLog(u1))*mCos(2.0*3.141592653589793*u2);}
}
function stageRng(seed,inCols,nOut){return new Pcg32(seed,(BigInt(inCols)<<32n)^BigInt(nOut));}

function standardizeFit(x){return stdFit(x);}
function standardizeApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>clampRange((v-s.means[j])/s.scales[j],-20.0,20.0))),a);}
function robustScaleFit(x){const medians=[],scales=[];for(let j=0;j<x[0].length;j++){const s=sortedColumn(x,j);medians.push(quantile(s,0.5));scales.push(maxNum(quantile(s,0.75)-quantile(s,0.25),1e-9));}return {medians,scales};}
function robustScaleApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>(v-s.medians[j])/s.scales[j])),a);}
function rankFit(x){const sorted=[];for(let j=0;j<x[0].length;j++)sorted.push(sortedColumn(x,j));return {sorted};}
function rankApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>midRank(s.sorted[j],v))),a);}
function blomApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>{const n=s.sorted[j].length,mid=midRank(s.sorted[j],v)*n;return probit((mid+0.125)/(n+0.25));})),a);}
function winsorFit(x,q){const lo=[],hi=[];for(let j=0;j<x[0].length;j++){const s=sortedColumn(x,j),low=quantile(s,q);lo.push(low);hi.push(maxNum(quantile(s,1.0-q),low));}return {lo,hi};}
function winsorApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>clampRange(v,s.lo[j],s.hi[j]))),a);}
function sigmoidApply(x,a){return finishStage(x,x.map(row=>row.map(mSigmoid)),a);}
function stepApply(x,t,a){return finishStage(x,x.map(row=>row.map(v=>v>t?1.0:0.0)),a);}
function normRowApply(x,a){return finishStage(x,x.map(row=>{let s=0.0;for(const v of row)s+=v*v;const inv=1.0/maxNum(Math.sqrt(s),1e-9);return row.map(v=>v*inv);}),a);}
function chebApply(x,degree,a){return finishStage(x,x.map(row=>{const out=[];for(const v of row){const u=clampRange(2.0*v-1.0,-1.0,1.0);let prev=1.0,cur=u;for(let d=0;d<degree;d++){out.push(cur);const next=2.0*u*cur-prev;prev=cur;cur=next;}}return out;}),a);}
function rffFit(x,n,gamma,seed){const rng=stageRng(seed,x[0].length,n),scale=Math.sqrt(2.0*gamma),w=[],b=[];for(let i=0;i<n;i++){const row=[];for(let j=0;j<x[0].length;j++)row.push(rng.gaussian()*scale);w.push(row);}for(let i=0;i<n;i++)b.push(rng.rangeF64(0.0,2.0*3.141592653589793));return {w,b};}
function rffApply(s,x,a,trig){const scale=Math.sqrt(2.0/s.w.length);return finishStage(x,x.map(row=>s.w.map((w,i)=>scale*trig(dot(w,row)+s.b[i]))),a);}

function hingeFit(x,knots){const fitted=[];for(let j=0;j<x[0].length;j++){const s=sortedColumn(x,j),col=[];for(let i=1;i<=knots;i++)col.push(quantile(s,i/(knots+1)));fitted.push(col);}return {knots:fitted};}
function hingeApply(s,x,a){return finishStage(x,x.map(row=>{const out=[];for(let j=0;j<row.length;j++)for(const t of s.knots[j])out.push(maxNum(row[j]-t,0.0));return out;}),a);}
function polyFit(x){const pairs=[],cols=x[0].length;for(let i=0;i<cols;i++)pairs.push([i,i]);let full=false;for(let i=0;i<cols&&!full;i++)for(let j=i+1;j<cols;j++){if(pairs.length>=512){full=true;break;}pairs.push([i,j]);}return {pairs};}
function pairsApply(s,x,a){return finishStage(x,x.map(row=>s.pairs.map(pair=>row[pair[0]]*row[pair[1]])),a);}
function quantileBinsApply(s,x,bins,a){return finishStage(x,x.map(row=>row.map((v,j)=>lowerBound(s.edges[j],v)/(bins-1.0))),a);}
function onehotApply(s,x,bins,a){return finishStage(x,x.map(row=>{const out=new Array(row.length*bins).fill(0.0);for(let j=0;j<row.length;j++)out[j*bins+Math.min(lowerBound(s.edges[j],row[j]),bins-1)]=1.0;return out;}),a);}
function pairProdFit(x,y,topk){
  const scored=[],cols=x[0].length,k=Math.min(topk,cols);
  for(let j=0;j<cols;j++)scored.push([j,Math.abs(pearson(x.map(row=>row[j]),y))]);
  scored.sort((a,b)=>{if(totalCompare(b[1],a[1])<0)return -1;if(totalCompare(a[1],b[1])<0)return 1;return a[0]-b[0];});
  const selected=scored.slice(0,k).map(v=>v[0]).sort((a,b)=>a-b),pairs=[];
  for(let i=0;i<selected.length;i++)for(let j=i+1;j<selected.length;j++)pairs.push([selected[i],selected[j]]);
  return {pairs};
}
function randProjFit(x,k,seed){const rng=stageRng(seed,x[0].length,k),w=[];for(let i=0;i<k;i++){const row=[];for(let j=0;j<x[0].length;j++)row.push(rng.gaussian());w.push(row);}return {w};}
function randProjApply(s,x,a){const inv=1.0/Math.sqrt(s.w.length);return finishStage(x,x.map(row=>s.w.map(w=>dot(w,row)*inv)),a);}
function rowStatsApply(x,a){return finishStage(x,x.map(row=>{
  const n=row.length;let mean=0.0;for(const v of row)mean+=v;mean/=n;
  let m2=0.0,m3=0.0,m4=0.0,energy=0.0,mn=Infinity,mx=-Infinity;
  for(const v of row){const d=v-mean;m2+=d*d;m3+=d*d*d;m4+=d*d*d*d;energy+=v*v;if(v<mn)mn=v;if(v>mx)mx=v;}
  m2/=n;m3/=n;m4/=n;energy/=n;const sd=Math.sqrt(m2),skew=m2>1e-24?m3/(m2*sd):0.0,kurt=m2>1e-24?m4/(m2*m2)-3.0:0.0,s=Array.from(row);s.sort(totalCompare);
  return [mean,sd,mn,mx,quantile(s,0.5),quantile(s,0.25),quantile(s,0.75),skew,kurt,energy];
}),a);}
function rowDiffApply(x,a){return finishStage(x,x.map(row=>{
  const d=row.length;if(d<2)return new Array(8).fill(0.0);const nd=d-1,diffs=[];for(let j=0;j<nd;j++)diffs.push(row[j+1]-row[j]);
  let mean=0.0;for(const v of diffs)mean+=v;mean/=nd;let m2=0.0,meanAbs=0.0,energy=0.0,mn=Infinity,mx=-Infinity,signs=0;
  for(let j=0;j<nd;j++){const v=diffs[j],dd=v-mean;m2+=dd*dd;meanAbs+=Math.abs(v);energy+=v*v;if(v<mn)mn=v;if(v>mx)mx=v;if(j>0&&diffs[j-1]*v<0.0)signs++;}
  return [mean,Math.sqrt(m2/nd),mn,mx,meanAbs/nd,signs/nd,pearson(row.slice(0,nd),row.slice(1)),energy/nd];
}),a);}
function obliqueFit(x,n,seed){
  const rng=stageRng(seed,x[0].length,n),inv=1.0/Math.sqrt(x[0].length),w=[],t=[];
  for(let i=0;i<n;i++){const row=[];for(let j=0;j<x[0].length;j++)row.push(rng.gaussian()*inv);w.push(row);}
  for(const wi of w){const p=x.map(row=>dot(wi,row));p.sort(totalCompare);t.push(quantile(p,0.5));}return {w,t};
}
function obliqueApply(s,x,a){return finishStage(x,x.map(row=>s.w.map((w,i)=>maxNum(dot(w,row)-s.t[i],0.0))),a);}
function landmarkFit(x,k,seed){const rng=stageRng(seed,x[0].length,k),landmarks=[];for(let i=0;i<k;i++){const row=[];for(let j=0;j<x[0].length;j++)row.push(rng.rangeF64(0.0,1.0));landmarks.push(row);}return {landmarks};}
function landmarkApply(s,x,gamma,a){return finishStage(x,x.map(row=>s.landmarks.map(mark=>{let d2=0.0;for(let j=0;j<Math.min(mark.length,row.length);j++){const d=row[j]-mark[j];d2+=d*d;}return mExp(-gamma*d2);})),a);}
function targetBinFit(x,y,bins,prior){
  const edges=quantileEdges(x,bins),enc=[];let yMean=0.0;for(const v of y)yMean+=v;yMean/=Math.max(y.length,1);
  for(let j=0;j<x[0].length;j++){const sums=new Array(bins).fill(0.0),counts=new Array(bins).fill(0.0);for(let r=0;r<x.length;r++){const b=Math.min(lowerBound(edges[j],x[r][j]),bins-1);sums[b]+=y[r];counts[b]+=1.0;}const col=[];for(let b=0;b<bins;b++)col.push((sums[b]+prior*yMean)/maxNum(counts[b]+prior,1e-9));enc.push(col);}
  return {edges,enc};
}
function targetBinApply(s,x,a){return finishStage(x,x.map(row=>row.map((v,j)=>{const bins=s.enc[j].length;return s.enc[j][Math.min(lowerBound(s.edges[j],v),bins-1)];})),a);}

function softThreshold(v,t){if(v>t)return v-t;if(v< -t)return v+t;return 0.0;}
function elasticFit(x,y,alpha,l1Ratio,sweeps){
  const n=Math.min(x.length,y.length),p=x[0].length,{means,scales}=stdFit(x),xs=[];
  for(let r=0;r<n;r++){const row=[];for(let j=0;j<p;j++)row.push(clampRange((x[r][j]-means[j])/scales[j],-20.0,20.0));xs.push(row);}
  let yMean=0.0;for(let i=0;i<n;i++)yMean+=y[i];yMean/=Math.max(n,1);
  const l1=alpha*l1Ratio,l2=alpha*(1.0-l1Ratio),colSq=new Array(p).fill(0.0),w=new Array(p).fill(0.0),resid=new Array(n);
  for(const row of xs)for(let j=0;j<p;j++)colSq[j]+=row[j]*row[j];for(let r=0;r<n;r++)resid[r]=y[r]-yMean;
  for(let sweep=0;sweep<Math.max(sweeps,1);sweep++){let maxDelta=0.0;for(let j=0;j<p;j++){
    if(colSq[j]<=1e-12)continue;let rho=0.0;for(let r=0;r<n;r++)rho+=xs[r][j]*resid[r];rho=rho/n+w[j]*colSq[j]/n;
    const next=softThreshold(rho,l1)/(colSq[j]/n+l2),delta=next-w[j];if(delta!==0.0){for(let r=0;r<n;r++)resid[r]-=delta*xs[r][j];w[j]=next;const ad=Math.abs(delta);if(ad>maxDelta)maxDelta=ad;}
  }if(maxDelta<1e-9)break;}
  return {means,scales,w0:yMean,w};
}
function elasticPredict(s,x){return x.map(row=>{let z=s.w0;for(let j=0;j<Math.min(s.means.length,row.length);j++)z+=s.w[j]*clampRange((row[j]-s.means[j])/s.scales[j],-20.0,20.0);return clampRange(z,-60.0,60.0);});}
function ridgeFit(x,y,lambdaFrac){
  const n=Math.min(x.length,y.length),p=x[0].length,{means,scales}=stdFit(x),lam=maxNum(lambdaFrac*n,1e-3),ata=new Array(p*p).fill(0.0),atb=new Array(p).fill(0.0),xs=new Array(p).fill(0.0);
  let yMean=0.0;for(let i=0;i<n;i++)yMean+=y[i];yMean/=Math.max(n,1);
  for(let r=0;r<n;r++){for(let j=0;j<p;j++)xs[j]=clampRange((x[r][j]-means[j])/scales[j],-20.0,20.0);const yr=y[r]-yMean;for(let i=0;i<p;i++){const xi=xs[i];atb[i]+=xi*yr;for(let j=i;j<p;j++)ata[i*p+j]+=xi*xs[j];}}
  for(let i=0;i<p;i++){ata[i*p+i]+=lam;for(let j=0;j<i;j++)ata[i*p+j]=ata[j*p+i];}
  return {means,scales,w:[yMean].concat(choleskySolve(ata,atb,p))};
}
function ridgePredict(s,x){return x.map(row=>{let z=s.w[0];for(let j=0;j<Math.min(s.means.length,row.length);j++)z+=s.w[j+1]*clampRange((row[j]-s.means[j])/s.scales[j],-20.0,20.0);return clampRange(z,-60.0,60.0);});}

function fitStage(spec,x,y){
  let state;
  switch(spec.kind){
    case "standardize":state=standardizeFit(x);break;
    case "robust_scale":state=robustScaleFit(x);break;
    case "rank":case "blom":state=rankFit(x);break;
    case "quantile_bins":case "bin_onehot":state={edges:quantileEdges(x,spec.bins)};break;
    case "hinge":state=hingeFit(x,spec.knots);break;
    case "winsor_clip":state=winsorFit(x,spec.q);break;
    case "poly_interact":state=polyFit(x);break;
    case "rff_cos":case "rff_sin":state=rffFit(x,spec.n,spec.gamma,spec.seed);break;
    case "rand_proj":state=randProjFit(x,spec.k,spec.seed);break;
    case "landmark_dist":state=landmarkFit(x,spec.k,spec.seed);break;
    case "oblique_hinge":state=obliqueFit(x,spec.n,spec.seed);break;
    case "pair_prod":state=pairProdFit(x,y,spec.topk);break;
    case "target_bin_encode":state=targetBinFit(x,y,spec.bins,spec.prior);break;
    case "cheb":case "row_stats":case "sigmoid":case "step":case "norm_row":case "row_diff_stats":state=null;break;
    default:throw new RangeError("unsupported fitted stage");
  }
  return [state,applyStage(spec,state,x)];
}
function applyStage(spec,state,x){
  switch(spec.kind){
    case "standardize":return standardizeApply(state,x,spec.append);
    case "robust_scale":return robustScaleApply(state,x,spec.append);
    case "rank":return rankApply(state,x,spec.append);
    case "blom":return blomApply(state,x,spec.append);
    case "quantile_bins":return quantileBinsApply(state,x,spec.bins,spec.append);
    case "bin_onehot":return onehotApply(state,x,spec.bins,spec.append);
    case "hinge":return hingeApply(state,x,spec.append);
    case "winsor_clip":return winsorApply(state,x,spec.append);
    case "cheb":return chebApply(x,spec.degree,spec.append);
    case "poly_interact":return pairsApply(state,x,spec.append);
    case "rff_cos":return rffApply(state,x,spec.append,mCos);
    case "rff_sin":return rffApply(state,x,spec.append,mSin);
    case "rand_proj":return randProjApply(state,x,spec.append);
    case "landmark_dist":return landmarkApply(state,x,spec.gamma,spec.append);
    case "oblique_hinge":return obliqueApply(state,x,spec.append);
    case "row_stats":return rowStatsApply(x,spec.append);
    case "pair_prod":return pairsApply(state,x,spec.append);
    case "target_bin_encode":return targetBinApply(state,x,spec.append);
    case "sigmoid":return sigmoidApply(x,spec.append);
    case "step":return stepApply(x,spec.t,spec.append);
    case "norm_row":return normRowApply(x,spec.append);
    case "row_diff_stats":return rowDiffApply(x,spec.append);
    default:throw new RangeError("unsupported applied stage");
  }
}
function fitHead(spec,x,y){if(spec.kind==="ridge")return ridgeFit(x,y,spec.lambdaFrac);if(spec.kind==="elasticnet")return elasticFit(x,y,spec.alpha,spec.l1Ratio,spec.sweeps);throw new RangeError("unsupported fitted head");}
function predictHead(spec,state,x){if(spec.kind==="ridge")return ridgePredict(state,x);if(spec.kind==="elasticnet")return elasticPredict(state,x);throw new RangeError("unsupported applied head");}
"#;
