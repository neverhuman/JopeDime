// JS all-function port of portable_math.rs. Byte-identical to full_ref.rs.
// Only bit-manipulation uses BigInt; NEVER native Math.exp/log/sin/... (the point).
'use strict';
const _buf = new ArrayBuffer(8);
const _f64 = new Float64Array(_buf);
const _u64 = new BigUint64Array(_buf);
const M64 = 0xffffffffffffffffn;
function toBits(x){ _f64[0]=x; return _u64[0]; }
function fromBits(b){ _u64[0]=b & M64; return _f64[0]; }
function highWord(x){ return Number((toBits(x)>>32n)&0xffffffffn); }
function lowWord(x){ return Number(toBits(x)&0xffffffffn); }
function withHighWord(x,hi){ return fromBits((BigInt(hi>>>0)<<32n)|(toBits(x)&0xffffffffn)); }
function withLowWord(x,lo){ return fromBits((toBits(x)&0xffffffff00000000n)|BigInt(lo>>>0)); }
function fabs(x){ return fromBits(toBits(x)&0x7fffffffffffffffn); }
function canonicalNan(){ return fromBits(0x7ff8000000000000n); }
function twoPow(n){ return fromBits(BigInt.asUintN(64, BigInt.asUintN(64, BigInt(0x3ff+n))<<52n)); }
function roundHalfAway(v){ return Math.trunc(v + (v>=0?0.5:-0.5)); }
function isNan(x){ return x!==x; }

function scalbn(x,n){
  const p1023=twoPow(1023), pm1022=twoPow(-1022), p53=twoPow(53); let y=x;
  if(n>1023){ y*=p1023; n-=1023; if(n>1023){ y*=p1023; n-=1023; if(n>1023) n=1023; } }
  else if(n<-1022){ y*=pm1022*p53; n+=1022-53; if(n<-1022){ y*=pm1022*p53; n+=1022-53; if(n<-1022) n=-1022; } }
  return y*twoPow(n);
}

function xexp(x){
  if(isNan(x)) return canonicalNan();
  const HALF=[0.5,-0.5], LN2HI=[6.9314718036912381649e-1,-6.9314718036912381649e-1], LN2LO=[1.9082149292705877000e-10,-1.9082149292705877000e-10];
  const INVLN2=1.4426950408889633870, P1=1.66666666666666019037e-1,P2=-2.77777777770155933842e-3,P3=6.61375632143793436117e-5,P4=-1.65339022054652515390e-6,P5=4.13813679705723846039e-8;
  const HUGE=1.0e300,TWOM1000=9.33263618503218878990e-302,O_THRESHOLD=7.09782712893383973096e2,U_THRESHOLD=-7.45133219101941108420e2;
  let hx=highWord(x); const xsb=(hx>>>31)&1; hx=hx&0x7fffffff;
  if(hx>=0x40862E42){ if(hx>=0x7ff00000){ if(((hx&0xfffff)|lowWord(x))!==0) return x+x; return xsb===0?x:0.0; } if(x>O_THRESHOLD) return HUGE*HUGE; if(x<U_THRESHOLD) return TWOM1000*TWOM1000; }
  let k=0,hi=0.0,lo=0.0,xr;
  if(hx>0x3fd62e42){ if(hx<0x3FF0A2B2){ hi=x-LN2HI[xsb]; lo=LN2LO[xsb]; k=1-xsb-xsb; } else { k=Math.trunc(INVLN2*x+HALF[xsb]); const t=k; hi=x-t*LN2HI[0]; lo=t*LN2LO[0]; } xr=hi-lo; }
  else if(hx<0x3e300000){ return 1.0+x; } else { xr=x; }
  const t=xr*xr, c=xr-t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
  if(k===0) return 1.0-((xr*c)/(c-2.0)-xr);
  const y=1.0-((lo-(xr*c)/(2.0-c))-hi); return scalbn(y,k);
}

function xln(x){
  if(isNan(x)) return canonicalNan();
  const LN2_HI=6.9314718036912381649e-1,LN2_LO=1.9082149292705877000e-10,TWO54=1.80143985094819840000e16;
  const LG1=6.666666666666735130e-1,LG2=3.999999999940941908e-1,LG3=2.857142874366239149e-1,LG4=2.222219843214978396e-1,LG5=1.818357216161805012e-1,LG6=1.531383769920937332e-1,LG7=1.479819860511658591e-1;
  let hx=highWord(x)|0; const lx=lowWord(x); let k=0,xx=x;
  if(hx<0x00100000){ if(((hx&0x7fffffff)|(lx|0))===0) return -TWO54/0.0; if(hx<0) return canonicalNan(); k-=54; xx*=TWO54; hx=highWord(xx)|0; }
  if(hx>=0x7ff00000) return x+x;
  k+=(hx>>20)-1023; hx&=0x000fffff; const i=(hx+0x95f64)&0x100000;
  xx=withHighWord(xx,(hx|(i^0x3ff00000))>>>0); k+=i>>20; const f=xx-1.0;
  if((0x000fffff&(2+hx))<3){ if(f===0.0){ if(k===0) return 0.0; const dk=k; return dk*LN2_HI+dk*LN2_LO; } const r=f*f*(0.5-0.33333333333333333*f); if(k===0) return f-r; const dk=k; return dk*LN2_HI-((r-dk*LN2_LO)-f); }
  const s=f/(2.0+f),dk=k,z=s*s,w=z*z; const t1=w*(LG2+w*(LG4+w*LG6)),t2=z*(LG1+w*(LG3+w*(LG5+w*LG7))),r=t2+t1;
  const i2=hx-0x6147a,j2=0x6b851-hx;
  if((i2|j2)>0){ const hfsq=0.5*f*f; if(k===0) return f-(hfsq-s*(hfsq+r)); return dk*LN2_HI-((hfsq-(s*(hfsq+r)+dk*LN2_LO))-f); }
  else if(k===0) return f-s*(f-r);
  return dk*LN2_HI-((s*(f-r)-dk*LN2_LO)-f);
}

function xlog1p(x){
  if(isNan(x)) return canonicalNan();
  const LN2_HI=6.9314718036912381649e-1,LN2_LO=1.9082149292705877000e-10;
  const LG1=6.666666666666735130e-1,LG2=3.999999999940941908e-1,LG3=2.857142874366239149e-1,LG4=2.222219843214978396e-1,LG5=1.818357216161805012e-1,LG6=1.531383769920937332e-1,LG7=1.479819860511658591e-1;
  let hx=highWord(x); let k=1,f=0.0,c=0.0;
  if(hx<0x3fda827a || (hx>>>31)!==0){ if(hx>=0xbff00000){ if(x===-1.0) return x/0.0; return canonicalNan(); } if(((hx<<1)>>>0) < ((0x3ca00000<<1)>>>0)) return x; if(hx<=0xbfd2bec4){ k=0; c=0.0; f=x; } }
  else if(hx>=0x7ff00000){ return x; }
  if(k!==0){ const u=1.0+x; let hu=highWord(u); hu=(hu+(0x3ff00000-0x3fe6a09e))>>>0; k=((hu>>>20)|0)-0x3ff; if(k<54){ c=(k>=2)?(1.0-(u-x)):(x-(u-1.0)); c/=u; } else c=0.0; hu=((hu&0x000fffff)+0x3fe6a09e)>>>0; f=withHighWord(u,hu)-1.0; }
  const hfsq=0.5*f*f,s=f/(2.0+f),z=s*s,w=z*z; const t1=w*(LG2+w*(LG4+w*LG6)),t2=z*(LG1+w*(LG3+w*(LG5+w*LG7))),r=t2+t1,dk=k;
  return s*(hfsq+r)+(dk*LN2_LO+c)-hfsq+f+dk*LN2_HI;
}

function xexpm1(x){
  if(isNan(x)) return canonicalNan();
  const LN2_HI=6.9314718036912381649e-1,LN2_LO=1.90821492927058770002e-10,INVLN2=1.4426950408889633870;
  const Q1=-3.33333333333331315428e-2,Q2=1.58730158725481460165e-3,Q3=-7.93650757867487942473e-5,Q4=4.00821782732936239552e-6,Q5=-2.01099218183624371326e-7;
  const hx=highWord(x)&0x7fffffff, sign=(highWord(x)>>>31)!==0; let c=0.0,k,xx=x;
  if(hx>=0x4043687A){ if(sign) return -1.0; if(hx>=0x40862E42) return x*twoPow(1023); }
  if(hx>0x3fd62e42){ let hi,lo; if(hx<0x3FF0A2B2){ if(!sign){ hi=x-LN2_HI; lo=LN2_LO; k=1; } else { hi=x+LN2_HI; lo=-LN2_LO; k=-1; } } else { k=Math.trunc(INVLN2*x+(sign?-0.5:0.5)); const t=k; hi=x-t*LN2_HI; lo=t*LN2_LO; } xx=hi-lo; c=(hi-xx)-lo; }
  else if(hx<0x3c900000){ return x; } else { k=0; }
  const hfx=0.5*xx,hxs=xx*hfx; const r1=1.0+hxs*(Q1+hxs*(Q2+hxs*(Q3+hxs*(Q4+hxs*Q5)))); const t=3.0-r1*hfx; let e=hxs*((r1-t)/(6.0-xx*t));
  if(k===0) return xx-(xx*e-hxs);
  e=xx*(e-c)-c; e-=hxs;
  if(k===-1) return 0.5*(xx-e)-0.5;
  if(k===1){ if(xx<-0.25) return -2.0*(e-(xx+0.5)); return 1.0+2.0*(xx-e); }
  const twopk=twoPow(k);
  if(!(k>=0 && k<=56)){ const y=(xx-e+1.0)*twopk; return y-1.0; }
  const inv=twoPow(-k);
  if(k<20) return (xx-e+(1.0-inv))*twopk;
  return (xx-e-inv+1.0)*twopk;
}

function xpow(x,y){
  const BP=[1.0,1.5],DP_H=[0.0,5.84962487220764160156e-1],DP_L=[0.0,1.35003920212974897128e-8];
  const TWO53=9007199254740992.0,HUGE=1.0e300,TINY=1.0e-300;
  const L1=5.99999999999994648725e-1,L2=4.28571428578550184252e-1,L3=3.33333329818377432918e-1,L4=2.72728123808534006489e-1,L5=2.30660745775561366331e-1,L6=2.06975017800338417784e-1;
  const P1=1.66666666666666019037e-1,P2=-2.77777777770155933842e-3,P3=6.61375632143793436117e-5,P4=-1.65339022054652515390e-6,P5=4.13813679705723846039e-8;
  const LG2=6.93147180559945286227e-1,LG2_H=6.93147182464599609375e-1,LG2_L=-1.90465429995776804525e-9,OVT=8.0085662595372944372e-17;
  const CP=9.61796693925975554329e-1,CP_H=9.61796700954437255859e-1,CP_L=-7.02846165095275826516e-9;
  const IVLN2=1.4426950408889633870,IVLN2_H=1.44269502162933349609e0,IVLN2_L=1.92596299112661746887e-8;
  let hx=highWord(x)|0; const lx=lowWord(x); const hy=highWord(y)|0; const ly=lowWord(y);
  let ix=hx&0x7fffffff; const iy=hy&0x7fffffff;
  if((iy|(ly|0))===0) return 1.0;
  if(isNan(x)||isNan(y)) return canonicalNan();
  if(ix>0x7ff00000||(ix===0x7ff00000&&lx!==0)||iy>0x7ff00000||(iy===0x7ff00000&&ly!==0)) return canonicalNan();
  let yisint=0;
  if(hx<0){ if(iy>=0x43400000) yisint=2; else if(iy>=0x3ff00000){ const k=(iy>>20)-0x3ff; if(k>20){ const j=ly>>>(52-k); if(((j<<(52-k))>>>0)===ly) yisint=2-(j&1); } else if(ly===0){ const j=iy>>(20-k); if((j<<(20-k))===iy) yisint=2-(j&1); } } }
  if(ly===0){ if(iy===0x7ff00000){ if(((ix-0x3ff00000)|(lx|0))===0) return 1.0; else if(ix>=0x3ff00000) return hy>=0?y:0.0; else return hy<0?-y:0.0; } if(iy===0x3ff00000) return hy<0?1.0/x:x; if(hy===0x40000000) return x*x; if(hy===0x3fe00000&&hx>=0) return Math.sqrt(x); }
  let ax=fabs(x);
  if(lx===0&&(ix===0x7ff00000||ix===0||ix===0x3ff00000)){ let z=ax; if(hy<0) z=1.0/z; if(hx<0){ if(((ix-0x3ff00000)|yisint)===0) z=canonicalNan(); else if(yisint===1) z=-z; } return z; }
  let n=(hx>>31)+1;
  if((n|yisint)===0) return canonicalNan();
  let s=1.0; if((n|(yisint-1))===0) s=-1.0;
  let t1,t2;
  if(iy>0x41e00000){
    if(iy>0x43f00000){ if(ix<=0x3fefffff) return hy<0?HUGE*HUGE:TINY*TINY; if(ix>=0x3ff00000) return hy>0?HUGE*HUGE:TINY*TINY; }
    if(ix<0x3fefffff) return hy<0?s*HUGE*HUGE:s*TINY*TINY;
    if(ix>0x3ff00000) return hy>0?s*HUGE*HUGE:s*TINY*TINY;
    const t=ax-1.0; const w=(t*t)*(0.5-t*(0.333333333333333333333-t*0.25)); const u=IVLN2_H*t; const v=t*IVLN2_L-w*IVLN2; const t1v=withLowWord(u+v,0); t1=t1v; t2=v-(t1v-u);
  } else {
    n=0;
    if(ix<0x00100000){ ax*=TWO53; n-=53; ix=highWord(ax)|0; }
    n+=(ix>>20)-0x3ff; const j0=ix&0x000fffff; ix=(j0|0x3ff00000); let k;
    if(j0<=0x3988E) k=0; else if(j0<0xBB67A) k=1; else { k=0; n+=1; ix-=0x00100000; }
    ax=withHighWord(ax,ix>>>0);
    const u=ax-BP[k]; const v=1.0/(ax+BP[k]); const ss=u*v; const s_h=withLowWord(ss,0);
    const t_h=withHighWord(0.0,(((ix>>1)|0x20000000)+0x00080000+(k<<18))>>>0);
    const t_l=ax-(t_h-BP[k]); const s_l=v*((u-s_h*t_h)-s_h*t_l);
    const s2=ss*ss; let r=s2*s2*(L1+s2*(L2+s2*(L3+s2*(L4+s2*(L5+s2*L6))))); r+=s_l*(s_h+ss);
    const s2b=s_h*s_h; const t_h2=withLowWord(3.0+s2b+r,0); const t_l2=r-((t_h2-3.0)-s2b);
    const u2=s_h*t_h2; const v2=s_l*t_h2+t_l2*ss; const p_h=withLowWord(u2+v2,0); const p_l=v2-(p_h-u2);
    const z_h=CP_H*p_h; const z_l=CP_L*p_h+p_l*CP+DP_L[k];
    const t=n; const t1v=withLowWord(((z_h+z_l)+DP_H[k])+t,0); t1=t1v; t2=z_l-(((t1v-t)-DP_H[k])-z_h);
  }
  const y1=withLowWord(y,0); const p_l=(y-y1)*t1+y*t2; let p_h=y1*t1; const z=p_l+p_h;
  let j=highWord(z)|0; const i=lowWord(z);
  if(j>=0x40900000){ if(((j-0x40900000)|(i|0))!==0) return s*HUGE*HUGE; if(p_l+OVT>z-p_h) return s*HUGE*HUGE; }
  else if((j&0x7fffffff)>=0x4090cc00){ if(((((j>>>0)-0xc090cc00)>>>0|0)|(i|0))!==0) return s*TINY*TINY; if(p_l<=z-p_h) return s*TINY*TINY; }
  const iabs=j&0x7fffffff; let k=(iabs>>20)-0x3ff; let nn=0;
  if(iabs>0x3fe00000){ nn=(j+(0x00100000>>(k+1)))|0; k=((nn&0x7fffffff)>>20)-0x3ff; const t=withHighWord(0.0,(nn&~(0x000fffff>>k))>>>0); nn=((nn&0x000fffff)|0x00100000)>>(20-k); if(j<0) nn=-nn; p_h-=t; }
  const t=withLowWord(p_l+p_h,0); const u=t*LG2_H; const v=(p_l-(t-p_h))*LG2+t*LG2_L;
  let z2=u+v; const w=v-(z2-u); const tt=z2*z2; const t1b=z2-tt*(P1+tt*(P2+tt*(P3+tt*(P4+tt*P5))));
  const r=(z2*t1b)/(t1b-2.0)-(w+z2*w); z2=1.0-(r-z2);
  j=highWord(z2)|0; j=(j+(nn<<20))|0;
  if((j>>20)<=0) z2=scalbn(z2,nn); else z2=withHighWord(z2,j>>>0);
  return s*z2;
}

function xtanh(x){
  if(isNan(x)) return canonicalNan();
  const sign=(toBits(x)>>63n)!==0n; const ax=fromBits(toBits(x)&0x7fffffffffffffffn); const w=Number((toBits(ax)>>32n)&0xffffffffn); let t;
  if(w>0x3fe193ea){ if(w>0x40340000){ t=1.0-0.0/ax; } else { const e=xexpm1(2.0*ax); t=1.0-2.0/(e+2.0); } }
  else if(w>0x3fd058ae){ const e=xexpm1(2.0*ax); t=e/(e+2.0); }
  else if(w>=0x00100000){ const e=xexpm1(-2.0*ax); t=-e/(e+2.0); }
  else { t=ax; }
  return sign?-t:t;
}

function kernelSin(x,y,iy){
  const S1=-1.66666666666666324348e-1,S2=8.33333333332248946124e-3,S3=-1.98412698298579493134e-4,S4=2.75573137070700676789e-6,S5=-2.50507602534068634195e-8,S6=1.58969099521155010221e-10;
  const z=x*x,w=z*z,r=S2+z*(S3+z*S4)+z*w*(S5+z*S6),v=z*x;
  if(iy===0) return x+v*(S1+z*r);
  return x-((z*(0.5*y-v*r)-y)-v*S1);
}
function kernelCos(x,y){
  const C1=4.16666666666666019037e-2,C2=-1.38888888888741095749e-3,C3=2.48015872894767294178e-5,C4=-2.75573143513906633035e-7,C5=2.08757232129817482790e-9,C6=-1.13596475577881948265e-11;
  const z=x*x,w=z*z,r=z*(C1+z*(C2+z*C3))+w*w*(C4+z*(C5+z*C6)),hz=0.5*z,w2=1.0-hz;
  return w2+(((1.0-w2)-hz)+(z*r-x*y));
}
function remPio2(x){
  const INVPIO2=6.36619772367581382433e-1,PIO2_1=1.57079632673412561417e0,PIO2_1T=6.07710050650619224932e-11,PIO2_2=6.07710050630396597660e-11,PIO2_2T=2.02226624879595063154e-21,PIO2_3=2.02226624871116645580e-21,PIO2_3T=8.47842766036889956997e-32;
  const INV_TWO_PI=1.59154943091895335769e-1,TWO_PI_HI=6.28318530717958623199e0,TWO_PI_LO=2.44929359829470641471e-16;
  const ix=highWord(x)&0x7fffffff;
  if(ix>=0x7ff00000){ const z=canonicalNan(); return [0,z,z]; }
  let xr=x;
  while((highWord(xr)&0x7fffffff)>=0x413921fb){ const k=roundHalfAway(xr*INV_TWO_PI); xr=(xr-k*TWO_PI_HI)-k*TWO_PI_LO; }
  const ixr=highWord(xr)&0x7fffffff;
  if(ixr<=0x3fe921fb){ return [0,xr,0.0]; }
  const half=xr>0.0?0.5:-0.5; const n=Math.trunc(xr*INVPIO2+half); const fnn=n;
  let r=xr-fnn*PIO2_1; let w=fnn*PIO2_1T; const j=(ixr>>20)|0;
  let y0=r-w; let hi=(highWord(y0)>>20)&0x7ff; let i=j-hi;
  if(i>16){ const t=r; w=fnn*PIO2_2; r=t-w; w=fnn*PIO2_2T-((t-r)-w); y0=r-w; hi=(highWord(y0)>>20)&0x7ff; i=j-hi;
    if(i>49){ const t2=r; w=fnn*PIO2_3; r=t2-w; w=fnn*PIO2_3T-((t2-r)-w); y0=r-w; } }
  const y1=(r-y0)-w; return [n&3,y0,y1];
}
function xsin(x){
  if(isNan(x)) return canonicalNan();
  const ix=highWord(x)&0x7fffffff;
  if(ix<=0x3fe921fb){ if(ix<0x3e500000) return x; return kernelSin(x,0.0,0); }
  if(ix>=0x7ff00000) return canonicalNan();
  const [n,y0,y1]=remPio2(x);
  switch(n){ case 0: return kernelSin(y0,y1,1); case 1: return kernelCos(y0,y1); case 2: return -kernelSin(y0,y1,1); default: return -kernelCos(y0,y1); }
}
function xcos(x){
  if(isNan(x)) return canonicalNan();
  const ix=highWord(x)&0x7fffffff;
  if(ix<=0x3fe921fb){ if(ix<0x3e46a09e) return 1.0; return kernelCos(x,0.0); }
  if(ix>=0x7ff00000) return canonicalNan();
  const [n,y0,y1]=remPio2(x);
  switch(n){ case 0: return kernelCos(y0,y1); case 1: return -kernelSin(y0,y1,1); case 2: return -kernelCos(y0,y1); default: return kernelSin(y0,y1,1); }
}
function xatan(x){
  if(isNan(x)) return canonicalNan();
  const ATANHI=[4.63647609000806093515e-1,7.85398163397448278999e-1,9.82793723247329054082e-1,1.57079632679489655800e0];
  const ATANLO=[2.26987774529616870924e-17,3.06161699786838301793e-17,1.39033110312309984516e-17,6.12323399573676603587e-17];
  const AT=[3.33333333333329318027e-1,-1.99999999998764832476e-1,1.42857142725034663711e-1,-1.11111104054623557880e-1,9.09088713343650656196e-2,-7.69187620504482999495e-2,6.66107313738753120669e-2,-5.83357013379057348645e-2,4.97687799461593236017e-2,-3.65315727442169155270e-2,1.62858201153657823623e-2];
  const hx=highWord(x)|0; const ix=hx&0x7fffffff;
  if(ix>=0x44100000){ if(ix>0x7ff00000||(ix===0x7ff00000&&lowWord(x)!==0)) return canonicalNan(); return hx>0?ATANHI[3]+ATANLO[3]:-ATANHI[3]-ATANLO[3]; }
  let id,xr=x;
  if(ix<0x3fdc0000){ if(ix<0x3e400000) return x; id=-1; }
  else { xr=fabs(x); if(ix<0x3ff30000){ if(ix<0x3fe60000){ id=0; xr=(2.0*xr-1.0)/(2.0+xr); } else { id=1; xr=(xr-1.0)/(xr+1.0); } } else if(ix<0x40038000){ id=2; xr=(xr-1.5)/(1.0+1.5*xr); } else { id=3; xr=-1.0/xr; } }
  const z=xr*xr,w=z*z;
  const s1=z*(AT[0]+w*(AT[2]+w*(AT[4]+w*(AT[6]+w*(AT[8]+w*AT[10])))));
  const s2=w*(AT[1]+w*(AT[3]+w*(AT[5]+w*(AT[7]+w*AT[9]))));
  if(id<0) return xr-xr*(s1+s2);
  const z2=ATANHI[id]-((xr*(s1+s2)-ATANLO[id])-xr);
  return hx<0?-z2:z2;
}
function xatan2(y,x){
  if(isNan(x)||isNan(y)) return canonicalNan();
  const TINY=1.0e-300,PI_O_4=7.85398163397448278990e-1,PI_O_2=1.57079632679489655800e0,PI=3.14159265358979311600e0,PI_LO=1.22464679914735317722e-16;
  const hx=highWord(x)|0; const ix=hx&0x7fffffff; const lx=lowWord(x); const hy=highWord(y)|0; const iy=hy&0x7fffffff; const ly=lowWord(y);
  if((ix|(((lx|((-lx)>>>0))>>>31)|0))>0x7ff00000 || (iy|(((ly|((-ly)>>>0))>>>31)|0))>0x7ff00000) return canonicalNan();
  if(hx===0x3ff00000 && lx===0) return xatan(y);
  const m=((hy>>31)&1)|((hx>>30)&2);
  if((iy|(ly|0))===0){ switch(m){ case 0: case 1: return y; case 2: return PI+TINY; default: return -PI-TINY; } }
  if((ix|(lx|0))===0) return hy<0?-PI_O_2-TINY:PI_O_2+TINY;
  if(ix===0x7ff00000){ if(iy===0x7ff00000){ switch(m){ case 0: return PI_O_4+TINY; case 1: return -PI_O_4-TINY; case 2: return 3.0*PI_O_4+TINY; default: return -3.0*PI_O_4-TINY; } } switch(m){ case 0: return 0.0; case 1: return -0.0; case 2: return PI+TINY; default: return -PI-TINY; } }
  if(iy===0x7ff00000) return hy<0?-PI_O_2-TINY:PI_O_2+TINY;
  const k=(iy-ix)>>20; let z;
  if(k>60) z=PI_O_2+0.5*PI_LO; else if(hx<0&&k<-60) z=0.0; else z=xatan(fabs(y/x));
  switch(m){ case 0: return z; case 1: return -z; case 2: return PI-(z-PI_LO); default: return (z-PI_LO)-PI; }
}

function grid(){
  const g=[];
  for(let i=-900;i<=900;i++) g.push(i*0.41);
  for(let e=-55;e<55;e++){ g.push(twoPow(e)); g.push(-twoPow(e)); g.push(1.0+twoPow(e)); }
  g.push(0.0,-0.0,1.0,-1.0,3.5,-3.5,709.0,-745.0,710.0,fromBits(1n),fromBits(0x000fffffffffffffn),
    fromBits(0x0010000000000000n),fromBits(0x8010000000000000n),1e300,1e-300,Number.MAX_VALUE,-Number.MAX_VALUE,
    Infinity,-Infinity,fromBits(0x7ff8000000000000n),fromBits(0xfff8000000001234n));
  return g;
}
function hexout(v){ return toBits(v).toString(16).padStart(16,'0'); }

let out='';
const g=grid();
for(const x of g){ const xb=toBits(x).toString(16).padStart(16,'0');
  out+=`exp ${xb} ${hexout(xexp(x))}\n`;
  out+=`ln ${xb} ${hexout(xln(x))}\n`;
  out+=`log1p ${xb} ${hexout(xlog1p(x))}\n`;
  out+=`expm1 ${xb} ${hexout(xexpm1(x))}\n`;
  out+=`tanh ${xb} ${hexout(xtanh(x))}\n`;
  out+=`sin ${xb} ${hexout(xsin(x))}\n`;
  out+=`cos ${xb} ${hexout(xcos(x))}\n`;
  out+=`atan ${xb} ${hexout(xatan(x))}\n`;
}
const powBases=[-Infinity,-Number.MAX_VALUE,-3.5,-1.0,fromBits(0x8010000000000000n),fromBits(0x8000000000000001n),
  -0.0,0.0,fromBits(1n),fromBits(0x0010000000000000n),0.5,1.0,2.0,3.5,Number.MAX_VALUE,Infinity,
  fromBits(0x7ff8000000000000n)];
const exps=[-Infinity,-8.0,-3.0,-2.5,-1.0,-0.5,-0.0,0.0,0.3333,0.5,1.0,2.0,2.5,3.0,8.0,Infinity,
  fromBits(0x7ff8000000000000n)];
for(const b of powBases) for(const e of exps){ out+=`pow ${toBits(b).toString(16).padStart(16,'0')} ${toBits(e).toString(16).padStart(16,'0')} ${hexout(xpow(b,e))}\n`; }
const ys=[-Infinity,-Number.MAX_VALUE,-3.0,-1.0,fromBits(0x8010000000000000n),fromBits(0x8000000000000001n),
  -0.0,0.0,fromBits(1n),fromBits(0x0010000000000000n),1.0,fromBits(0x3ff0000080000000n),3.0,Number.MAX_VALUE,Infinity,
  fromBits(0x7ff8000000000000n)], xs=ys;
for(const yy of ys) for(const xx of xs){ out+=`atan2 ${toBits(yy).toString(16).padStart(16,'0')} ${toBits(xx).toString(16).padStart(16,'0')} ${hexout(xatan2(yy,xx))}\n`; }
process.stdout.write(out);
