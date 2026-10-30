// C++ all-function port of portable_math.rs. Byte-identical to full_ref.rs.
// g++ -O3 -ffp-contract=off -std=c++17 full_ref.cpp -o full_ref_cpp
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <string>

static inline uint64_t to_bits(double x){ uint64_t b; std::memcpy(&b,&x,8); return b; }
static inline double from_bits(uint64_t b){ double x; std::memcpy(&x,&b,8); return x; }
static inline uint32_t high_word(double x){ return (uint32_t)(to_bits(x) >> 32); }
static inline uint32_t low_word(double x){ return (uint32_t)to_bits(x); }
static inline double with_high_word(double x, uint32_t hi){ return from_bits(((uint64_t)hi<<32)|(to_bits(x)&0xffffffffULL)); }
static inline double with_low_word(double x, uint32_t lo){ return from_bits((to_bits(x)&0xffffffff00000000ULL)|(uint64_t)lo); }
static inline double fabs_(double x){ return from_bits(to_bits(x)&0x7fffffffffffffffULL); }
static inline double canonical_nan(){ return from_bits(0x7ff8000000000000ULL); }
static inline double two_pow(int32_t n){ return from_bits((uint64_t)((int64_t)(0x3ff+n))<<52); }

static double p_scalbn(double x, int32_t n){
    double p1023=two_pow(1023), pm1022=two_pow(-1022), p53=two_pow(53); double y=x;
    if (n>1023){ y*=p1023; n-=1023; if(n>1023){ y*=p1023; n-=1023; if(n>1023) n=1023; } }
    else if (n<-1022){ y*=pm1022*p53; n+=1022-53; if(n<-1022){ y*=pm1022*p53; n+=1022-53; if(n<-1022) n=-1022; } }
    return y*two_pow(n);
}

static double xexp(double x){
    if (x!=x) return canonical_nan();
    const double HALF[2]={0.5,-0.5};
    const double LN2HI[2]={6.9314718036912381649e-1,-6.9314718036912381649e-1};
    const double LN2LO[2]={1.9082149292705877000e-10,-1.9082149292705877000e-10};
    const double INVLN2=1.4426950408889633870;
    const double P1=1.66666666666666019037e-1,P2=-2.77777777770155933842e-3,P3=6.61375632143793436117e-5,P4=-1.65339022054652515390e-6,P5=4.13813679705723846039e-8;
    const double HUGE=1.0e300,TWOM1000=9.33263618503218878990e-302,O_THRESHOLD=7.09782712893383973096e2,U_THRESHOLD=-7.45133219101941108420e2;
    uint32_t hx=high_word(x); size_t xsb=(hx>>31)&1; hx&=0x7fffffff;
    if (hx>=0x40862E42){
        if (hx>=0x7ff00000){ if(((hx&0xfffff)|low_word(x))!=0) return x+x; return xsb==0?x:0.0; }
        if (x>O_THRESHOLD) return HUGE*HUGE;
        if (x<U_THRESHOLD) return TWOM1000*TWOM1000;
    }
    int32_t k=0; double hi=0.0,lo=0.0,xr;
    if (hx>0x3fd62e42){
        if (hx<0x3FF0A2B2){ hi=x-LN2HI[xsb]; lo=LN2LO[xsb]; k=1-(int32_t)xsb-(int32_t)xsb; }
        else { k=(int32_t)(INVLN2*x+HALF[xsb]); double t=(double)k; hi=x-t*LN2HI[0]; lo=t*LN2LO[0]; }
        xr=hi-lo;
    } else if (hx<0x3e300000){ return 1.0+x; } else { xr=x; }
    double t=xr*xr; double c=xr-t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
    if (k==0) return 1.0-((xr*c)/(c-2.0)-xr);
    double y=1.0-((lo-(xr*c)/(2.0-c))-hi);
    return p_scalbn(y,k);
}

static double xln(double x){
    if (x!=x) return canonical_nan();
    const double LN2_HI=6.9314718036912381649e-1,LN2_LO=1.9082149292705877000e-10,TWO54=1.80143985094819840000e16;
    const double LG1=6.666666666666735130e-1,LG2=3.999999999940941908e-1,LG3=2.857142874366239149e-1,LG4=2.222219843214978396e-1,LG5=1.818357216161805012e-1,LG6=1.531383769920937332e-1,LG7=1.479819860511658591e-1;
    int32_t hx=(int32_t)high_word(x); uint32_t lx=low_word(x); int32_t k=0; double xx=x;
    if (hx<0x00100000){
        if (((hx&0x7fffffff)|(int32_t)lx)==0) return -TWO54/0.0;
        if (hx<0) return canonical_nan();
        k-=54; xx*=TWO54; hx=(int32_t)high_word(xx);
    }
    if (hx>=0x7ff00000) return x+x;
    k+=(hx>>20)-1023; hx&=0x000fffff;
    int32_t i=(hx+0x95f64)&0x100000;
    xx=with_high_word(xx,(uint32_t)(hx|(i^0x3ff00000))); k+=i>>20;
    double f=xx-1.0;
    if ((0x000fffff&(2+hx))<3){
        if (f==0.0){ if(k==0) return 0.0; double dk=(double)k; return dk*LN2_HI+dk*LN2_LO; }
        double r=f*f*(0.5-0.33333333333333333*f);
        if (k==0) return f-r;
        double dk=(double)k; return dk*LN2_HI-((r-dk*LN2_LO)-f);
    }
    double s=f/(2.0+f); double dk=(double)k; double z=s*s,w=z*z;
    double t1=w*(LG2+w*(LG4+w*LG6)); double t2=z*(LG1+w*(LG3+w*(LG5+w*LG7))); double r=t2+t1;
    int32_t i2=hx-0x6147a, j2=0x6b851-hx;
    if ((i2|j2)>0){ double hfsq=0.5*f*f; if(k==0) return f-(hfsq-s*(hfsq+r)); return dk*LN2_HI-((hfsq-(s*(hfsq+r)+dk*LN2_LO))-f); }
    else if (k==0) return f-s*(f-r);
    return dk*LN2_HI-((s*(f-r)-dk*LN2_LO)-f);
}

static double xlog1p(double x){
    if (x!=x) return canonical_nan();
    const double LN2_HI=6.9314718036912381649e-1,LN2_LO=1.9082149292705877000e-10;
    const double LG1=6.666666666666735130e-1,LG2=3.999999999940941908e-1,LG3=2.857142874366239149e-1,LG4=2.222219843214978396e-1,LG5=1.818357216161805012e-1,LG6=1.531383769920937332e-1,LG7=1.479819860511658591e-1;
    uint32_t hx=high_word(x); int32_t k=1; double f=0.0,c=0.0;
    if (hx<0x3fda827a || (hx>>31)!=0){
        if (hx>=0xbff00000){ if(x==-1.0) return x/0.0; return canonical_nan(); }
        if ((hx<<1) < (0x3ca00000u<<1)) return x;
        if (hx<=0xbfd2bec4){ k=0; c=0.0; f=x; }
    } else if (hx>=0x7ff00000){ return x; }
    if (k!=0){
        double u=1.0+x; uint32_t hu=high_word(u);
        hu=hu+(0x3ff00000u-0x3fe6a09eu);
        k=(int32_t)(hu>>20)-0x3ff;
        if (k<54){ c=(k>=2)?(1.0-(u-x)):(x-(u-1.0)); c/=u; } else c=0.0;
        hu=(hu&0x000fffff)+0x3fe6a09eu;
        f=with_high_word(u,hu)-1.0;
    }
    double hfsq=0.5*f*f, s=f/(2.0+f), z=s*s, w=z*z;
    double t1=w*(LG2+w*(LG4+w*LG6)); double t2=z*(LG1+w*(LG3+w*(LG5+w*LG7))); double r=t2+t1;
    double dk=(double)k;
    return s*(hfsq+r)+(dk*LN2_LO+c)-hfsq+f+dk*LN2_HI;
}

static double xexpm1(double x){
    if (x!=x) return canonical_nan();
    const double LN2_HI=6.9314718036912381649e-1,LN2_LO=1.90821492927058770002e-10,INVLN2=1.4426950408889633870;
    const double Q1=-3.33333333333331315428e-2,Q2=1.58730158725481460165e-3,Q3=-7.93650757867487942473e-5,Q4=4.00821782732936239552e-6,Q5=-2.01099218183624371326e-7;
    uint32_t hx=high_word(x)&0x7fffffff; bool sign=(high_word(x)>>31)!=0;
    double c=0.0; int32_t k; double xx=x;
    if (hx>=0x4043687A){ if(sign) return -1.0; if(hx>=0x40862E42) return x*two_pow(1023); }
    if (hx>0x3fd62e42){
        double hi,lo;
        if (hx<0x3FF0A2B2){ if(!sign){ hi=x-LN2_HI; lo=LN2_LO; k=1; } else { hi=x+LN2_HI; lo=-LN2_LO; k=-1; } }
        else { k=(int32_t)(INVLN2*x+(sign?-0.5:0.5)); double t=(double)k; hi=x-t*LN2_HI; lo=t*LN2_LO; }
        xx=hi-lo; c=(hi-xx)-lo;
    } else if (hx<0x3c900000){ return x; } else { k=0; }
    double hfx=0.5*xx, hxs=xx*hfx;
    double r1=1.0+hxs*(Q1+hxs*(Q2+hxs*(Q3+hxs*(Q4+hxs*Q5))));
    double t=3.0-r1*hfx; double e=hxs*((r1-t)/(6.0-xx*t));
    if (k==0) return xx-(xx*e-hxs);
    e=xx*(e-c)-c; e-=hxs;
    if (k==-1) return 0.5*(xx-e)-0.5;
    if (k==1){ if(xx<-0.25) return -2.0*(e-(xx+0.5)); return 1.0+2.0*(xx-e); }
    double twopk=two_pow(k);
    if (!(k>=0 && k<=56)){ double y=(xx-e+1.0)*twopk; return y-1.0; }
    double inv=two_pow(-k);
    if (k<20) return (xx-e+(1.0-inv))*twopk;
    return (xx-e-inv+1.0)*twopk;
}

static double xpow(double x, double y){
    const double BP[2]={1.0,1.5}, DP_H[2]={0.0,5.84962487220764160156e-1}, DP_L[2]={0.0,1.35003920212974897128e-8};
    const double TWO53=9007199254740992.0, HUGE=1.0e300, TINY=1.0e-300;
    const double L1=5.99999999999994648725e-1,L2=4.28571428578550184252e-1,L3=3.33333329818377432918e-1,L4=2.72728123808534006489e-1,L5=2.30660745775561366331e-1,L6=2.06975017800338417784e-1;
    const double P1=1.66666666666666019037e-1,P2=-2.77777777770155933842e-3,P3=6.61375632143793436117e-5,P4=-1.65339022054652515390e-6,P5=4.13813679705723846039e-8;
    const double LG2=6.93147180559945286227e-1,LG2_H=6.93147182464599609375e-1,LG2_L=-1.90465429995776804525e-9,OVT=8.0085662595372944372e-17;
    const double CP=9.61796693925975554329e-1,CP_H=9.61796700954437255859e-1,CP_L=-7.02846165095275826516e-9;
    const double IVLN2=1.4426950408889633870,IVLN2_H=1.44269502162933349609e0,IVLN2_L=1.92596299112661746887e-8;
    int32_t hx=(int32_t)high_word(x); uint32_t lx=low_word(x); int32_t hy=(int32_t)high_word(y); uint32_t ly=low_word(y);
    int32_t ix=hx&0x7fffffff, iy=hy&0x7fffffff;
    if ((iy|(int32_t)ly)==0) return 1.0;
    if (x!=x||y!=y) return canonical_nan();
    if (ix>0x7ff00000||(ix==0x7ff00000&&lx!=0)||iy>0x7ff00000||(iy==0x7ff00000&&ly!=0)) return canonical_nan();
    int32_t yisint=0;
    if (hx<0){
        if (iy>=0x43400000) yisint=2;
        else if (iy>=0x3ff00000){ int32_t k=(iy>>20)-0x3ff;
            if (k>20){ uint32_t j=ly>>(52-k); if((j<<(52-k))==ly) yisint=2-(int32_t)(j&1); }
            else if (ly==0){ int32_t j=iy>>(20-k); if((j<<(20-k))==iy) yisint=2-(j&1); } }
    }
    if (ly==0){
        if (iy==0x7ff00000){ if(((ix-0x3ff00000)|(int32_t)lx)==0) return 1.0; else if(ix>=0x3ff00000) return hy>=0?y:0.0; else return hy<0?-y:0.0; }
        if (iy==0x3ff00000) return hy<0?1.0/x:x;
        if (hy==0x40000000) return x*x;
        if (hy==0x3fe00000 && hx>=0) return std::sqrt(x);
    }
    double ax=fabs_(x);
    if (lx==0 && (ix==0x7ff00000||ix==0||ix==0x3ff00000)){
        double z=ax; if(hy<0) z=1.0/z;
        if (hx<0){ if(((ix-0x3ff00000)|yisint)==0) z=canonical_nan(); else if(yisint==1) z=-z; }
        return z;
    }
    int32_t n=(hx>>31)+1;
    if ((n|yisint)==0) return canonical_nan();
    double s=1.0; if((n|(yisint-1))==0) s=-1.0;
    double t1,t2;
    if (iy>0x41e00000){
        if (iy>0x43f00000){ if(ix<=0x3fefffff) return hy<0?HUGE*HUGE:TINY*TINY; if(ix>=0x3ff00000) return hy>0?HUGE*HUGE:TINY*TINY; }
        if (ix<0x3fefffff) return hy<0?s*HUGE*HUGE:s*TINY*TINY;
        if (ix>0x3ff00000) return hy>0?s*HUGE*HUGE:s*TINY*TINY;
        double t=ax-1.0; double w=(t*t)*(0.5-t*(0.333333333333333333333-t*0.25));
        double u=IVLN2_H*t; double v=t*IVLN2_L-w*IVLN2; double t1v=with_low_word(u+v,0); t1=t1v; t2=v-(t1v-u);
    } else {
        n=0;
        if (ix<0x00100000){ ax*=TWO53; n-=53; ix=(int32_t)high_word(ax); }
        n+=(ix>>20)-0x3ff; int32_t j0=ix&0x000fffff; ix=j0|0x3ff00000; int32_t k;
        if (j0<=0x3988E) k=0; else if(j0<0xBB67A) k=1; else { k=0; n+=1; ix-=0x00100000; }
        ax=with_high_word(ax,(uint32_t)ix);
        double u=ax-BP[k]; double v=1.0/(ax+BP[k]); double ss=u*v; double s_h=with_low_word(ss,0);
        double t_h=with_high_word(0.0,(uint32_t)(((ix>>1)|0x20000000)+0x00080000+(k<<18)));
        double t_l=ax-(t_h-BP[k]); double s_l=v*((u-s_h*t_h)-s_h*t_l);
        double s2=ss*ss; double r=s2*s2*(L1+s2*(L2+s2*(L3+s2*(L4+s2*(L5+s2*L6))))); r+=s_l*(s_h+ss);
        double s2b=s_h*s_h; double t_h2=with_low_word(3.0+s2b+r,0); double t_l2=r-((t_h2-3.0)-s2b);
        double u2=s_h*t_h2; double v2=s_l*t_h2+t_l2*ss; double p_h=with_low_word(u2+v2,0); double p_l=v2-(p_h-u2);
        double z_h=CP_H*p_h; double z_l=CP_L*p_h+p_l*CP+DP_L[k];
        double t=(double)n; double t1v=with_low_word(((z_h+z_l)+DP_H[k])+t,0); t1=t1v; t2=z_l-(((t1v-t)-DP_H[k])-z_h);
    }
    double y1=with_low_word(y,0); double p_l=(y-y1)*t1+y*t2; double p_h=y1*t1; double z=p_l+p_h;
    int32_t j=(int32_t)high_word(z); uint32_t i=low_word(z);
    if (j>=0x40900000){ if(((j-0x40900000)|(int32_t)i)!=0) return s*HUGE*HUGE; if(p_l+OVT>z-p_h) return s*HUGE*HUGE; }
    else if ((j&0x7fffffff)>=0x4090cc00){ if(((int32_t)((uint32_t)j-0xc090cc00u)|(int32_t)i)!=0) return s*TINY*TINY; if(p_l<=z-p_h) return s*TINY*TINY; }
    int32_t iabs=j&0x7fffffff; int32_t k=(iabs>>20)-0x3ff; int32_t nn=0;
    if (iabs>0x3fe00000){
        nn=j+(0x00100000>>(k+1)); k=((nn&0x7fffffff)>>20)-0x3ff;
        double t=with_high_word(0.0,(uint32_t)(nn&~(0x000fffff>>k)));
        nn=((nn&0x000fffff)|0x00100000)>>(20-k); if(j<0) nn=-nn; p_h-=t;
    }
    double t=with_low_word(p_l+p_h,0); double u=t*LG2_H; double v=(p_l-(t-p_h))*LG2+t*LG2_L;
    double z2=u+v; double w=v-(z2-u); double tt=z2*z2; double t1b=z2-tt*(P1+tt*(P2+tt*(P3+tt*(P4+tt*P5))));
    double r=(z2*t1b)/(t1b-2.0)-(w+z2*w); z2=1.0-(r-z2);
    j=(int32_t)((uint32_t)high_word(z2)+((uint32_t)nn<<20));
    if ((j>>20)<=0) z2=p_scalbn(z2,nn); else z2=with_high_word(z2,(uint32_t)j);
    return s*z2;
}

static double xtanh(double x){
    if (x!=x) return canonical_nan();
    bool sign=(to_bits(x)>>63)!=0; double ax=from_bits(to_bits(x)&0x7fffffffffffffffULL);
    uint32_t w=(uint32_t)(to_bits(ax)>>32); double t;
    if (w>0x3fe193ea){ if(w>0x40340000){ t=1.0-0.0/ax; } else { double e=xexpm1(2.0*ax); t=1.0-2.0/(e+2.0); } }
    else if (w>0x3fd058ae){ double e=xexpm1(2.0*ax); t=e/(e+2.0); }
    else if (w>=0x00100000){ double e=xexpm1(-2.0*ax); t=-e/(e+2.0); }
    else { t=ax; }
    return sign?-t:t;
}

static double kernel_sin(double x, double y, int32_t iy){
    const double S1=-1.66666666666666324348e-1,S2=8.33333333332248946124e-3,S3=-1.98412698298579493134e-4,S4=2.75573137070700676789e-6,S5=-2.50507602534068634195e-8,S6=1.58969099521155010221e-10;
    double z=x*x,w=z*z; double r=S2+z*(S3+z*S4)+z*w*(S5+z*S6); double v=z*x;
    if (iy==0) return x+v*(S1+z*r);
    return x-((z*(0.5*y-v*r)-y)-v*S1);
}
static double kernel_cos(double x, double y){
    const double C1=4.16666666666666019037e-2,C2=-1.38888888888741095749e-3,C3=2.48015872894767294178e-5,C4=-2.75573143513906633035e-7,C5=2.08757232129817482790e-9,C6=-1.13596475577881948265e-11;
    double z=x*x,w=z*z; double r=z*(C1+z*(C2+z*C3))+w*w*(C4+z*(C5+z*C6)); double hz=0.5*z; double w2=1.0-hz;
    return w2+(((1.0-w2)-hz)+(z*r-x*y));
}
static void rem_pio2(double x, int32_t* nout, double* y0out, double* y1out){
    const double INVPIO2=6.36619772367581382433e-1,PIO2_1=1.57079632673412561417e0,PIO2_1T=6.07710050650619224932e-11,PIO2_2=6.07710050630396597660e-11,PIO2_2T=2.02226624879595063154e-21,PIO2_3=2.02226624871116645580e-21,PIO2_3T=8.47842766036889956997e-32;
    const double INV_TWO_PI=1.59154943091895335769e-1,TWO_PI_HI=6.28318530717958623199e0,TWO_PI_LO=2.44929359829470641471e-16;
    uint32_t ix=high_word(x)&0x7fffffff;
    if (ix>=0x7ff00000){ double z=canonical_nan(); *nout=0; *y0out=z; *y1out=z; return; }
    double xr=x;
    while ((high_word(xr)&0x7fffffff)>=0x413921fb){ double k=std::round(xr*INV_TWO_PI); xr=(xr-k*TWO_PI_HI)-k*TWO_PI_LO; }
    uint32_t ixr=high_word(xr)&0x7fffffff;
    if (ixr<=0x3fe921fb){ *nout=0; *y0out=xr; *y1out=0.0; return; }
    double half=xr>0.0?0.5:-0.5; int32_t n=(int32_t)(xr*INVPIO2+half); double fnn=(double)n;
    double r=xr-fnn*PIO2_1; double w=fnn*PIO2_1T; int32_t j=(int32_t)(ixr>>20);
    double y0=r-w; int32_t hi=(int32_t)(high_word(y0)>>20)&0x7ff; int32_t i=j-hi;
    if (i>16){ double t=r; w=fnn*PIO2_2; r=t-w; w=fnn*PIO2_2T-((t-r)-w); y0=r-w; hi=(int32_t)(high_word(y0)>>20)&0x7ff; i=j-hi;
        if (i>49){ double t2=r; w=fnn*PIO2_3; r=t2-w; w=fnn*PIO2_3T-((t2-r)-w); y0=r-w; } }
    double y1=(r-y0)-w; *nout=n&3; *y0out=y0; *y1out=y1;
}
static double xsin(double x){
    if (x!=x) return canonical_nan();
    uint32_t ix=high_word(x)&0x7fffffff;
    if (ix<=0x3fe921fb){ if(ix<0x3e500000) return x; return kernel_sin(x,0.0,0); }
    if (ix>=0x7ff00000) return canonical_nan();
    int32_t n; double y0,y1; rem_pio2(x,&n,&y0,&y1);
    switch(n){ case 0: return kernel_sin(y0,y1,1); case 1: return kernel_cos(y0,y1); case 2: return -kernel_sin(y0,y1,1); default: return -kernel_cos(y0,y1); }
}
static double xcos(double x){
    if (x!=x) return canonical_nan();
    uint32_t ix=high_word(x)&0x7fffffff;
    if (ix<=0x3fe921fb){ if(ix<0x3e46a09e) return 1.0; return kernel_cos(x,0.0); }
    if (ix>=0x7ff00000) return canonical_nan();
    int32_t n; double y0,y1; rem_pio2(x,&n,&y0,&y1);
    switch(n){ case 0: return kernel_cos(y0,y1); case 1: return -kernel_sin(y0,y1,1); case 2: return -kernel_cos(y0,y1); default: return kernel_sin(y0,y1,1); }
}
static double xatan(double x){
    if (x!=x) return canonical_nan();
    const double ATANHI[4]={4.63647609000806093515e-1,7.85398163397448278999e-1,9.82793723247329054082e-1,1.57079632679489655800e0};
    const double ATANLO[4]={2.26987774529616870924e-17,3.06161699786838301793e-17,1.39033110312309984516e-17,6.12323399573676603587e-17};
    const double AT[11]={3.33333333333329318027e-1,-1.99999999998764832476e-1,1.42857142725034663711e-1,-1.11111104054623557880e-1,9.09088713343650656196e-2,-7.69187620504482999495e-2,6.66107313738753120669e-2,-5.83357013379057348645e-2,4.97687799461593236017e-2,-3.65315727442169155270e-2,1.62858201153657823623e-2};
    int32_t hx=(int32_t)high_word(x); int32_t ix=hx&0x7fffffff;
    if (ix>=0x44100000){ if(ix>0x7ff00000||(ix==0x7ff00000&&low_word(x)!=0)) return canonical_nan(); return hx>0?ATANHI[3]+ATANLO[3]:-ATANHI[3]-ATANLO[3]; }
    int32_t id; double xr=x;
    if (ix<0x3fdc0000){ if(ix<0x3e400000) return x; id=-1; }
    else { xr=fabs_(x);
        if (ix<0x3ff30000){ if(ix<0x3fe60000){ id=0; xr=(2.0*xr-1.0)/(2.0+xr); } else { id=1; xr=(xr-1.0)/(xr+1.0); } }
        else if (ix<0x40038000){ id=2; xr=(xr-1.5)/(1.0+1.5*xr); } else { id=3; xr=-1.0/xr; } }
    double z=xr*xr,w=z*z;
    double s1=z*(AT[0]+w*(AT[2]+w*(AT[4]+w*(AT[6]+w*(AT[8]+w*AT[10])))));
    double s2=w*(AT[1]+w*(AT[3]+w*(AT[5]+w*(AT[7]+w*AT[9]))));
    if (id<0) return xr-xr*(s1+s2);
    double z2=ATANHI[id]-((xr*(s1+s2)-ATANLO[id])-xr);
    return hx<0?-z2:z2;
}
static double xatan2(double y, double x){
    if (x!=x||y!=y) return canonical_nan();
    const double TINY=1.0e-300,PI_O_4=7.85398163397448278990e-1,PI_O_2=1.57079632679489655800e0,PI=3.14159265358979311600e0,PI_LO=1.22464679914735317722e-16;
    int32_t hx=(int32_t)high_word(x); int32_t ix=hx&0x7fffffff; uint32_t lx=low_word(x);
    int32_t hy=(int32_t)high_word(y); int32_t iy=hy&0x7fffffff; uint32_t ly=low_word(y);
    if ((ix|(int32_t)((lx|(0u-lx))>>31))>0x7ff00000 || (iy|(int32_t)((ly|(0u-ly))>>31))>0x7ff00000) return canonical_nan();
    if (hx==0x3ff00000 && lx==0) return xatan(y);
    int32_t m=((hy>>31)&1)|((hx>>30)&2);
    if ((iy|(int32_t)ly)==0){ switch(m){ case 0: case 1: return y; case 2: return PI+TINY; default: return -PI-TINY; } }
    if ((ix|(int32_t)lx)==0) return hy<0?-PI_O_2-TINY:PI_O_2+TINY;
    if (ix==0x7ff00000){
        if (iy==0x7ff00000){ switch(m){ case 0: return PI_O_4+TINY; case 1: return -PI_O_4-TINY; case 2: return 3.0*PI_O_4+TINY; default: return -3.0*PI_O_4-TINY; } }
        switch(m){ case 0: return 0.0; case 1: return -0.0; case 2: return PI+TINY; default: return -PI-TINY; }
    }
    if (iy==0x7ff00000) return hy<0?-PI_O_2-TINY:PI_O_2+TINY;
    int32_t k=(iy-ix)>>20; double z;
    if (k>60) z=PI_O_2+0.5*PI_LO; else if(hx<0&&k<-60) z=0.0; else z=xatan(fabs_(y/x));
    switch(m){ case 0: return z; case 1: return -z; case 2: return PI-(z-PI_LO); default: return (z-PI_LO)-PI; }
}

static std::vector<double> grid(){
    std::vector<double> g;
    for (int32_t i=-900;i<=900;i++) g.push_back((double)i*0.41);
    for (int e=-55;e<55;e++){ g.push_back(two_pow(e)); g.push_back(-two_pow(e)); g.push_back(1.0+two_pow(e)); }
    double sp[]={0.0,-0.0,1.0,-1.0,3.5,-3.5,709.0,-745.0,710.0,from_bits(1),from_bits(0x000fffffffffffffULL),
        std::numeric_limits<double>::min(),-std::numeric_limits<double>::min(),1e300,1e-300,
        std::numeric_limits<double>::max(),std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
        from_bits(0x7ff8000000000000ULL),from_bits(0xfff8000000001234ULL)};
    for (double v: sp) g.push_back(v);
    return g;
}
static std::string hexout(double v){ char b[32]; snprintf(b,sizeof b,"%016llx",(unsigned long long)to_bits(v)); return b; }

int main(){
    std::vector<double> g=grid(); std::string out; char buf[96];
    for (double x: g){ unsigned long long xb=(unsigned long long)to_bits(x);
        snprintf(buf,sizeof buf,"exp %016llx %s\n",xb,hexout(xexp(x)).c_str()); out+=buf;
        snprintf(buf,sizeof buf,"ln %016llx %s\n",xb,hexout(xln(x)).c_str()); out+=buf;
        snprintf(buf,sizeof buf,"log1p %016llx %s\n",xb,hexout(xlog1p(x)).c_str()); out+=buf;
        snprintf(buf,sizeof buf,"expm1 %016llx %s\n",xb,hexout(xexpm1(x)).c_str()); out+=buf;
        snprintf(buf,sizeof buf,"tanh %016llx %s\n",xb,hexout(xtanh(x)).c_str()); out+=buf;
        snprintf(buf,sizeof buf,"sin %016llx %s\n",xb,hexout(xsin(x)).c_str()); out+=buf;
        snprintf(buf,sizeof buf,"cos %016llx %s\n",xb,hexout(xcos(x)).c_str()); out+=buf;
        snprintf(buf,sizeof buf,"atan %016llx %s\n",xb,hexout(xatan(x)).c_str()); out+=buf;
    }
    double pow_bases[]={-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::lowest(),-3.5,-1.0,
        -std::numeric_limits<double>::min(),from_bits(0x8000000000000001ULL),-0.0,0.0,from_bits(1),
        std::numeric_limits<double>::min(),0.5,1.0,2.0,3.5,std::numeric_limits<double>::max(),
        std::numeric_limits<double>::infinity(),from_bits(0x7ff8000000000000ULL)};
    double exps[]={-std::numeric_limits<double>::infinity(),-8.0,-3.0,-2.5,-1.0,-0.5,-0.0,0.0,0.3333,0.5,
        1.0,2.0,2.5,3.0,8.0,std::numeric_limits<double>::infinity(),from_bits(0x7ff8000000000000ULL)};
    for (double b: pow_bases){
        for (double e: exps){ snprintf(buf,sizeof buf,"pow %016llx %016llx %s\n",(unsigned long long)to_bits(b),(unsigned long long)to_bits(e),hexout(xpow(b,e)).c_str()); out+=buf; } }
    double ys[]={-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::lowest(),-3.0,-1.0,
        -std::numeric_limits<double>::min(),from_bits(0x8000000000000001ULL),-0.0,0.0,from_bits(1),
        std::numeric_limits<double>::min(),1.0,from_bits(0x3ff0000080000000ULL),3.0,std::numeric_limits<double>::max(),
        std::numeric_limits<double>::infinity(),from_bits(0x7ff8000000000000ULL)};
    double xs[sizeof(ys)/sizeof(ys[0])]; std::memcpy(xs,ys,sizeof ys);
    for (double yy: ys) for (double xx: xs){ snprintf(buf,sizeof buf,"atan2 %016llx %016llx %s\n",(unsigned long long)to_bits(yy),(unsigned long long)to_bits(xx),hexout(xatan2(yy,xx)).c_str()); out+=buf; }
    fputs(out.c_str(),stdout); return 0;
}
