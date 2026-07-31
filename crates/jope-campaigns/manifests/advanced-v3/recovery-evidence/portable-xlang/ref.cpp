// C++ port of the portable exp/ln kernel. Must print byte-identical output to
// ref.rs. Compile: g++ -O3 -ffp-contract=off -std=c++17 ref.cpp -o ref_cpp
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <limits>
#include <string>

static inline uint64_t to_bits(double x){ uint64_t b; std::memcpy(&b,&x,8); return b; }
static inline double from_bits(uint64_t b){ double x; std::memcpy(&x,&b,8); return x; }
static inline uint32_t high_word(double x){ return (uint32_t)(to_bits(x) >> 32); }
static inline uint32_t low_word(double x){ return (uint32_t)to_bits(x); }
static inline double with_high_word(double x, uint32_t hi){
    return from_bits(((uint64_t)hi << 32) | (to_bits(x) & 0xffffffffULL));
}
static inline double two_pow(int32_t n){ return from_bits((uint64_t)((int64_t)(0x3ff + n)) << 52); }

static double scalbn(double x, int32_t n){
    double p1023 = two_pow(1023), pm1022 = two_pow(-1022), p53 = two_pow(53);
    double y = x;
    if (n > 1023){ y *= p1023; n -= 1023; if (n > 1023){ y *= p1023; n -= 1023; if (n > 1023) n = 1023; } }
    else if (n < -1022){ y *= pm1022*p53; n += 1022-53; if (n < -1022){ y *= pm1022*p53; n += 1022-53; if (n < -1022) n = -1022; } }
    return y * two_pow(n);
}

static double xexp(double x){
    const double HALF[2] = {0.5,-0.5};
    const double LN2HI[2] = {6.9314718036912381649e-1,-6.9314718036912381649e-1};
    const double LN2LO[2] = {1.9082149292705877000e-10,-1.9082149292705877000e-10};
    const double INVLN2=1.4426950408889633870;
    const double P1=1.66666666666666019037e-1,P2=-2.77777777770155933842e-3,P3=6.61375632143793436117e-5,
                 P4=-1.65339022054652515390e-6,P5=4.13813679705723846039e-8;
    const double HUGE=1.0e300,TWOM1000=9.33263618503218878990e-302,
                 O_THRESHOLD=7.09782712893383973096e2,U_THRESHOLD=-7.45133219101941108420e2;
    uint32_t hx = high_word(x);
    size_t xsb = (hx>>31)&1;
    hx &= 0x7fffffff;
    if (hx >= 0x40862E42){
        if (hx >= 0x7ff00000){
            if (((hx & 0xfffff) | low_word(x)) != 0) return x+x;
            return xsb==0 ? x : 0.0;
        }
        if (x > O_THRESHOLD) return HUGE*HUGE;
        if (x < U_THRESHOLD) return TWOM1000*TWOM1000;
    }
    int32_t k=0; double hi=0.0, lo=0.0, xr;
    if (hx > 0x3fd62e42){
        if (hx < 0x3FF0A2B2){ hi = x-LN2HI[xsb]; lo=LN2LO[xsb]; k = 1-(int32_t)xsb-(int32_t)xsb; }
        else { k = (int32_t)(INVLN2*x + HALF[xsb]); double t=(double)k; hi = x-t*LN2HI[0]; lo = t*LN2LO[0]; }
        xr = hi-lo;
    } else if (hx < 0x3e300000){ return 1.0+x; } else { xr = x; }
    double t = xr*xr;
    double c = xr - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
    if (k==0) return 1.0 - ((xr*c)/(c-2.0) - xr);
    double y = 1.0 - ((lo - (xr*c)/(2.0-c)) - hi);
    return scalbn(y,k);
}

static double xln(double x){
    const double LN2_HI=6.9314718036912381649e-1,LN2_LO=1.9082149292705877000e-10,TWO54=1.80143985094819840000e16;
    const double LG1=6.666666666666735130e-1,LG2=3.999999999940941908e-1,LG3=2.857142874366239149e-1,
                 LG4=2.222219843214978396e-1,LG5=1.818357216161805012e-1,LG6=1.531383769920937332e-1,LG7=1.479819860511658591e-1;
    int32_t hx = (int32_t)high_word(x);
    uint32_t lx = low_word(x);
    int32_t k=0; double xx=x;
    if (hx < 0x00100000){
        if (((hx & 0x7fffffff) | (int32_t)lx) == 0) return -TWO54/0.0;
        if (hx < 0) return (x-x)/0.0;
        k -= 54; xx *= TWO54; hx = (int32_t)high_word(xx);
    }
    if (hx >= 0x7ff00000) return x+x;
    k += (hx>>20) - 1023;
    hx &= 0x000fffff;
    int32_t i = (hx + 0x95f64) & 0x100000;
    xx = with_high_word(xx, (uint32_t)(hx | (i ^ 0x3ff00000)));
    k += i>>20;
    double f = xx - 1.0;
    if ((0x000fffff & (2 + hx)) < 3){
        if (f == 0.0){ if (k==0) return 0.0; double dk=(double)k; return dk*LN2_HI + dk*LN2_LO; }
        double r = f*f*(0.5 - 0.33333333333333333*f);
        if (k==0) return f-r;
        double dk=(double)k; return dk*LN2_HI - ((r - dk*LN2_LO) - f);
    }
    double s = f/(2.0+f);
    double dk=(double)k;
    double z = s*s, w = z*z;
    double t1 = w*(LG2+w*(LG4+w*LG6));
    double t2 = z*(LG1+w*(LG3+w*(LG5+w*LG7)));
    double r = t2+t1;
    int32_t i2 = hx - 0x6147a;
    int32_t j2 = 0x6b851 - hx;
    if ((i2|j2) > 0){
        double hfsq = 0.5*f*f;
        if (k==0) return f-(hfsq - s*(hfsq+r));
        return dk*LN2_HI - ((hfsq - (s*(hfsq+r)+dk*LN2_LO)) - f);
    } else if (k==0) return f - s*(f-r);
    return dk*LN2_HI - ((s*(f-r) - dk*LN2_LO) - f);
}

static std::vector<double> grid(){
    std::vector<double> g;
    for (int32_t i=-1900;i<=1900;i++) g.push_back((double)i * 0.37);
    for (int e=-60;e<60;e++){ g.push_back(two_pow(e)); g.push_back(-two_pow(e)); g.push_back(1.0+two_pow(e)); }
    double specials[] = {0.0,-0.0,1.0,-1.0,709.0,-745.0,710.0,-746.0,
        from_bits(1),from_bits(0x000fffffffffffffULL),1e300,1e-300,
        std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()};
    for (double v : specials) g.push_back(v);
    return g;
}

static std::string hexout(double v){
    if (v != v) return "NaN"; // canonicalize NaN payloads
    char b[32]; snprintf(b,sizeof b,"%016llx",(unsigned long long)to_bits(v)); return b;
}

int main(){
    std::string out;
    char buf[80];
    for (double x : grid()){
        uint64_t xb = to_bits(x);
        snprintf(buf,sizeof buf,"exp %016llx %s\n",(unsigned long long)xb,hexout(xexp(x)).c_str());
        out += buf;
        snprintf(buf,sizeof buf,"ln %016llx %s\n",(unsigned long long)xb,hexout(xln(x)).c_str());
        out += buf;
    }
    fputs(out.c_str(), stdout);
    return 0;
}
