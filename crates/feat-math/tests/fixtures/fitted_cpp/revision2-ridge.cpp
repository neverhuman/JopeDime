// jope-dime fitted-cpp-typed-table-v1 emitter-revision=2
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
using V = std::vector<double>;
using U = std::vector<std::size_t>;
using M = std::vector<V>;
using B = std::vector<std::uint64_t>;
struct MathDomainError {};
[[maybe_unused]] static double f64_from_bits(std::uint64_t bits) noexcept { double value; std::memcpy(&value,&bits,sizeof(value)); return value; }
[[maybe_unused]] static double clampv(double value,double lo,double hi) noexcept { if(value<lo)return lo; if(value>hi)return hi; return value; }
[[maybe_unused]] static double finitev(double value) noexcept { return std::isfinite(value)?value:0.0; }
[[maybe_unused]] static double sigmoid(double value) noexcept { if(value>=0.0)return 1.0/(1.0+std::exp(-std::min(value,40.0))); double z=std::exp(std::max(value,-40.0)); return z/(1.0+z); }
// Match Rust slice::partition_point's fixed-iteration binary-search path.
// Fitted interpolated quantiles can cross by one ULP; preserving the search
// path is required to execute those immutable legacy states bit-for-bit.
[[maybe_unused]] static std::size_t lower(const V& values,double value) noexcept { if(values.empty())return 0;std::size_t size=values.size(),base=0;while(size>1){const std::size_t half=size/2,mid=base+half;if(values[mid]<value)base=mid;size-=half;}return base+static_cast<std::size_t>(values[base]<value); }
[[maybe_unused]] static std::size_t upper(const V& values,double value) noexcept { if(values.empty())return 0;std::size_t size=values.size(),base=0;while(size>1){const std::size_t half=size/2,mid=base+half;if(values[mid]<=value)base=mid;size-=half;}return base+static_cast<std::size_t>(values[base]<=value); }
[[maybe_unused]] static double mid_rank(const V& values,double value) noexcept { return static_cast<double>(lower(values,value)+upper(values,value))/(2.0*static_cast<double>(values.size())); }
[[maybe_unused]] static double probit(double p) noexcept {
  const V a={-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.38357751867269e+02,-3.066479806614716e+01,2.506628277459239e+00};
  const V b={-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01};
  const V c={-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
  const V e={7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00};
  p=clampv(p,1e-12,1.0-1e-12); const double p_low=0.02425;
  if(p<p_low){const double q=std::sqrt(-2.0*std::log(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((e[0]*q+e[1])*q+e[2])*q+e[3])*q+1.0);}
  if(p<=1.0-p_low){const double q=p-0.5,r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);}
  const double q=std::sqrt(-2.0*std::log(1.0-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((e[0]*q+e[1])*q+e[2])*q+e[3])*q+1.0);
}
[[maybe_unused]] static double dot(const V& a,const V& b) noexcept { double s=0.0; for(std::size_t i=0;i<std::min(a.size(),b.size());++i)s+=a[i]*b[i]; return s; }
[[maybe_unused]] static void finish_block(const V& input,const V& input_mask,V output,V output_mask,bool append,V& x,V& mask){ for(std::size_t i=0;i<output.size();++i)if(output_mask[i]!=0.0||!std::isfinite(output[i]))output[i]=0.0;if(append){x=input;mask=input_mask;x.insert(x.end(),output.begin(),output.end());mask.insert(mask.end(),output_mask.begin(),output_mask.end());}else{x=std::move(output);mask=std::move(output_mask);}if(x.size()>512){x.resize(512);mask.resize(512);} }
[[maybe_unused]] static std::uint64_t bits(double value) noexcept {std::uint64_t result;std::memcpy(&result,&value,8);return result;}
[[maybe_unused]] static double piece(double value,double lo,double hi) noexcept { if(!(hi>lo))return 0.0; return clampv((value-lo)/(hi-lo),0.0,1.0); }
[[maybe_unused]] static double yj(double value,double lambda) noexcept { if(value>=0.0){ if(std::abs(lambda)<1e-8)return std::log1p(value); return (std::pow(value+1.0,lambda)-1.0)/lambda; } double q=2.0-lambda; if(std::abs(q)<1e-8)return -std::log1p(-value); return -(std::pow(1.0-value,q)-1.0)/q; }
struct GNode { bool leaf; double value; std::size_t feature; std::uint16_t threshold; std::size_t left; std::size_t right; };
struct Block { V x; V mask; };
struct KCell { bool present; double value; };
using Key = std::vector<KCell>;
[[maybe_unused]] static bool operator==(const KCell& a,const KCell& b) noexcept {return a.present==b.present&&(!a.present||a.value==b.value);}
[[maybe_unused]] static bool operator<(const KCell& a,const KCell& b) noexcept {if(a.present!=b.present)return !a.present;if(!a.present)return false;if(a.value<b.value)return true;if(a.value>b.value)return false;return false;}
[[maybe_unused]] static bool key_less(const Key& a,const Key& b) noexcept {return std::lexicographical_compare(a.begin(),a.end(),b.begin(),b.end());}
[[maybe_unused]] static bool key_equal(const Key& a,const Key& b) noexcept {return a==b;}
[[maybe_unused]] static Key raw_key(const double* raw,const std::uint8_t* missing,const U& columns){Key key;key.reserve(columns.size());for(std::size_t column:columns)key.push_back(KCell{missing[column]==0,raw[column]});return key;}
[[maybe_unused]] static double route_gbm(const std::vector<GNode>& tree,const M& edges,const V& x) noexcept { std::size_t at=0; for(;;){const GNode& n=tree[at];if(n.leaf)return n.value;double value=n.feature<x.size()?x[n.feature]:0.0;std::size_t bin=lower(edges[n.feature],value);at=bin<=n.threshold?n.left:n.right;} }
[[maybe_unused]] static double rank_inverse(double value,const V& scores,const V& values) noexcept { if(scores.empty()||values.empty())return 0.0;if(!(value>scores[0]))return values[0];if(value>=scores.back())return values.back();std::size_t hi=lower(scores,value),lo=hi-1;double width=scores[hi]-scores[lo];return width>0.0?values[lo]+(value-scores[lo])/width*(values[hi]-values[lo]):values[lo]; }
[[maybe_unused]] static double quantile_linear(const V& sorted,double q) noexcept { if(sorted.empty())return 0.0;double position=q*static_cast<double>(sorted.size()-1);std::size_t lo=static_cast<std::size_t>(position),hi=std::min(lo+1,sorted.size()-1);double fraction=position-static_cast<double>(lo);return sorted[lo]*(1.0-fraction)+sorted[hi]*fraction; }
[[maybe_unused]] static V dft_mags(const V& row,std::size_t k){V out;out.reserve(k);double width=static_cast<double>(row.size());for(std::size_t f=1;f<=k;++f){double re=0.0,im=0.0;for(std::size_t j=0;j<row.size();++j){double angle=(2.0*3.14159265358979323846264338327950288)*static_cast<double>(f)*static_cast<double>(j)/width;re+=row[j]*std::cos(angle);im+=row[j]*std::sin(angle);}out.push_back(std::sqrt(re*re+im*im)/width);}return out;}
[[maybe_unused]] static V row_stats(const V& row){double n=static_cast<double>(row.size()),mean=0.0;for(double value:row)mean+=value;mean/=n;double m2=0.0,m3=0.0,m4=0.0,energy=0.0,mn=std::numeric_limits<double>::infinity(),mx=-std::numeric_limits<double>::infinity();for(double value:row){double delta=value-mean;m2+=delta*delta;m3+=delta*delta*delta;m4+=delta*delta*delta*delta;energy+=value*value;if(value<mn)mn=value;if(value>mx)mx=value;}m2/=n;m3/=n;m4/=n;energy/=n;double stddev=std::sqrt(m2),skew=m2>1e-24?m3/(m2*stddev):0.0,kurt=m2>1e-24?m4/(m2*m2)-3.0:0.0;V sorted=row;std::sort(sorted.begin(),sorted.end());return V{mean,stddev,mn,mx,quantile_linear(sorted,0.5),quantile_linear(sorted,0.25),quantile_linear(sorted,0.75),skew,kurt,energy};}
[[maybe_unused]] static V row_spectral(const V& row){std::size_t count=row.size()/2;if(count==0)return V(5,0.0);V mags=dft_mags(row,count);double total=0.0;for(double value:mags)total+=value;double denominator=std::max(total,1e-12),entropy=0.0,m1=0.0,m2=0.0;std::size_t peak=0;for(std::size_t i=0;i<mags.size();++i){double probability=mags[i]/denominator;if(probability>1e-12)entropy-=probability*std::log(probability);double frequency=static_cast<double>(i+1);m1+=frequency*probability;m2+=frequency*frequency*probability;if(mags[i]>mags[peak])peak=i;}return V{entropy,m1,m2,static_cast<double>(peak+1),mags[peak]};}
[[maybe_unused]] static double pearson_shift(const V& row){std::size_t n=row.size()-1;if(n<2)return 0.0;double ma=0.0,mb=0.0;for(std::size_t i=0;i<n;++i){ma+=row[i];mb+=row[i+1];}ma/=static_cast<double>(n);mb/=static_cast<double>(n);double sab=0.0,saa=0.0,sbb=0.0;for(std::size_t i=0;i<n;++i){double da=row[i]-ma,db=row[i+1]-mb;sab+=da*db;saa+=da*da;sbb+=db*db;}return saa<=0.0||sbb<=0.0?0.0:sab/std::sqrt(saa*sbb);}
[[maybe_unused]] static V row_diff_stats(const V& row){if(row.size()<2)return V(8,0.0);std::size_t n=row.size()-1;double nf=static_cast<double>(n);V diffs;diffs.reserve(n);for(std::size_t i=0;i<n;++i)diffs.push_back(row[i+1]-row[i]);double mean=0.0;for(double value:diffs)mean+=value;mean/=nf;double m2=0.0,mean_abs=0.0,energy=0.0,mn=std::numeric_limits<double>::infinity(),mx=-std::numeric_limits<double>::infinity();std::size_t signs=0;for(std::size_t i=0;i<n;++i){double value=diffs[i],delta=value-mean;m2+=delta*delta;mean_abs+=std::abs(value);energy+=value*value;if(value<mn)mn=value;if(value>mx)mx=value;if(i>0&&diffs[i-1]*value<0.0)++signs;}return V{mean,std::sqrt(m2/nf),mn,mx,mean_abs/nf,static_cast<double>(signs)/nf,pearson_shift(row),energy/nf};}
[[maybe_unused]] static double row_sum(const double* row,std::size_t n){double value=0.0;for(std::size_t i=0;i<n;++i)value+=row[i];return value;}
[[maybe_unused]] static double row_mean(const double* row,std::size_t n){return row_sum(row,n)/static_cast<double>(n);}
[[maybe_unused]] static double row_l1(const double* row,std::size_t n){double value=0.0;for(std::size_t i=0;i<n;++i)value+=std::abs(row[i]);return value;}
[[maybe_unused]] static double row_l2(const double* row,std::size_t n){double value=0.0;for(std::size_t i=0;i<n;++i)value+=row[i]*row[i];return std::sqrt(value);}
[[maybe_unused]] static double row_min(const double* row,std::size_t n){double value=std::numeric_limits<double>::infinity();for(std::size_t i=0;i<n;++i)if(row[i]<value)value=row[i];return value;}
[[maybe_unused]] static double row_max(const double* row,std::size_t n){double value=-std::numeric_limits<double>::infinity();for(std::size_t i=0;i<n;++i)if(row[i]>value)value=row[i];return value;}
[[maybe_unused]] static double row_std(const double* row,std::size_t n){double mean=row_mean(row,n),value=0.0;for(std::size_t i=0;i<n;++i){double delta=row[i]-mean;value+=delta*delta;}return std::sqrt(value/static_cast<double>(n));}
[[maybe_unused]] static double m_clamp(double value,double lo,double hi){return clampv(value,lo,hi);}
[[maybe_unused]] static double m_div(double left,double right){return std::abs(right)<1e-12?0.0:left/right;}
[[maybe_unused]] static double m_pow(double left,double right){return std::pow(left,right);}
[[maybe_unused]] static double m_sqrt(double value){return std::sqrt(std::fmax(value,0.0));}
[[maybe_unused]] static double m_log1p(double value){return value<=-1.0?0.0:std::log1p(value);}
[[maybe_unused]] static double m_sin(double value){return std::sin(value);}
[[maybe_unused]] static double m_cos(double value){return std::cos(value);}
[[maybe_unused]] static double m_tanh(double value){return std::tanh(value);}
[[maybe_unused]] static double m_sigmoid(double value){return 1.0/(1.0+std::exp(-clampv(value,-30.0,30.0)));}
[[maybe_unused]] static double m_clip(double value,double a,double b){double lo=std::fmin(a,b),hi=std::fmax(a,b);if(std::isnan(lo)||std::isnan(hi))throw MathDomainError{};return clampv(value,lo,hi);}
[[maybe_unused]] static double m_exp(double value){return std::exp(clampv(value,-30.0,30.0));}
[[maybe_unused]] static double m_log(double value){return value<=1e-300?0.0:std::log(value);}
[[maybe_unused]] static double m_asinh(double value){double a=std::abs(value),s=value>0.0?1.0:(value<0.0?-1.0:0.0);return s*std::log(a+std::sqrt(value*value+1.0));}
[[maybe_unused]] static double m_atan(double value){return std::atan(value);}
[[maybe_unused]] static double m_erf(double value){double a=std::abs(value),s=value>0.0?1.0:(value<0.0?-1.0:0.0),t=1.0/(1.0+0.3275911*a);double poly=t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))));return s*(1.0-poly*std::exp(-(a*a)));}
[[maybe_unused]] static double m_sign(double value){return value>0.0?1.0:(value<0.0?-1.0:0.0);}
[[maybe_unused]] static double m_round(double value){return value>=0.0?std::floor(value+0.5):std::ceil(value-0.5);}
[[maybe_unused]] static double math_safe(double value){return std::isfinite(value)?clampv(value,-1e12,1e12):0.0;}

static double head_0(const V& x){const V means=V{f64_from_bits(0x4000000000000000ULL),f64_from_bits(0x0000000000000000ULL),f64_from_bits(0x4008000000000000ULL),f64_from_bits(0x0000000000000000ULL)};const V scales=V{f64_from_bits(0x3ff6a09e667f3bcdULL),f64_from_bits(0x3e112e0be826d695ULL),f64_from_bits(0x3ff6a09e667f3bcdULL),f64_from_bits(0x3e112e0be826d695ULL)};const V weights=V{f64_from_bits(0x3fd999999999999aULL),f64_from_bits(0x3fc1ec367ecf3c18ULL),f64_from_bits(0x0000000000000000ULL),f64_from_bits(0x3fc1ec367ecf3bfdULL),f64_from_bits(0x0000000000000000ULL)};double value=weights[0];for(std::size_t j=0;j<means.size();++j){const std::size_t source=j;if(source<x.size())value+=weights[j+1]*clampv((x[source]-means[j])/scales[j],f64_from_bits(0xc034000000000000ULL),f64_from_bits(0x4034000000000000ULL));}return clampv(value,-60.0,60.0);}

static bool encode_row(const double* raw,const std::uint8_t* missing,V& x,V& mask){
  x.clear(); mask.clear();
  const bool absent_0=missing[0]!=0||!std::isfinite(raw[0]); const double value_0=absent_0?f64_from_bits(0x4000000000000000ULL):raw[0]; x.push_back(value_0); mask.push_back(absent_0?1.0:0.0);
  x.push_back(absent_0?1.0:0.0); mask.push_back(0.0);
  const bool absent_1=missing[1]!=0||!std::isfinite(raw[1]); const double value_1=absent_1?f64_from_bits(0x4008000000000000ULL):raw[1]; x.push_back(value_1); mask.push_back(absent_1?1.0:0.0);
  x.push_back(absent_1?1.0:0.0); mask.push_back(0.0);
  return true;
}

static V transform(V x,V mask){
  (void)mask;
  return x;
}

static double predict_one(V x,V mask){x=transform(std::move(x),std::move(mask));double value=head_0(x);
  value=finitev(value);
  return value;
}

extern "C" std::size_t jope_fitted_columns() noexcept { return 2; }
extern "C" int jope_fitted_predict(
    const double* values,std::size_t values_len,
    const std::uint8_t* missing,std::size_t missing_len,
    std::size_t rows,std::size_t columns,
    double* output,std::size_t output_len) noexcept {
  if(columns!=2)return 1;
  if(rows>std::numeric_limits<std::size_t>::max()/columns)return 2;
  const std::size_t cells=rows*columns;
  if(values_len!=cells||missing_len!=cells||output_len<rows)return 3;
  if(rows!=0&&(!values||!missing||!output))return 4;
  try {
    V x,mask;
    for(std::size_t row=0;row<rows;++row){
      const std::size_t offset=row*columns;
      if(!encode_row(values+offset,missing+offset,x,mask))return 5;
      output[row]=predict_one(x,mask);
    }
    return 0;
  } catch(...) { return 6; }
}
