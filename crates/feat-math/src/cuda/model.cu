// Fit and host inference use the companion native library.
#include "model.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <float.h>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif
#if defined(__CUDACC__) && !defined(JOPE_CUDA_HOST_ONLY)
#include <cuda_runtime.h>
#define JOPE_HAS_CUDA 1
#else
#define JOPE_HAS_CUDA 0
#endif
static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024,
              "64-bit IEEE double required");
extern "C" {
std::uint32_t jope_cuda_native_abi_version();
std::int32_t jope_cuda_native_fit(const double*,std::size_t,const double*,std::size_t,
                                 std::size_t,std::size_t,std::uint64_t*);
std::int32_t jope_cuda_native_predict(std::uint64_t,const double*,std::size_t,
                                     std::size_t,std::size_t,double*,std::size_t);
std::int32_t jope_cuda_native_score(std::uint64_t,const double*,std::size_t,
                                   const double*,std::size_t,std::size_t,std::size_t,double*);
std::int32_t jope_cuda_native_free(std::uint64_t);
std::int32_t jope_cuda_native_open(std::uint64_t*);
std::int32_t jope_cuda_native_linear_width(std::uint64_t,std::size_t*);
std::int32_t jope_cuda_native_linear(std::uint64_t,const double*,std::size_t,
                                    std::size_t,std::size_t,double*,std::size_t,double*,std::size_t);
std::int32_t jope_cuda_native_text(std::uint64_t,const unsigned char*,std::size_t,double*,std::size_t);
}
namespace jope_cuda {
namespace {
void checked(int status) {
    if(status) throw Error(status, "native model rejected operation (status "+std::to_string(status)+")");
}
std::size_t product(std::size_t a,std::size_t b) {
    if(b && a>std::numeric_limits<std::size_t>::max()/b) throw Error(1,"shape overflow");
    const auto n=a*b;
    if(n>static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double))
        throw Error(1,"buffer exceeds addressable size");
    return n;
}
std::vector<double> flatten(const Rows& rows,std::size_t width) {
    if(!width) throw Error(1,"empty feature width");
    const auto n=product(rows.size(),width);
    for(const auto& row:rows) if(row.size()!=width) throw Error(1,"row width mismatch");
    std::vector<double> values;values.reserve(n);
    for(const auto& row:rows) values.insert(values.end(),row.begin(),row.end());
    return values;
}
const std::vector<std::string>& headers() {
    static const std::vector<std::string> value=__BOUND_HEADERS__;
    return value;
}
std::string quote(const std::string& s) {
    std::string value="\"";
    for(char c:s){if(c=='"')value+='"';value+=c;}
    return value+'"';
}
#if JOPE_HAS_CUDA
void cuda_checked(cudaError_t status) {
    if(status!=cudaSuccess)throw Error(6,std::string("CUDA: ")+cudaGetErrorString(status));
}
class DeviceBuffer {
public:
    double* data=nullptr;
    explicit DeviceBuffer(std::size_t n){if(n)cuda_checked(cudaMalloc(reinterpret_cast<void**>(&data),product(n,sizeof(double))));}
    ~DeviceBuffer(){if(data)cudaFree(data);}
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
};
// Keep pageable host storage alive until all queued copies and work finish,
// including while a CUDA exception unwinds the call.
class StreamFence {
    cudaStream_t stream_;
    bool pending_=true;
public:
    explicit StreamFence(cudaStream_t stream):stream_(stream){}
    ~StreamFence() noexcept{if(pending_)cudaStreamSynchronize(stream_);}
    void finish(){cuda_checked(cudaStreamSynchronize(stream_));pending_=false;}
    StreamFence(const StreamFence&)=delete;
    StreamFence& operator=(const StreamFence&)=delete;
};
// One thread per row, with sequential correctly rounded multiply/add. No
// reassociation, atomics, mixed precision, or device transcendental functions.
__global__ void linear_head(const double* x,const double* w,double* out,std::size_t n,std::size_t p) {
    const auto stride=static_cast<std::size_t>(blockDim.x)*gridDim.x;
    for(auto row=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;row<n;row+=stride){
        double value=w[0];
        for(std::size_t j=0;j<p;++j)value=__dadd_rn(value,__dmul_rn(w[j+1],x[row*p+j]));
        out[row]=value < -60.0 ? -60.0 : (value > 60.0 ? 60.0 : value);
    }
}
#endif
}
void Model::check() const {
    if(!handle_)throw Error(2,"fit or load required");
    if(jope_cuda_native_abi_version()!=1)throw Error(4,"native ABI mismatch");
}
Model::~Model() noexcept{close();}
Model::Model(Model&& other) noexcept:handle_(std::exchange(other.handle_,0)),width_(std::exchange(other.width_,0)){}
Model& Model::operator=(Model&& other) noexcept{
    if(this!=&other){close();handle_=std::exchange(other.handle_,0);width_=std::exchange(other.width_,0);}return *this;
}
void Model::close() noexcept{if(handle_)jope_cuda_native_free(std::exchange(handle_,0));width_=0;}
Model& Model::fit(const Rows& rows,const std::vector<double>& targets){
    if(jope_cuda_native_abi_version()!=1)throw Error(4,"native ABI mismatch");
    if(rows.empty()||targets.size()!=rows.size())throw Error(1,"invalid training rows or targets");
    const auto width=rows.front().size();const auto x=flatten(rows,width);
    std::uint64_t handle=0;
    checked(jope_cuda_native_fit(x.data(),x.size(),targets.data(),targets.size(),rows.size(),width,&handle));
    close();handle_=handle;width_=width;return *this;
}
Model Model::load(){
    Model m;checked(jope_cuda_native_open(&m.handle_));m.width_=headers().size();return m;
}
std::vector<double> Model::predict(const Rows& rows) const{
    check();if(!headers().empty())throw Error(5,"use predict_strings for fitted artifacts");
    const auto x=flatten(rows,width_);std::vector<double> out(rows.size());
    checked(jope_cuda_native_predict(handle_,x.data(),x.size(),rows.size(),width_,out.data(),out.size()));return out;
}
double Model::score(const Rows& rows,const std::vector<double>& targets) const{
    check();if(!headers().empty())throw Error(5,"numeric score requires a training recipe");
    if(rows.empty()||targets.size()!=rows.size())throw Error(1,"invalid score targets");
    const auto x=flatten(rows,width_);double out=0;
    checked(jope_cuda_native_score(handle_,x.data(),x.size(),targets.data(),targets.size(),rows.size(),width_,&out));return out;
}
std::vector<double> Model::predict_csv(const std::string& csv,std::size_t expected_rows) const{
    check();product(expected_rows,1);std::vector<double> out(expected_rows);
    checked(jope_cuda_native_text(handle_,reinterpret_cast<const unsigned char*>(csv.data()),csv.size(),out.data(),out.size()));return out;
}
std::vector<double> Model::predict_strings(const StringRows& rows) const{
    check();if(headers().empty())throw Error(5,"string rows require a fitted artifact");
    std::string csv;
    for(const auto& cell:headers()){if(!csv.empty())csv+=',';csv+=quote(cell);}csv+='\n';
    for(const auto& row:rows){
        if(row.size()!=headers().size())throw Error(1,"string row width mismatch");
        for(std::size_t j=0;j<row.size();++j){if(j)csv+=',';csv+=quote(row[j]);}csv+='\n';
    }
    return predict_csv(csv,rows.size());
}
bool Model::supports_cuda() const{
    check();std::size_t width=0;const auto status=jope_cuda_native_linear_width(handle_,&width);
    if(status==5)return false;
    checked(status);return true;
}
bool Model::cuda_available() noexcept{
#if JOPE_HAS_CUDA
    int count=0;return cudaGetDeviceCount(&count)==cudaSuccess && count>0;
#else
    return false;
#endif
}
std::vector<double> Model::predict_cuda(const Rows& rows,void* stream) const{
    check();const auto x=flatten(rows,width_);
    std::size_t p=0;checked(jope_cuda_native_linear_width(handle_,&p));
    if(p==std::numeric_limits<std::size_t>::max())throw Error(1,"coefficient width overflow");
#if JOPE_HAS_CUDA
    // No fallback on an unavailable device, allocation, stream, or launch error.
    if(std::fegetround()!=FE_TONEAREST)throw Error(6,"round-to-nearest mode required");
#if defined(__SSE2__)
    if((_mm_getcsr() & 0x6000u)!=0)throw Error(6,"host SSE round-to-nearest mode required");
    if((_mm_getcsr() & 0x8040u)!=0)throw Error(6,"host denormal flushing must be disabled");
#endif
    int device=0;cuda_checked(cudaGetDevice(&device));cudaDeviceProp prop{};
    cuda_checked(cudaGetDeviceProperties(&prop,device));
    if(prop.major<2)throw Error(6,"device double precision is unavailable");
    const auto lane=static_cast<cudaStream_t>(stream);
    if(rows.empty()){cuda_checked(cudaStreamSynchronize(lane));return {};}
    std::vector<double> features(product(rows.size(),p)),weights(p+1),out(rows.size());
    checked(jope_cuda_native_linear(handle_,x.data(),x.size(),rows.size(),width_,features.data(),features.size(),weights.data(),weights.size()));
    DeviceBuffer dx(features.size()),dw(weights.size()),dy(rows.size());
    // Default-stream staging need not finish DMA before a nonblocking stream
    // starts. Queue transfers and work together, then wait before buffers die.
    StreamFence fence(lane);
    if(!features.empty())cuda_checked(cudaMemcpyAsync(dx.data,features.data(),features.size()*sizeof(double),cudaMemcpyHostToDevice,lane));
    cuda_checked(cudaMemcpyAsync(dw.data,weights.data(),weights.size()*sizeof(double),cudaMemcpyHostToDevice,lane));
    const auto blocks=static_cast<unsigned>(std::min<std::size_t>(1+(rows.size()-1)/128,65535));
    linear_head<<<blocks,128,0,lane>>>(dx.data,dw.data,dy.data,rows.size(),p);
    cuda_checked(cudaGetLastError());
    cuda_checked(cudaMemcpyAsync(out.data(),dy.data,out.size()*sizeof(double),cudaMemcpyDeviceToHost,lane));
    fence.finish();
    for(double v:out)if(!std::isfinite(v))throw Error(6,"nonfinite device prediction");
    return out;
#else
    (void)stream;(void)x;throw Error(6,"model was built without CUDA device support");
#endif
}
} // namespace jope_cuda
