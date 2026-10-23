#include "model.hpp"
#include <cassert>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#if JOPE_TEST_DEVICE
#include <cuda_runtime.h>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif
static bool same(const std::vector<double>& a,const std::vector<double>& b){
 return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*8)==0);
}
#endif
int main(){
#if JOPE_TEST_DEVICE
 assert(jope_cuda::Model::cuda_available());
 jope_cuda::Model m;m.fit({{0,1},{1,2},{2,0},{3,2}},{1,2,4,7});
 cudaStream_t stream=nullptr;assert(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
 for(std::size_t n: {0,1,127,128,129,4097}){
  jope_cuda::Rows rows;for(std::size_t i=0;i<n;++i)rows.push_back({static_cast<double>(i%13)-4.0,static_cast<double>(i%7)});
  if(n)rows[0]={std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()};
  assert(same(m.predict(rows),m.predict_cuda(rows)));
  assert(same(m.predict(rows),m.predict_cuda(rows,stream)));
 }
 const auto mode=std::fegetround();assert(std::fesetround(FE_DOWNWARD)==0);
 bool rejected=false;try{m.predict_cuda({{1,2}});}catch(const jope_cuda::Error& e){rejected=e.status==6;}
 assert(std::fesetround(mode)==0);assert(rejected);
#if defined(__SSE2__)
 const auto csr=_mm_getcsr();_mm_setcsr(csr|0x8040u);
 rejected=false;try{m.predict_cuda({{1,2}});}catch(const jope_cuda::Error& e){rejected=e.status==6;}
 _mm_setcsr(csr);assert(rejected);
#endif
 assert(cudaStreamDestroy(stream)==cudaSuccess);
 std::cout<<"CUDA device contract passed\n";
#else
 std::cout<<"CUDA device contract not requested\n";
#endif
}
