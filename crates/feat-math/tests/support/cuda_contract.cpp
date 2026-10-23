#include "model.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <type_traits>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif
static bool same(const std::vector<double>& a,const std::vector<double>& b){
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*8)==0);
}
template<class F> static void rejected(F f,int status){bool ok=false;try{f();}catch(const jope_cuda::Error& e){ok=e.status==status;}assert(ok);}
int main(){
    using jope_cuda::Model;using jope_cuda::Rows;
    static_assert(!std::is_copy_constructible<Model>::value);
    static_assert(std::is_nothrow_move_constructible<Model>::value);
    Model m;rejected([&]{m.predict({{0,1}});},2);rejected([&]{m.supports_cuda();},2);
    rejected([&]{Model::load();},5);
    rejected([&]{m.fit({},{});},1);rejected([&]{m.fit({{}},{1});},1);
    rejected([&]{m.fit({{0,1},{2}}, {1,2});},1);
    Rows train{{0,1},{1,2},{2,0},{3,2}},test{{0.5,1},{2.5,0}};std::vector<double> y{1,2,4,7};
    m.fit(train,y);const auto before=m.predict(test);assert(m.predict({}).empty());
    assert(m.supports_cuda());assert(std::isfinite(m.score(train,y)));
    rejected([&]{m.predict({{1}});},1);rejected([&]{m.score(train,{1});},1);
    rejected([&]{m.fit(train,{1});},1);assert(same(before,m.predict(test)));
    rejected([&]{m.predict_strings({{"1","2"}});},5);
    rejected([&]{m.predict_csv("a,b\n1,2\n",1);},5);
    auto task=std::async(std::launch::async,[&]{return m.predict(test);});
    assert(same(before,m.predict(test)));assert(same(before,task.get()));
    Model other;other.fit(train,{9,7,5,3});assert(!same(before,other.predict(test)));
    Model moved(std::move(m));rejected([&]{m.predict(test);},2);assert(same(before,moved.predict(test)));
    other=std::move(moved);assert(same(before,other.predict(test)));rejected([&]{moved.predict(test);},2);
    auto* same_model=&other;other=std::move(*same_model);assert(same(before,other.predict(test)));
#if defined(JOPE_CUDA_COMPILED) && defined(__SSE2__)
    // MXCSR can disagree with the x87 rounding mode read by fegetround().
    // This refusal must happen before device lookup, including on a CPU host.
    const auto csr=_mm_getcsr();_mm_setcsr((csr & ~0x6000u)|0x2000u);
    bool rounding_refused=false;
    try{other.predict_cuda(test);}catch(const jope_cuda::Error& e){
        rounding_refused=e.status==6 && std::string(e.what()).find("round-to-nearest")!=std::string::npos;
    }
    _mm_setcsr(csr);assert(rounding_refused);
#endif
#ifndef JOPE_CUDA_COMPILED
    assert(!Model::cuda_available());rejected([&]{other.predict_cuda(test);},6);
#endif
    other.close();other.close();rejected([&]{other.predict(test);},2);
    other.fit(train,y);assert(same(before,other.predict(test)));
    std::cout<<"CUDA C++ lifecycle passed\n";
}
